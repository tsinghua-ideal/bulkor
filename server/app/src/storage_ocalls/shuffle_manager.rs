use crate::storage_ocalls::{get_sts_ptr, set_ptr, UntrustedAllocation};
use nix::fcntl::{posix_fadvise, PosixFadviseAdvice};
use std::fs::{self, remove_file, rename, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::{fs::FileExt, io::AsRawFd};
use std::path::{Path, PathBuf};
use std::sync::{atomic::Ordering, Arc, Mutex};

// Should be consistent with oram_storage/shuffle_manager.rs
const NUM_THREADS: usize = 8;

pub struct ShuffleManager {
    pub storage_id: u64,
    data_size: usize, //Bucket data
    meta_size: usize, //Bucket meta (not block meta) plus extra
    num_bins: usize,
    num_bins_on_disk: usize,
    //after separation or before combination
    //denoted as idle bins
    data_bin_buf: Vec<Arc<Mutex<(Vec<u8>, Vec<u8>, Vec<u8>)>>>,
    meta_bin_buf: Vec<Arc<Mutex<(Vec<u8>, Vec<u8>, Vec<u8>)>>>,
    data_bin_size: u64,
    meta_bin_size: u64,
    data_bin_file: File,
    meta_bin_file: File,
    //bins used during shuffle
    //denoted as work bins
    //bin format: (data, meta, nonce, hash)
    src_bins: Vec<Arc<Mutex<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)>>>,
    dst_bins: Vec<Arc<Mutex<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)>>>,
    src_bin_size: u64,
    dst_bin_size: u64,
    src_bin_file: File,
    dst_bin_file: File,
    //temp buf
    pub tmp_buf: Vec<Arc<Mutex<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)>>>,
}

impl ShuffleManager {
    pub fn new(
        storage_id: u64,
        data_size: usize,
        meta_size: usize,
        num_bins: usize,
        in_memory_ratio: usize,
    ) -> Self {
        let num_bins_on_disk = num_bins - ((num_bins - 1) / in_memory_ratio + 1);

        let mut data_bin_buf = Vec::new();
        data_bin_buf.resize_with(num_bins - num_bins_on_disk, || {
            Arc::new(Mutex::new((Vec::new(), Vec::new(), Vec::new())))
        });
        let mut meta_bin_buf = Vec::new();
        meta_bin_buf.resize_with(num_bins - num_bins_on_disk, || {
            Arc::new(Mutex::new((Vec::new(), Vec::new(), Vec::new())))
        });
        let mut src_bins = Vec::new();
        src_bins.resize_with(num_bins - num_bins_on_disk, || {
            Arc::new(Mutex::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())))
        });
        let mut dst_bins = Vec::new();
        dst_bins.resize_with(num_bins - num_bins_on_disk, || {
            Arc::new(Mutex::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())))
        });

        let data_bin_file_name = PathBuf::from("shuffle_data_bin");
        let meta_bin_file_name = PathBuf::from("shuffle_meta_bin");
        let data_bin_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&data_bin_file_name)
            .unwrap();
        let meta_bin_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&meta_bin_file_name)
            .unwrap();

        let src_bin_file_name = PathBuf::from("shuffle_src_bin");
        let dst_bin_file_name = PathBuf::from("shuffle_dst_bin");
        let src_bin_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&src_bin_file_name)
            .unwrap();
        let dst_bin_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&dst_bin_file_name)
            .unwrap();

        let mut tmp_buf = Vec::new();
        tmp_buf.resize_with(NUM_THREADS, || {
            Arc::new(Mutex::new((Vec::new(), Vec::new(), Vec::new(), Vec::new())))
        });

        ShuffleManager {
            storage_id,
            data_size,
            meta_size,
            num_bins,
            num_bins_on_disk,
            data_bin_buf,
            meta_bin_buf,
            data_bin_size: 0,
            meta_bin_size: 0,
            data_bin_file,
            meta_bin_file,
            src_bins,
            dst_bins,
            src_bin_size: 0,
            dst_bin_size: 0,
            src_bin_file,
            dst_bin_file,
            tmp_buf,
        }
    }

    pub fn clear_content(&mut self) {
        //sometimes the user-space allocator would not return the memory to OS, causing the memory overflow.
        let release_mem = unsafe { libc::malloc_trim(0) };
        println!("release the memory successfully? {:?}", release_mem == 1);
    }

    pub fn pull_buckets(&mut self, b_idx: usize, e_idx: usize, data: &mut [u8], meta: &mut [u8]) {
        let storage = unsafe {
            core::mem::transmute::<_, *mut UntrustedAllocation>(self.storage_id)
                .as_mut()
                .unwrap()
        };

        let data_size = data.len();
        let meta_size = meta.len();
        assert_eq!(data_size / self.data_size, meta_size / self.meta_size);
        let data_item_size = self.data_size;
        let meta_item_size = self.meta_size;
        let count_in_mem = storage.count_in_mem;

        //since data and meta is not loaded into memory, we read them directly from disk
        for (count, idx) in (b_idx..e_idx).into_iter().enumerate() {
            let p = get_sts_ptr(&storage.ptrs, idx).2;
            if idx < count_in_mem {
                storage
                    .data_file
                    .read_exact_at(
                        &mut data[data_item_size * count..data_item_size * (count + 1)],
                        (data_item_size * (idx * 2 + p)) as u64,
                    )
                    .unwrap();
            } else {
                storage
                    .data_file
                    .read_exact_at(
                        &mut data[data_item_size * count..data_item_size * (count + 1)],
                        (data_item_size * (count_in_mem * 2 + (idx - count_in_mem) * 3 + p)) as u64,
                    )
                    .unwrap();
            }
        }

        for (count, idx) in (b_idx..e_idx).into_iter().enumerate() {
            let p = get_sts_ptr(&storage.ptrs, idx).2;
            if idx < count_in_mem {
                storage
                    .meta_file
                    .read_exact_at(
                        &mut meta[meta_item_size * count..meta_item_size * (count + 1)],
                        (meta_item_size * (idx * 2 + p)) as u64,
                    )
                    .unwrap();
            } else {
                storage
                    .meta_file
                    .read_exact_at(
                        &mut meta[meta_item_size * count..meta_item_size * (count + 1)],
                        (meta_item_size * (count_in_mem * 2 + (idx - count_in_mem) * 3 + p)) as u64,
                    )
                    .unwrap();
            }
        }
        posix_fadvise(
            storage.data_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
        posix_fadvise(
            storage.meta_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
    }

    pub fn pull_bin(
        &self,
        tid: usize,
        cur_bin_num: usize,
        bin_type: u8,
        bin_size: &mut usize,
        data_item_size: usize,
        meta_item_size: usize,
        has_data: bool,
        has_meta: bool,
        nonce_size: usize,
        hash_size: usize,
    ) {
        assert!(cur_bin_num < self.num_bins);
        let mut nonce = vec![0; nonce_size];
        let mut hash = vec![0; hash_size];

        if bin_type == 0 {
            assert!(!has_meta && has_data || has_meta && !has_data);
            if cur_bin_num < self.num_bins_on_disk {
                if has_data {
                    let mut offset = self.data_bin_size * cur_bin_num as u64;
                    let mut data_size_buf = [0u8; 8];
                    self.data_bin_file
                        .read_exact_at(&mut data_size_buf, offset)
                        .unwrap();
                    offset += 8;
                    let actual_data_size = usize::from_ne_bytes(data_size_buf);
                    *bin_size = actual_data_size / data_item_size;
                    let mut data = vec![0; actual_data_size];
                    self.data_bin_file.read_exact_at(&mut data, offset).unwrap();
                    offset += actual_data_size as u64;
                    self.data_bin_file
                        .read_exact_at(&mut nonce, offset)
                        .unwrap();
                    offset += nonce_size as u64;
                    self.data_bin_file.read_exact_at(&mut hash, offset).unwrap();
                    *self.tmp_buf[tid].lock().unwrap() = (data, Vec::new(), nonce, hash);
                } else {
                    let mut offset = self.meta_bin_size * cur_bin_num as u64;
                    let mut meta_size_buf = [0u8; 8];
                    self.meta_bin_file
                        .read_exact_at(&mut meta_size_buf, offset)
                        .unwrap();
                    offset += 8;
                    let actual_meta_size = usize::from_ne_bytes(meta_size_buf);
                    *bin_size = actual_meta_size / meta_item_size;
                    let mut meta = vec![0; actual_meta_size];
                    self.meta_bin_file.read_exact_at(&mut meta, offset).unwrap();
                    offset += actual_meta_size as u64;
                    self.meta_bin_file
                        .read_exact_at(&mut nonce, offset)
                        .unwrap();
                    offset += nonce_size as u64;
                    self.meta_bin_file.read_exact_at(&mut hash, offset).unwrap();
                    *self.tmp_buf[tid].lock().unwrap() = (Vec::new(), meta, nonce, hash);
                }
            } else {
                if has_data {
                    let (data, nonce, hash) = std::mem::take(
                        &mut *self.data_bin_buf[cur_bin_num - self.num_bins_on_disk]
                            .lock()
                            .unwrap(),
                    );
                    let actual_data_size = data.len();
                    *bin_size = actual_data_size / data_item_size;
                    *self.tmp_buf[tid].lock().unwrap() = (data, Vec::new(), nonce, hash);
                } else {
                    let (meta, nonce, hash) = std::mem::take(
                        &mut *self.meta_bin_buf[cur_bin_num - self.num_bins_on_disk]
                            .lock()
                            .unwrap(),
                    );
                    let actual_meta_size = meta.len();
                    *bin_size = actual_meta_size / meta_item_size;
                    *self.tmp_buf[tid].lock().unwrap() = (Vec::new(), meta, nonce, hash);
                }
            }
        } else if bin_type == 1 {
            assert!(has_meta);
            if cur_bin_num < self.num_bins_on_disk {
                let mut offset = self.src_bin_size * cur_bin_num as u64;
                let mut data_size_buf = [0u8; 8];
                let mut meta_size_buf = [0u8; 8];
                self.src_bin_file
                    .read_exact_at(&mut data_size_buf, offset)
                    .unwrap();
                offset += 8;
                let actual_data_size = usize::from_ne_bytes(data_size_buf);
                self.src_bin_file
                    .read_exact_at(&mut meta_size_buf, offset)
                    .unwrap();
                offset += 8;
                let actual_meta_size = usize::from_ne_bytes(meta_size_buf);
                *bin_size = actual_meta_size / meta_item_size;
                assert!(has_data == (actual_data_size != 0));
                assert!(has_meta == (actual_meta_size != 0));
                let mut data = vec![0; actual_data_size];
                self.src_bin_file.read_exact_at(&mut data, offset).unwrap();
                offset += actual_data_size as u64;
                let mut meta = vec![0; actual_meta_size];
                self.src_bin_file.read_exact_at(&mut meta, offset).unwrap();
                offset += actual_meta_size as u64;
                self.src_bin_file.read_exact_at(&mut nonce, offset).unwrap();
                offset += nonce_size as u64;
                self.src_bin_file.read_exact_at(&mut hash, offset).unwrap();
                *self.tmp_buf[tid].lock().unwrap() = (data, meta, nonce, hash);
            } else {
                let t = std::mem::take(
                    &mut *self.src_bins[cur_bin_num - self.num_bins_on_disk]
                        .lock()
                        .unwrap(),
                );
                *bin_size = t.1.len() / meta_item_size;
                *self.tmp_buf[tid].lock().unwrap() = t;
            }
        } else {
            unreachable!()
        }
        posix_fadvise(
            self.src_bin_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
        posix_fadvise(
            self.data_bin_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
        posix_fadvise(
            self.meta_bin_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
    }

    pub fn push_buckets(&mut self, tid: usize, b_idx: usize, e_idx: usize) {
        let storage = unsafe {
            core::mem::transmute::<_, *mut UntrustedAllocation>(self.storage_id)
                .as_mut()
                .unwrap()
        };

        let (data, meta, _, _) = std::mem::take(&mut *self.tmp_buf[tid].lock().unwrap());
        let data_size = data.len();
        let meta_size = meta.len();
        assert_eq!(data_size / self.data_size, meta_size / self.meta_size);
        let data_item_size = self.data_size;
        let meta_item_size = self.meta_size;
        let count_in_mem = storage.count_in_mem;
        let data_mut = unsafe {
            core::slice::from_raw_parts_mut(storage.data_pointer, 2 * count_in_mem * data_item_size)
        };
        let meta_mut = unsafe {
            core::slice::from_raw_parts_mut(storage.meta_pointer, 2 * count_in_mem * meta_item_size)
        };

        for (count, idx) in (b_idx..e_idx).into_iter().enumerate() {
            let (sts_on_disk, _, mut p) = get_sts_ptr(&storage.ptrs, idx);
            //every bucket write once
            assert!(!sts_on_disk);
            if idx < count_in_mem {
                p = p ^ 1;
                (&mut data_mut[data_item_size * (idx * 2 + p)..data_item_size * (idx * 2 + p + 1)])
                    .copy_from_slice(&data[data_item_size * count..data_item_size * (count + 1)]);
                // storage
                //     .data_file
                //     .write_all_at(
                //         &data[data_item_size * count..data_item_size * (count + 1)],
                //         (data_item_size * (idx * 2 + p)) as u64,
                //     )
                //     .unwrap();
            } else {
                p = (p + 1) % 3;
                storage
                    .data_file
                    .write_all_at(
                        &data[data_item_size * count..data_item_size * (count + 1)],
                        (data_item_size * (count_in_mem * 2 + (idx - count_in_mem) * 3 + p)) as u64,
                    )
                    .unwrap();
            }
            set_ptr(&mut storage.ptrs, idx, p);
        }

        for (count, idx) in (b_idx..e_idx).into_iter().enumerate() {
            let (sts_on_disk, _, p) = get_sts_ptr(&storage.ptrs, idx);
            //already set above
            assert!(sts_on_disk);
            if idx < count_in_mem {
                (&mut meta_mut[meta_item_size * (idx * 2 + p)..meta_item_size * (idx * 2 + p + 1)])
                    .clone_from_slice(&meta[meta_item_size * count..meta_item_size * (count + 1)]);
                (&mut meta_mut[meta_item_size * (idx * 2 + (p ^ 1))
                    ..meta_item_size * (idx * 2 + (p ^ 1) + 1)])
                    .copy_from_slice(&vec![0; meta_item_size]);
                // storage
                //     .meta_file
                //     .write_all_at(
                //         &meta[meta_item_size * count..meta_item_size * (count + 1)],
                //         (meta_item_size * (idx * 2 + p)) as u64,
                //     )
                //     .unwrap();
            } else {
                storage
                    .meta_file
                    .write_all_at(
                        &meta[meta_item_size * count..meta_item_size * (count + 1)],
                        (meta_item_size * (count_in_mem * 2 + (idx - count_in_mem) * 3 + p)) as u64,
                    )
                    .unwrap();
            }
        }
        storage.data_file.sync_all().unwrap();
        storage.meta_file.sync_all().unwrap();
        posix_fadvise(
            storage.data_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
        posix_fadvise(
            storage.meta_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
    }

    pub fn push_bin(&mut self, tid: usize, cur_bin_num: usize, bin_type: u8) {
        assert!(cur_bin_num < self.num_bins);
        let (data, meta, nonce, hash) = std::mem::take(&mut *self.tmp_buf[tid].lock().unwrap());
        let data_size = data.len();
        let meta_size = meta.len();
        if bin_type == 0 {
            assert!(meta_size == 0 && data_size > 0 || meta_size > 0 && data_size == 0);
            if cur_bin_num < self.num_bins_on_disk {
                if data_size > 0 {
                    let mut offset = self.data_bin_size * cur_bin_num as u64;
                    self.data_bin_file
                        .write_at(&(data_size).to_ne_bytes(), offset)
                        .unwrap();
                    offset += 8;
                    self.data_bin_file.write_at(&data, offset).unwrap();
                    offset += data.len() as u64;
                    self.data_bin_file.write_at(&nonce, offset).unwrap();
                    offset += nonce.len() as u64;
                    self.data_bin_file.write_at(&hash, offset).unwrap();
                } else {
                    let mut offset = self.meta_bin_size * cur_bin_num as u64;
                    self.meta_bin_file
                        .write_at(&(meta_size).to_ne_bytes(), offset)
                        .unwrap();
                    offset += 8;
                    self.meta_bin_file.write_at(&meta, offset).unwrap();
                    offset += meta.len() as u64;
                    self.meta_bin_file.write_at(&nonce, offset).unwrap();
                    offset += nonce.len() as u64;
                    self.meta_bin_file.write_at(&hash, offset).unwrap();
                }
            } else {
                if data_size > 0 {
                    *self.data_bin_buf[cur_bin_num - self.num_bins_on_disk]
                        .lock()
                        .unwrap() = (data, nonce, hash);
                } else {
                    *self.meta_bin_buf[cur_bin_num - self.num_bins_on_disk]
                        .lock()
                        .unwrap() = (meta, nonce, hash);
                }
            }
        } else if bin_type == 1 {
            if cur_bin_num < self.num_bins_on_disk {
                //if dst_bin_file_idx is reset in bin_switch, the assertion should be true
                //assert_eq!(self.dst_bin_file_idx[cur_bin_num], u64::MAX);
                let mut offset = self.dst_bin_size * cur_bin_num as u64;
                self.dst_bin_file
                    .write_at(&(data_size).to_ne_bytes(), offset)
                    .unwrap();
                offset += 8;
                self.dst_bin_file
                    .write_at(&(meta_size).to_ne_bytes(), offset)
                    .unwrap();
                offset += 8;
                self.dst_bin_file.write_at(&data, offset).unwrap();
                offset += data.len() as u64;
                self.dst_bin_file.write_at(&meta, offset).unwrap();
                offset += meta.len() as u64;
                self.dst_bin_file.write_at(&nonce, offset).unwrap();
                offset += nonce.len() as u64;
                self.dst_bin_file.write_at(&hash, offset).unwrap();
            } else {
                *self.dst_bins[cur_bin_num - self.num_bins_on_disk]
                    .lock()
                    .unwrap() = (data, meta, nonce, hash);
            }
        } else {
            unreachable!()
        }
        self.dst_bin_file.sync_all().unwrap();
        self.data_bin_file.sync_all().unwrap();
        self.meta_bin_file.sync_all().unwrap();
        posix_fadvise(
            self.dst_bin_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
        posix_fadvise(
            self.data_bin_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
        posix_fadvise(
            self.meta_bin_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
    }

    pub fn bin_switch(&mut self) {
        //reset file pointer
        rename("shuffle_src_bin", "shuffle_src_bin_del").unwrap();
        rename("shuffle_dst_bin", "shuffle_src_bin").unwrap();
        std::mem::swap(&mut self.src_bin_file, &mut self.dst_bin_file);
        self.dst_bin_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&"shuffle_dst_bin")
            .unwrap();
        remove_file("shuffle_src_bin_del").unwrap();

        for idx in 0..self.num_bins {
            if idx >= self.num_bins_on_disk {
                *self.src_bins[idx - self.num_bins_on_disk].lock().unwrap() = std::mem::take(
                    &mut *self.dst_bins[idx - self.num_bins_on_disk].lock().unwrap(),
                );
            }
        }
    }

    pub fn set_fixed_bin_size(
        &mut self,
        data_bin_size: u64,
        meta_bin_size: u64,
        src_bin_size: u64,
        dst_bin_size: u64,
    ) {
        self.data_bin_size = data_bin_size;
        self.meta_bin_size = meta_bin_size;
        self.src_bin_size = src_bin_size;
        self.dst_bin_size = dst_bin_size;
    }
}
