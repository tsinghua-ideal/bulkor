use crate::storage_ocalls::{get_sts_ptr, set_ptr, UntrustedAllocation};
use std::fs::{self, remove_file, rename, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
pub struct ShuffleManager {
    pub storage_id: u64,
    data_size: usize, //Bucket data
    meta_size: usize, //Bucket meta (not block meta) plus extra
    num_bins: usize,
    num_bins_on_disk: usize,
    //after separation or before combination
    //denoted as idle bins
    data_bin_buf: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    meta_bin_buf: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    data_bin_file_idx: Vec<u64>,
    meta_bin_file_idx: Vec<u64>,
    data_bin_file: File,
    meta_bin_file: File,
    //bins used during shuffle
    //denoted as work bins
    //bin format: (data, meta, random_keys, nonce, hash)
    src_bins: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)>,
    dst_bins: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>)>,
    src_bin_file_idx: Vec<u64>,
    dst_bin_file_idx: Vec<u64>,
    src_bin_file: File,
    dst_bin_file: File,
    //temp buf
    pub tmp_buf: (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>),
}

impl ShuffleManager {
    pub fn new(storage_id: u64, data_size: usize, meta_size: usize, num_bins: usize) -> Self {
        let num_bins_on_disk = num_bins / 2;

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

        ShuffleManager {
            storage_id,
            data_size,
            meta_size,
            num_bins,
            num_bins_on_disk,
            data_bin_buf: vec![Default::default(); num_bins - num_bins_on_disk],
            meta_bin_buf: vec![Default::default(); num_bins - num_bins_on_disk],
            data_bin_file_idx: vec![u64::MAX; num_bins_on_disk],
            meta_bin_file_idx: vec![u64::MAX; num_bins_on_disk],
            data_bin_file,
            meta_bin_file,
            src_bins: vec![Default::default(); num_bins - num_bins_on_disk],
            dst_bins: vec![Default::default(); num_bins - num_bins_on_disk],
            src_bin_file_idx: vec![u64::MAX; num_bins_on_disk],
            dst_bin_file_idx: vec![u64::MAX; num_bins_on_disk],
            src_bin_file,
            dst_bin_file,
            tmp_buf: Default::default(),
        }
    }

    pub fn clear_content(&mut self) {
        //sometimes the user-space allocator would not return the memory to OS, causing the memory overflow.
        let release_mem = unsafe { libc::malloc_trim(0) };
        println!("release the memory successfully? {:?}", release_mem == 1);

        self.data_bin_file_idx.clear();
        self.meta_bin_file_idx.clear();
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
    }

    pub fn pull_bin(
        &mut self,
        cur_bin_num: usize,
        bin_type: u8,
        bin_size: &mut usize,
        data_item_size: usize,
        meta_item_size: usize,
        has_data: bool,
        has_meta: bool,
        has_random_key: bool,
        nonce_size: usize,
        hash_size: usize,
    ) {
        assert!(cur_bin_num < self.num_bins);
        let mut nonce = vec![0; nonce_size];
        let mut hash = vec![0; hash_size];

        if bin_type == 0 {
            assert!(!has_random_key);
            assert!(!has_meta && has_data || has_meta && !has_data);
            if cur_bin_num < self.num_bins_on_disk {
                if has_data {
                    let mut offset = self.data_bin_file_idx[cur_bin_num];
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
                    self.tmp_buf = (data, Vec::new(), Vec::new(), nonce, hash);
                } else {
                    let mut offset = self.meta_bin_file_idx[cur_bin_num];
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
                    self.tmp_buf = (Vec::new(), meta, Vec::new(), nonce, hash);
                }
            } else {
                if has_data {
                    let (data, nonce, hash) =
                        std::mem::take(&mut self.data_bin_buf[cur_bin_num - self.num_bins_on_disk]);
                    let actual_data_size = data.len();
                    *bin_size = actual_data_size / data_item_size;
                    self.tmp_buf = (data, Vec::new(), Vec::new(), nonce, hash);
                } else {
                    let (meta, nonce, hash) =
                        std::mem::take(&mut self.meta_bin_buf[cur_bin_num - self.num_bins_on_disk]);
                    let actual_meta_size = meta.len();
                    *bin_size = actual_meta_size / meta_item_size;
                    self.tmp_buf = (Vec::new(), meta, Vec::new(), nonce, hash);
                }
            }
        } else if bin_type == 1 {
            assert!(has_meta);
            if cur_bin_num < self.num_bins_on_disk {
                let mut offset = self.src_bin_file_idx[cur_bin_num];
                let mut data_size_buf = [0u8; 8];
                let mut meta_size_buf = [0u8; 8];
                let mut random_key_size_buf = [0u8; 8];
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
                self.src_bin_file
                    .read_exact_at(&mut random_key_size_buf, offset)
                    .unwrap();
                offset += 8;
                let actual_random_key_size = usize::from_ne_bytes(random_key_size_buf);
                *bin_size = actual_meta_size / meta_item_size;
                assert!(has_data == (actual_data_size != 0));
                assert!(has_meta == (actual_meta_size != 0));
                assert!(has_random_key == (actual_random_key_size != 0));
                let mut data = vec![0; actual_data_size];
                self.src_bin_file.read_exact_at(&mut data, offset).unwrap();
                offset += actual_data_size as u64;
                let mut meta = vec![0; actual_meta_size];
                self.src_bin_file.read_exact_at(&mut meta, offset).unwrap();
                offset += actual_meta_size as u64;
                let mut random_keys = vec![0; actual_random_key_size];
                self.src_bin_file
                    .read_exact_at(&mut random_keys, offset)
                    .unwrap();
                offset += actual_random_key_size as u64;
                self.src_bin_file.read_exact_at(&mut nonce, offset).unwrap();
                offset += nonce_size as u64;
                self.src_bin_file.read_exact_at(&mut hash, offset).unwrap();
                self.tmp_buf = (data, meta, random_keys, nonce, hash);
            } else {
                let t = std::mem::take(&mut self.src_bins[cur_bin_num - self.num_bins_on_disk]);
                *bin_size = t.1.len() / meta_item_size;
                self.tmp_buf = t;
            }
        } else {
            unreachable!()
        }
    }

    pub fn push_buckets(&mut self, b_idx: usize, e_idx: usize) {
        let storage = unsafe {
            core::mem::transmute::<_, *mut UntrustedAllocation>(self.storage_id)
                .as_mut()
                .unwrap()
        };

        let (data, meta, _, _, _) = std::mem::take(&mut self.tmp_buf);
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
    }

    pub fn push_bin(&mut self, cur_bin_num: usize, bin_type: u8) {
        assert!(cur_bin_num < self.num_bins);
        let (data, meta, random_key, nonce, hash) = std::mem::take(&mut self.tmp_buf);
        let data_size = data.len();
        let meta_size = meta.len();
        let random_key_size = random_key.len();
        if bin_type == 0 {
            assert_eq!(random_key_size, 0);
            assert!(meta_size == 0 && data_size > 0 || meta_size > 0 && data_size == 0);
            if cur_bin_num < self.num_bins_on_disk {
                if data_size > 0 {
                    if self.data_bin_file_idx[cur_bin_num] == u64::MAX {
                        self.data_bin_file_idx[cur_bin_num] =
                            self.data_bin_file.seek(SeekFrom::End(0)).unwrap();
                    }
                    self.data_bin_file
                        .write_all(&(data_size).to_ne_bytes())
                        .unwrap();
                    self.data_bin_file.write_all(&data).unwrap();
                    self.data_bin_file.write_all(&nonce).unwrap();
                    self.data_bin_file.write_all(&hash).unwrap();
                } else {
                    if self.meta_bin_file_idx[cur_bin_num] == u64::MAX {
                        self.meta_bin_file_idx[cur_bin_num] =
                            self.meta_bin_file.seek(SeekFrom::End(0)).unwrap();
                    }
                    self.meta_bin_file
                        .write_all(&(meta_size).to_ne_bytes())
                        .unwrap();
                    self.meta_bin_file.write_all(&meta).unwrap();
                    self.meta_bin_file.write_all(&nonce).unwrap();
                    self.meta_bin_file.write_all(&hash).unwrap();
                }
            } else {
                if data_size > 0 {
                    self.data_bin_buf[cur_bin_num - self.num_bins_on_disk] = (data, nonce, hash);
                } else {
                    self.meta_bin_buf[cur_bin_num - self.num_bins_on_disk] = (meta, nonce, hash);
                }
            }
        } else if bin_type == 1 {
            if cur_bin_num < self.num_bins_on_disk {
                //if dst_bin_file_idx is reset in bin_switch, the assertion should be true
                //assert_eq!(self.dst_bin_file_idx[cur_bin_num], u64::MAX);
                self.dst_bin_file_idx[cur_bin_num] =
                    self.dst_bin_file.seek(SeekFrom::Current(0)).unwrap();
                self.dst_bin_file
                    .write_all(&(data_size).to_ne_bytes())
                    .unwrap();
                self.dst_bin_file
                    .write_all(&(meta_size).to_ne_bytes())
                    .unwrap();
                self.dst_bin_file
                    .write_all(&(random_key_size).to_ne_bytes())
                    .unwrap();
                self.dst_bin_file.write_all(&data).unwrap();
                self.dst_bin_file.write_all(&meta).unwrap();
                self.dst_bin_file.write_all(&random_key).unwrap();
                self.dst_bin_file.write_all(&nonce).unwrap();
                self.dst_bin_file.write_all(&hash).unwrap();
            } else {
                self.dst_bins[cur_bin_num - self.num_bins_on_disk] =
                    (data, meta, random_key, nonce, hash);
            }
        } else {
            unreachable!()
        }
    }

    pub fn bin_switch(&mut self, begin_bin_idx: usize, end_bin_idx: usize) {
        assert_eq!(end_bin_idx - begin_bin_idx, self.num_bins);
        //reset file pointer
        rename("shuffle_src_bin", "shuffle_src_bin_del").unwrap();
        rename("shuffle_dst_bin", "shuffle_src_bin").unwrap();
        std::mem::swap(&mut self.src_bin_file, &mut self.dst_bin_file);
        self.src_bin_file_idx = std::mem::take(&mut self.dst_bin_file_idx);
        self.dst_bin_file_idx = vec![u64::MAX; self.num_bins_on_disk];
        self.dst_bin_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&"shuffle_dst_bin")
            .unwrap();
        self.src_bin_file.seek(SeekFrom::Start(0)).unwrap();
        remove_file("shuffle_src_bin_del").unwrap();

        for idx in begin_bin_idx..end_bin_idx {
            if idx >= self.num_bins_on_disk {
                self.src_bins[idx - self.num_bins_on_disk] =
                    std::mem::take(&mut self.dst_bins[idx - self.num_bins_on_disk]);
            }
        }
    }
}
