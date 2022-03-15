use crate::storage_ocalls::{get_sts_ptr, set_ptr};
use std::fs::{self, remove_file, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
pub struct ShuffleManager {
    data_size: usize, //Bucket data
    meta_size: usize, //Bucket meta (not block meta) plus extra
    num_bins: usize,
    num_bins_on_disk: usize,
    //before separation or after combination, info from oram tree
    count_in_mem: usize,
    pub ptrs: Vec<u8>,
    pub data_bucket_file: File,
    pub meta_bucket_file: File,
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
}

impl ShuffleManager {
    pub fn new(
        data_size: usize,
        meta_size: usize,
        count_in_mem: usize,
        ptrs: Vec<u8>,
        data_bucket_file: File,
        meta_bucket_file: File,
        num_bins: usize,
    ) -> Self {
        //temporarily set
        //in this setting, the second half are placed in memory, while the first half are placed on disk
        //it is easy to handle bin switch, because for cur_num_bins <= num_bins/2, the bins we are
        //processing is exactly in memory.
        //TODO: We need to deal with the case while num_bins_on_disk > num_bins/2
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
            data_size,
            meta_size,
            num_bins,
            num_bins_on_disk,
            count_in_mem,
            ptrs,
            data_bucket_file,
            meta_bucket_file,
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
        }
    }

    pub fn pull_buckets(&mut self, b_idx: usize, e_idx: usize, data: &mut [u8], meta: &mut [u8]) {
        let data_size = data.len();
        let meta_size = meta.len();
        assert_eq!(data_size / self.data_size, meta_size / self.meta_size);
        let data_item_size = self.data_size;
        let meta_item_size = self.meta_size;
        let count_in_mem = self.count_in_mem;

        //since data and meta is not loaded into memory, we read them directly from disk
        for (count, idx) in (b_idx..e_idx).into_iter().enumerate() {
            let p = get_sts_ptr(&self.ptrs, idx).2;
            if idx < self.count_in_mem {
                self.data_bucket_file
                    .read_exact_at(
                        &mut data[data_item_size * count..data_item_size * (count + 1)],
                        (data_item_size * (idx * 2 + p)) as u64,
                    )
                    .unwrap();
            } else {
                self.data_bucket_file
                    .read_exact_at(
                        &mut data[data_item_size * count..data_item_size * (count + 1)],
                        (data_item_size * (count_in_mem * 2 + (idx - count_in_mem) * 3 + p)) as u64,
                    )
                    .unwrap();
            }
        }

        for (count, idx) in (b_idx..e_idx).into_iter().enumerate() {
            let p = get_sts_ptr(&self.ptrs, idx).2;
            if idx < self.count_in_mem {
                self.meta_bucket_file
                    .read_exact_at(
                        &mut meta[meta_item_size * count..meta_item_size * (count + 1)],
                        (meta_item_size * (idx * 2 + p)) as u64,
                    )
                    .unwrap();
            } else {
                self.meta_bucket_file
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
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
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
                    let actual_data_size = usize::from_le_bytes(data_size_buf);
                    *bin_size = actual_data_size / data_item_size;
                    let mut data = vec![0; actual_data_size];
                    self.data_bin_file.read_exact_at(&mut data, offset).unwrap();
                    offset += actual_data_size as u64;
                    self.data_bin_file
                        .read_exact_at(&mut nonce, offset)
                        .unwrap();
                    offset += nonce_size as u64;
                    self.data_bin_file.read_exact_at(&mut hash, offset).unwrap();
                    return (data, Vec::new(), Vec::new(), nonce, hash);
                } else {
                    let mut offset = self.meta_bin_file_idx[cur_bin_num];
                    let mut meta_size_buf = [0u8; 8];
                    self.meta_bin_file
                        .read_exact_at(&mut meta_size_buf, offset)
                        .unwrap();
                    offset += 8;
                    let actual_meta_size = usize::from_le_bytes(meta_size_buf);
                    *bin_size = actual_meta_size / meta_item_size;
                    let mut meta = vec![0; actual_meta_size];
                    self.meta_bin_file.read_exact_at(&mut meta, offset).unwrap();
                    offset += actual_meta_size as u64;
                    self.meta_bin_file
                        .read_exact_at(&mut nonce, offset)
                        .unwrap();
                    offset += nonce_size as u64;
                    self.meta_bin_file.read_exact_at(&mut hash, offset).unwrap();
                    return (Vec::new(), meta, Vec::new(), nonce, hash);
                }
            } else {
                if has_data {
                    let (data, nonce, hash) =
                        std::mem::take(&mut self.data_bin_buf[cur_bin_num - self.num_bins_on_disk]);
                    let actual_data_size = data.len();
                    *bin_size = actual_data_size / data_item_size;
                    return (data, Vec::new(), Vec::new(), nonce, hash);
                } else {
                    let (meta, nonce, hash) =
                        std::mem::take(&mut self.meta_bin_buf[cur_bin_num - self.num_bins_on_disk]);
                    let actual_meta_size = meta.len();
                    *bin_size = actual_meta_size / meta_item_size;
                    return (Vec::new(), meta, Vec::new(), nonce, hash);
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
                let actual_data_size = usize::from_le_bytes(data_size_buf);
                self.src_bin_file
                    .read_exact_at(&mut meta_size_buf, offset)
                    .unwrap();
                offset += 8;
                let actual_meta_size = usize::from_le_bytes(meta_size_buf);
                self.src_bin_file
                    .read_exact_at(&mut random_key_size_buf, offset)
                    .unwrap();
                offset += 8;
                let actual_random_key_size = usize::from_le_bytes(random_key_size_buf);
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
                return (data, meta, random_keys, nonce, hash);
            } else {
                let t = std::mem::take(&mut self.src_bins[cur_bin_num - self.num_bins_on_disk]);
                *bin_size = t.1.len() / meta_item_size;
                return t;
            }
        } else {
            unreachable!()
        }
    }

    pub fn push_buckets(&mut self, b_idx: usize, e_idx: usize, data: &[u8], meta: &[u8]) {
        let data_size = data.len();
        let meta_size = meta.len();
        assert_eq!(data_size / self.data_size, meta_size / self.meta_size);
        let data_item_size = self.data_size;
        let meta_item_size = self.meta_size;
        let count_in_mem = self.count_in_mem;
        for (count, idx) in (b_idx..e_idx).into_iter().enumerate() {
            let (sts_on_disk, _, mut p) = get_sts_ptr(&self.ptrs, idx);
            //every bucket write once
            assert!(!sts_on_disk);
            if idx < self.count_in_mem {
                p = p ^ 1;
                self.data_bucket_file
                    .write_all_at(
                        &data[data_item_size * count..data_item_size * (count + 1)],
                        (data_item_size * (idx * 2 + p)) as u64,
                    )
                    .unwrap();
            } else {
                p = (p + 1) % 3;
                self.data_bucket_file
                    .write_all_at(
                        &data[data_item_size * count..data_item_size * (count + 1)],
                        (data_item_size * (count_in_mem * 2 + (idx - count_in_mem) * 3 + p)) as u64,
                    )
                    .unwrap();
            }
            set_ptr(&mut self.ptrs, idx, p);
        }

        for (count, idx) in (b_idx..e_idx).into_iter().enumerate() {
            let (sts_on_disk, _, p) = get_sts_ptr(&self.ptrs, idx);
            //already set above
            assert!(sts_on_disk);
            if idx < self.count_in_mem {
                self.meta_bucket_file
                    .write_all_at(
                        &meta[meta_item_size * count..meta_item_size * (count + 1)],
                        (meta_item_size * (idx * 2 + p)) as u64,
                    )
                    .unwrap();
            } else {
                self.meta_bucket_file
                    .write_all_at(
                        &meta[meta_item_size * count..meta_item_size * (count + 1)],
                        (meta_item_size * (count_in_mem * 2 + (idx - count_in_mem) * 3 + p)) as u64,
                    )
                    .unwrap();
            }
        }
    }

    pub fn push_bin(
        &mut self,
        cur_bin_num: usize,
        bin_type: u8,
        data: &[u8],
        meta: &[u8],
        random_key: &[u8],
        nonce: &[u8],
        hash: &[u8],
    ) {
        assert!(cur_bin_num < self.num_bins);
        let data_size = data.len();
        let meta_size = meta.len();
        let random_key_size = random_key.len();
        if bin_type == 0 {
            assert_eq!(random_key_size, 0);
            assert!(meta_size == 0 && data_size > 0 || meta_size > 0 && data_size == 0);
            if cur_bin_num < self.num_bins_on_disk {
                //println!("cur_num_bins must be equal to num_bins");
                if data_size > 0 {
                    if self.data_bin_file_idx[cur_bin_num] == u64::MAX {
                        self.data_bin_file_idx[cur_bin_num] =
                            self.data_bin_file.seek(SeekFrom::End(0)).unwrap();
                    }
                    self.data_bin_file
                        .write_all(&(data_size).to_le_bytes())
                        .unwrap();
                    self.data_bin_file.write_all(data).unwrap();
                    self.data_bin_file.write_all(nonce).unwrap();
                    self.data_bin_file.write_all(hash).unwrap();
                } else {
                    if self.meta_bin_file_idx[cur_bin_num] == u64::MAX {
                        self.meta_bin_file_idx[cur_bin_num] =
                            self.meta_bin_file.seek(SeekFrom::End(0)).unwrap();
                    }
                    self.meta_bin_file
                        .write_all(&(meta_size).to_le_bytes())
                        .unwrap();
                    self.meta_bin_file.write_all(meta).unwrap();
                    self.meta_bin_file.write_all(nonce).unwrap();
                    self.meta_bin_file.write_all(hash).unwrap();
                }
            } else {
                if data_size > 0 {
                    self.data_bin_buf[cur_bin_num - self.num_bins_on_disk] =
                        (data.to_vec(), nonce.to_vec(), hash.to_vec());
                } else {
                    self.meta_bin_buf[cur_bin_num - self.num_bins_on_disk] =
                        (meta.to_vec(), nonce.to_vec(), hash.to_vec());
                }
            }
        } else if bin_type == 1 {
            if cur_bin_num < self.num_bins_on_disk {
                //if dst_bin_file_idx is reset in bin_switch, the assertion should be true
                //assert_eq!(self.dst_bin_file_idx[cur_bin_num], u64::MAX);
                self.dst_bin_file_idx[cur_bin_num] =
                    self.dst_bin_file.seek(SeekFrom::Current(0)).unwrap();
                self.dst_bin_file
                    .write_all(&(data_size).to_le_bytes())
                    .unwrap();
                self.dst_bin_file
                    .write_all(&(meta_size).to_le_bytes())
                    .unwrap();
                self.dst_bin_file
                    .write_all(&(random_key_size).to_le_bytes())
                    .unwrap();
                self.dst_bin_file.write_all(data).unwrap();
                self.dst_bin_file.write_all(meta).unwrap();
                self.dst_bin_file.write_all(random_key).unwrap();
                self.dst_bin_file.write_all(nonce).unwrap();
                self.dst_bin_file.write_all(hash).unwrap();
            } else {
                self.dst_bins[cur_bin_num - self.num_bins_on_disk] = (
                    data.to_vec(),
                    meta.to_vec(),
                    random_key.to_vec(),
                    nonce.to_vec(),
                    hash.to_vec(),
                );
            }
        } else {
            unreachable!()
        }
    }

    pub fn bin_switch(&mut self, begin_bin_idx: usize, end_bin_idx: usize) {
        //reset file pointer
        self.src_bin_file.seek(SeekFrom::Start(0)).unwrap();
        self.dst_bin_file.seek(SeekFrom::Start(0)).unwrap();
        //TODO: the size of bins from src and dst may not be equal
        //but currently, this implementation works
        if end_bin_idx - begin_bin_idx == self.num_bins {
            std::mem::swap(&mut self.src_bin_file, &mut self.dst_bin_file);
            std::mem::swap(&mut self.src_bin_file_idx, &mut self.dst_bin_file_idx);
        }
        for idx in begin_bin_idx..end_bin_idx {
            if idx >= self.num_bins_on_disk {
                std::mem::swap(
                    &mut self.src_bins[idx - self.num_bins_on_disk],
                    &mut self.dst_bins[idx - self.num_bins_on_disk],
                );
            }
        }
    }
}
