use std::fs::{self, remove_file, File, OpenOptions};
use std::path::{Path, PathBuf};
pub struct ShuffleManager {
    z: u64,
    buf_count: usize,
    data_size: usize,
    meta_size: usize,
    num_bins: usize,
    num_bins_in_memory: usize,
    //before separation or after combination
    data_bucket_buf: Vec<u8>,
    meta_bucket_buf: Vec<u8>,
    data_bucket_file: File,
    meta_bucket_file: File,
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
        z: u64,
        buf_count: usize,
        data_size: usize,
        meta_size: usize,
        data_bucket_buf: Vec<u8>,
        meta_bucket_buf: Vec<u8>,
        data_bucket_file: File,
        meta_bucket_file: File,
        num_bins: usize,
    ) -> Self {
        //temporarily set
        let num_bins_in_memory = num_bins / 2;

        let data_bin_file_name = PathBuf::from("data_bin");
        let meta_bin_file_name = PathBuf::from("meta_bin");
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

        let src_bin_file_name = PathBuf::from("src_bin");
        let dst_bin_file_name = PathBuf::from("dst_bin");
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
            z,
            buf_count,
            data_size,
            meta_size,
            num_bins,
            num_bins_in_memory,
            data_bucket_buf,
            meta_bucket_buf,
            data_bucket_file,
            meta_bucket_file,
            data_bin_buf: vec![Default::default(); num_bins_in_memory],
            meta_bin_buf: vec![Default::default(); num_bins_in_memory],
            data_bin_file_idx: vec![Default::default(); num_bins - num_bins_in_memory],
            meta_bin_file_idx: vec![Default::default(); num_bins - num_bins_in_memory],
            data_bin_file,
            meta_bin_file,
            src_bins: vec![Default::default(); num_bins_in_memory],
            dst_bins: vec![Default::default(); num_bins_in_memory],
            src_bin_file_idx: vec![Default::default(); num_bins - num_bins_in_memory],
            dst_bin_file_idx: vec![Default::default(); num_bins - num_bins_in_memory],
            src_bin_file,
            dst_bin_file,
        }
    }
}

#[no_mangle]
pub extern "C" fn shuffle_pull_buckets(
    shuffle_id: u64,
    b_idx: usize,
    e_idx: usize,
    data: *mut u8,
    data_size: usize,
    meta: *mut u8,
    meta_size: usize,
) {
}

#[no_mangle]
pub extern "C" fn shuffle_pull_bin(
    shuffle_id: u64,
    cur_bin_num: usize,
    bin_type: u8,
    bin_size: *mut usize,
    data: *mut u8,
    data_size: usize,
    meta: *mut u8,
    meta_size: usize,
    random_key: *mut u8,
    random_key_size: usize,
    nonce: *mut u8,
    nonce_size: usize,
    hash: *mut u8,
    hash_size: usize,
) {
}

#[no_mangle]
pub extern "C" fn shuffle_push_buckets(
    shuffle_id: u64,
    b_idx: usize,
    e_idx: usize,
    databuf: *const u8,
    databuf_size: usize,
    metabuf: *const u8,
    metabuf_size: usize,
) {
}

#[no_mangle]
pub extern "C" fn shuffle_push_bin(
    shuffle_id: u64,
    cur_bin_num: usize,
    bin_type: u8,
    data: *const u8,
    data_size: usize,
    meta: *const u8,
    meta_size: usize,
    random_key: *const u8,
    random_key_size: usize,
    nonce: *const u8,
    nonce_size: usize,
    hash: *const u8,
    hash_size: usize,
) {
}

#[no_mangle]
pub extern "C" fn bin_switch(begin_bin_idx: usize, end_bin_idx: usize) {}

#[no_mangle]
pub extern "C" fn get_oram_tree(shuffle_id: u64, allocation_id: *mut u64) {}
