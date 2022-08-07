// Copyright (c) 2018-2021 The MobileCoin Foundation

//! An implementation of the fog-ocall-oram-storage-edl interface
//!
//! This crate implements and exports the functions defined in the EDL file.
//! This is the only public API of this crate, everything else is an
//! implementation detail.
//!
//! Main ideas:
//! Instead of a global data structure protected by a mutex, this API does
//! the following:
//!
//! On enclave allocation request:
//! - Create an UntrustedAllocation on the heap (Box::new)
//! - This "control structure" contains the creation parameters of the
//!   allocation, and pointers to the block storage regions, created using
//!   ~malloc
//! - The allocation_id u64, is the value of this pointer The box is
//!   reconstituted whenever the enclave wants to access the allocation
//! - The box is freed when the enclave releases the allocation (This probably
//!   won't actually happen in production)
//!
//! When debug assertions are on, we keep track in a global variable which ids
//! are valid and which ones aren't so that we can give nice panic messages and
//! avoid memory corruption, if something really bad is happening in the enclave
//! and it is corrupting the id numbers.
//!
//! Note: There is some gnarly pointer-arithmetic stuff happening around the
//! copy_slice_nonoverlapping stuff. The reason this is happening is, on the
//! untrusted side, we do not know data_item_size and meta_item_size statically.
//! So while on the trusted side, it all works nicely in the type system, in
//! this side, we have to do a little arithmetic ourselves.
//! It is untenable for the untrusted side to also know these sizes statically,
//! it would create a strange coupling in the build process.

#![deny(missing_docs)]

use nix::fcntl::{posix_fadvise, PosixFadviseAdvice};
use regex::Regex;
use std::{
    alloc::{alloc, alloc_zeroed, dealloc, Layout},
    boxed::Box,
    cmp::max,
    collections::BTreeMap,
    convert::TryInto,
    fs::{self, remove_file, File, OpenOptions},
    io::{Read, Write},
    os::unix::{fs::FileExt, prelude::AsRawFd},
    path::{Path, PathBuf},
    slice,
    string::ToString,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        Mutex,
    },
};
mod shuffle_manager;
use shuffle_manager::ShuffleManager;

lazy_static! {
    /// The tree-top caching threshold, specified as log2 of a number of bytes.
    ///
    /// This is the approximate number of bytes that can be stored on the heap in the untrusted memory
    /// for a single ORAM storage object.
    ///
    /// This is expected to be tuned as a function of
    /// (1) number of (recursive) ORAM's needed
    /// (2) untrusted memory heap size, set at build time
    ///
    /// Changing this number influences any ORAM storage objects created after the change,
    /// but not before. So, it should normally be changed during enclave init, if at all.
    /// And it should be consistent with the TREEMID_CACHING_THRESHOLD_LOG2 in oram_storage/mod.rs
    pub static ref TREEMID_CACHING_THRESHOLD_LOG2: AtomicU32 = AtomicU32::new(33u32-1u32.log2()); // 8 GB

    /// It is used to recover the already allocated ORAM tree
    /// by mapping the level id to oram pointer
    pub static ref LEVEL_TO_ORAM_INST: Mutex<BTreeMap<u32, u64>> = Mutex::new(Default::default());

    /// It is used to save the pointers to stash
    /// by mapping snapshot_id to stashs <level, (stash_data, stash_meta)>
    pub static ref SNAPSHOT_TO_STASH: Mutex<BTreeMap<u64, BTreeMap<u32, (Vec<u8>, Vec<u8>)>>> = Mutex::new(Default::default());

    /// It is used to save the pointers to treetop
    /// by mapping snapshot_id to treetops <level, (treetop_data, treetop_meta)>
    pub static ref SNAPSHOT_TO_TREETOP: Mutex<BTreeMap<u64, BTreeMap<u32, (Vec<u8>, Vec<u8>)>>> = Mutex::new(Default::default());

    /// It is used to save the pointers to trusted merkle roots
    /// by mapping snapshot_id to treetops <level, merkle_roots>
    pub static ref SNAPSHOT_TO_MROOTS: Mutex<BTreeMap<u64, BTreeMap<u32, Vec<u8>>>> = Mutex::new(Default::default());

    /// It is used to save the pointers to trivial position map
    /// by mapping snapshot_id to trivial position map
    pub static ref SNAPSHOT_TO_TPOSMAP: Mutex<BTreeMap<u64, Vec<u8>>> = Mutex::new(Default::default());

    /// Last valid snapshot on disk
    pub static ref LAST_ON_DISK_SNAPSHOT_ID: AtomicU64 = AtomicU64::new(0u64);

    /// Shuffle manager handler
    pub static ref SHUFFLE_MANAGER_ID: AtomicU64 = AtomicU64::new(0u64);

    /// tmp position map (data, nonce, hash)
    pub static ref TMP_POS_MAP: Mutex<(Vec<u8>, Vec<u8>, Vec<u8>)> = Mutex::new((Vec::new(), Vec::new(), Vec::new()));
}

//return (does snapshot become old, the newest value is on the left/right)
fn get_sts_ptr(ptrs: &Vec<u8>, idx: usize) -> (bool, bool, usize) {
    let ptr_arr = ptrs[idx / 2];
    let r = ptr_arr >> ((1 - idx % 2) << 2);
    let on_disk_snapshot_become_old = (r >> 3) & 1 == 1;
    let in_mem_snapshot_become_old = (r >> 2) & 1 == 1;
    let pos_newest_value = (r & 0b11) as usize;
    (
        on_disk_snapshot_become_old,
        in_mem_snapshot_become_old,
        pos_newest_value,
    )
}

// v=0 for left, v=1 for right
// automatically set snapshot becomes old
fn set_ptr(ptrs: &mut Vec<u8>, idx: usize, v: usize) {
    let mut ptr_arr = ptrs[idx / 2];
    let v = (v as u8) << ((1 - idx % 2) << 2);
    let ptr_mask = 0b11 << ((1 - idx % 2) << 2);
    let sts_mask = 0b1100 << ((1 - idx % 2) << 2);
    //set ptr
    ptr_arr = ptr_arr & !ptr_mask;
    ptr_arr = ptr_arr | v;
    //set states
    ptr_arr = ptr_arr | sts_mask;
    ptrs[idx / 2] = ptr_arr;
}

fn clear_all_in_mem_sts(ptrs: &mut Vec<u8>, end_idx: usize) {
    assert!(end_idx != usize::MAX);
    let end = end_idx / 2;
    let mut ptr = ptrs[end];
    match end_idx % 2 {
        0 => ptr = ptr & 0b10111111,
        1 => ptr = ptr & 0b10111011,
        _ => unreachable!(),
    }
    ptrs[end] = ptr;
    for ptr in &mut ptrs[..end] {
        *ptr = *ptr & 0b10111011;
    }
}

fn clear_all_sts(ptrs: &mut Vec<u8>, end_idx: usize) {
    if end_idx == usize::MAX {
        for ptr in ptrs {
            *ptr = *ptr & 0b00110011;
        }
    } else {
        let end = end_idx / 2;
        let mut ptr = ptrs[end];
        match end_idx % 2 {
            0 => ptr = ptr & 0b00111111,
            1 => ptr = ptr & 0b00110011,
            _ => unreachable!(),
        }
        ptrs[end] = ptr;
        for ptr in &mut ptrs[..end] {
            *ptr = *ptr & 0b00110011;
        }
    }
}

/// Resources held on untrusted side in connection to an allocation request by
/// enclave
///
/// This is not actually part of the public interface of the crate, the only
/// thing exported by the crate is the enclave EDL functions
struct UntrustedAllocation {
    /// The level of this ORAM
    level: u32,
    /// The number of data and meta items in files
    count: usize,
    /// The number of data and meta items stored in this allocation
    count_in_mem: usize,
    // The maximum count for the treetop storage in untrusted memory,
    // based on what we loaded from TREEMID_CACHING_THRESHOLD_LOG2 at construction time
    // This must never change after construction.
    treetop_max_count: u64,
    /// The size of a data item in bytes
    data_item_size: usize,
    /// The size of a meta item in bytes
    meta_item_size: usize,
    /// indicate whether the left or the right is the newest
    /// each bucket pair only need one bit to indicate
    ptrs: Vec<u8>,
    /// The pointer to the data items, using pointer may be for alignment issue
    data_pointer: *mut u8,
    /// The pointer to the meta items, using pointer may be for alignment issue
    meta_pointer: *mut u8,
    /// The file descriptor for data file
    data_file: File,
    /// The file descriptor for meta file
    meta_file: File,
    /// A flag set to true when a thread is in the critical section and released
    /// when it leaves. This is used to trigger assertions if there is a
    /// race happening on this API This is simpler and less expensive than
    /// an actual mutex to protect critical sections
    critical_section_flag: AtomicBool,
    /// A flag set to true when there is an active checkout. This is used to
    /// trigger assertions if checkout are not followed by checkin
    /// operation.
    checkout_flag: AtomicBool,
}

/// Tracks total memory allocated via this mechanism for logging purposes
static TOTAL_MEM_FOOTPRINT_KB: AtomicU64 = AtomicU64::new(0);

/// Helper which computes the total memory in kb allocated for count,
/// data_item_size, meta_item_size
fn compute_mem_kb(count: usize, data_item_size: usize, meta_item_size: usize) -> u64 {
    // Doubling
    let num_bytes = 2 * (count * (data_item_size + meta_item_size)) as u64;
    // Divide by 1024 and round up, to compute num_bytes in kb
    (num_bytes + 1023) / 1024
}

impl UntrustedAllocation {
    /// Create a new untrusted allocation for given count and item sizes, on the
    /// heap
    ///
    /// Data and meta item sizes must be divisible by 8, consistent with the
    /// contract described in the edl file
    pub fn new(
        level: u32,
        snapshot_id: u64,
        is_latest: bool,
        count: usize,
        data_item_size: usize,
        meta_item_size: usize,
    ) -> Self {
        let treetop_max_count: u64 = max(
            2u64,
            (1u64 << TREEMID_CACHING_THRESHOLD_LOG2.load(Ordering::SeqCst)) / data_item_size as u64,
        );
        let count_in_mem = if count <= treetop_max_count as usize {
            count
        } else {
            treetop_max_count as usize
        };
        let mem_kb = compute_mem_kb(count_in_mem, data_item_size, meta_item_size);
        let total_mem_kb = mem_kb + TOTAL_MEM_FOOTPRINT_KB.fetch_add(mem_kb, Ordering::SeqCst);
        log::info!("Untrusted is allocating oram storage: count_in_mem = {}, count_on_disk = {}, data_size = {}, meta_size = {}, mem = {} KB. Total mem allocated this way = {} KB", count_in_mem, count, data_item_size, meta_item_size, mem_kb, total_mem_kb);
        assert!(
            data_item_size % 8 == 0,
            "data item size is not good: {}",
            data_item_size
        );
        assert!(
            meta_item_size % 8 == 0,
            "meta item size is not good: {}",
            meta_item_size
        );

        let ptr_file_name =
            PathBuf::from("ptr_".to_string() + &snapshot_id.to_string() + "-" + &level.to_string());
        let data_file_name = PathBuf::from("data_".to_string() + &level.to_string());
        let meta_file_name = PathBuf::from("meta_".to_string() + &level.to_string());
        let mut ptr_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&ptr_file_name)
            .unwrap();
        let mut data_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&data_file_name)
            .unwrap();
        let mut meta_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&meta_file_name)
            .unwrap();

        let data_pointer = unsafe {
            alloc(Layout::from_size_align(2 * count_in_mem * data_item_size, 8).unwrap())
        };
        if data_pointer.is_null() {
            panic!(
                "Could not allocate memory for data segment: {}",
                2 * count_in_mem * data_item_size
            )
        }
        let meta_pointer = if snapshot_id > 0 && !is_latest {
            //zero is filled during build oram from shuffle manager
            unsafe { alloc(Layout::from_size_align(2 * count_in_mem * meta_item_size, 8).unwrap()) }
        } else {
            unsafe {
                alloc_zeroed(Layout::from_size_align(2 * count_in_mem * meta_item_size, 8).unwrap())
            }
        };

        if meta_pointer.is_null() {
            panic!(
                "Could not allocate memory for meta segment: {}",
                2 * count_in_mem * meta_item_size
            )
        }

        let ptrs_len = (count - 1) / 2 + 1;
        let mut ptrs = vec![0 as u8; ptrs_len];

        //for stale position ORAMs, just initialize them although the snapshot id is not 0
        if snapshot_id == 0 || (!is_latest && level > 0) {
            //the following step mainly aims to continue the initalization when the system crash
            //during the initialization. But it is not necessary and will hinder the clear if
            //is_latest && level > 0 is true.
            // let mut data_file_len = data_file.metadata().unwrap().len() as usize;
            // let mut meta_file_len = meta_file.metadata().unwrap().len() as usize;
            // let ptr_file_len = ptr_file.metadata().unwrap().len() as usize;
            let mut data_file_len = 0;
            let mut meta_file_len = 0;
            let ptr_file_len = 0;

            // Initialization is not finished, so do the initialization
            let zeros = vec![0u8; data_item_size];
            while data_file_len < (2 * count_in_mem + 3 * (count - count_in_mem)) * data_item_size {
                data_file.write_all(&zeros).unwrap();
                data_file_len += data_item_size;
            }
            while meta_file_len < (2 * count_in_mem + 3 * (count - count_in_mem)) * meta_item_size {
                meta_file.write_all(&zeros[..meta_item_size]).unwrap();
                meta_file_len += meta_item_size;
            }
            if ptr_file_len < ptrs_len {
                ptr_file.write_all(&ptrs[ptr_file_len..ptrs_len]).unwrap();
            }
            data_file.sync_all().unwrap();
            meta_file.sync_all().unwrap();
            ptr_file.sync_all().unwrap();
            posix_fadvise(
                data_file.as_raw_fd(),
                0,
                0,
                PosixFadviseAdvice::POSIX_FADV_DONTNEED,
            )
            .unwrap();
            posix_fadvise(
                meta_file.as_raw_fd(),
                0,
                0,
                PosixFadviseAdvice::POSIX_FADV_DONTNEED,
            )
            .unwrap();
            posix_fadvise(
                ptr_file.as_raw_fd(),
                0,
                0,
                PosixFadviseAdvice::POSIX_FADV_DONTNEED,
            )
            .unwrap();
        } else {
            //recovery
            //For level==0 && !is_latest case, no data and meta is loaded
            if is_latest {
                let data = unsafe {
                    core::slice::from_raw_parts_mut(data_pointer, 2 * count_in_mem * data_item_size)
                };
                let meta = unsafe {
                    core::slice::from_raw_parts_mut(meta_pointer, 2 * count_in_mem * meta_item_size)
                };
                data_file.read_exact_at(data, 0).unwrap();
                meta_file.read_exact_at(meta, 0).unwrap();
            }
            ptr_file.read_exact_at(&mut ptrs, 0).unwrap();
        }

        let critical_section_flag = AtomicBool::new(false);
        let checkout_flag = AtomicBool::new(false);

        Self {
            level,
            count,
            count_in_mem,
            treetop_max_count,
            data_item_size,
            meta_item_size,
            ptrs,
            data_pointer,
            meta_pointer,
            data_file,
            meta_file,
            critical_section_flag,
            checkout_flag,
        }
    }
}

impl Drop for UntrustedAllocation {
    fn drop(&mut self) {
        unsafe {
            dealloc(
                self.data_pointer as *mut u8,
                Layout::from_size_align_unchecked(2 * self.count_in_mem * self.data_item_size, 8),
            );
            dealloc(
                self.meta_pointer as *mut u8,
                Layout::from_size_align_unchecked(2 * self.count_in_mem * self.meta_item_size, 8),
            );
        }
        let mem_kb = compute_mem_kb(self.count_in_mem, self.data_item_size, self.meta_item_size);
        TOTAL_MEM_FOOTPRINT_KB.fetch_sub(mem_kb, Ordering::SeqCst);
    }
}

#[no_mangle]
pub unsafe extern "C" fn pull_all_elements(
    id: u64,
    databuf: *mut u8,
    databuf_len: usize,
    metabuf: *mut u8,
    metabuf_len: usize,
) {
    let ptr: *const UntrustedAllocation = core::mem::transmute(id);

    let count_in_mem = (*ptr).count_in_mem;
    let data_item_size = (*ptr).data_item_size;
    let meta_item_size = (*ptr).meta_item_size;

    let ptrs = &(*ptr).ptrs;
    let data_buf = core::slice::from_raw_parts_mut(databuf, databuf_len);
    let meta_buf = core::slice::from_raw_parts_mut(metabuf, metabuf_len);

    for idx in 0..(*ptr).count {
        let p = get_sts_ptr(ptrs, idx).2;
        if idx < count_in_mem {
            (*ptr)
                .data_file
                .read_exact_at(
                    &mut data_buf[data_item_size * idx..data_item_size * (idx + 1)],
                    (data_item_size * (2 * idx + p)) as u64,
                )
                .unwrap();
            (*ptr)
                .meta_file
                .read_exact_at(
                    &mut meta_buf[meta_item_size * idx..meta_item_size * (idx + 1)],
                    (meta_item_size * (2 * idx + p)) as u64,
                )
                .unwrap();
        } else {
            (*ptr)
                .data_file
                .read_exact_at(
                    &mut data_buf[data_item_size * idx..data_item_size * (idx + 1)],
                    (data_item_size * (count_in_mem * 2 + (idx - count_in_mem) * 3 + p)) as u64,
                )
                .unwrap();
            (*ptr)
                .meta_file
                .read_exact_at(
                    &mut meta_buf[meta_item_size * idx..meta_item_size * (idx + 1)],
                    (meta_item_size * (count_in_mem * 2 + (idx - count_in_mem) * 3 + p)) as u64,
                )
                .unwrap();
        }
    }

    //check the buf
    for idx in 0..(*ptr).count as usize {
        let p = get_sts_ptr(ptrs, idx).2;
        let default = vec![0u8; meta_item_size];
        if &meta_buf[meta_item_size * idx..meta_item_size * (idx + 1)] == &default {
            if idx < count_in_mem {
                let mut buf = vec![0; meta_item_size * 2];
                (*ptr)
                    .meta_file
                    .read_exact_at(&mut buf, (meta_item_size * (2 * idx)) as u64)
                    .unwrap();
                println!(
                    "untrusted domain, dummy meta idx = {:?}, buf = {:?}, p = {:?}",
                    idx, buf, p
                );
            } else {
                let mut buf = vec![0; meta_item_size * 3];
                (*ptr)
                    .meta_file
                    .read_exact_at(
                        &mut buf,
                        (meta_item_size * (count_in_mem * 2 + (idx - count_in_mem) * 3)) as u64,
                    )
                    .unwrap();
                println!(
                    "untrusted domain, dummy meta idx = {:?}, buf = {:?}, p = {:?}",
                    idx, buf, p
                );
            }
            continue;
        }
    }
}

// These extern "C" functions must match edl file

/// # Safety
///
/// meta_size and data_size must be divisible by 8
/// id_out must be a valid pointer to a u64
#[no_mangle]
pub extern "C" fn allocate_oram_storage(
    level: u32,
    snapshot_id: u64,
    is_latest: u8,
    count: u64,
    data_size: u64,
    meta_size: u64,
    id_out: *mut u64,
) {
    let mut m = LEVEL_TO_ORAM_INST.lock().unwrap();
    let id = m.entry(level).or_insert(0);
    //currentl we only handle the shuffle in the machine crash case
    if *id > 0 {
        assert!(is_latest != 0);
    }
    println!("id = {:?}", id);
    if *id == 0 {
        println!(
            "before allocing, level = {:?}, snapshot_id = {:?}",
            level, snapshot_id
        );
        let result = Box::new(UntrustedAllocation::new(
            level,
            snapshot_id,
            is_latest != 0,
            count as usize,
            data_size as usize,
            meta_size as usize,
        ));
        *id = Box::into_raw(result) as u64;
        #[cfg(debug_assertions)]
        debug_checks::add_id(*id);
    }
    unsafe {
        *id_out = *id;
    }
}

#[no_mangle]
pub extern "C" fn allocate_shuffle_manager(
    allocation_id: u64,
    z: u64,
    bin_size_in_block: usize,
    shuffle_id: *mut u64,
) {
    let storage = unsafe {
        core::mem::transmute::<_, *mut UntrustedAllocation>(allocation_id)
            .as_mut()
            .unwrap()
    };
    assert!(
        !storage.critical_section_flag.swap(true, Ordering::SeqCst),
        "Could not enter critical section when releasing storage"
    );

    let data_item_size = storage.data_item_size;
    let meta_item_size = storage.meta_item_size;

    let mut num_bins = 2 * z as usize * storage.count / bin_size_in_block;
    if num_bins == 0 {
        num_bins = 1;
    }
    let ratio = storage.count / storage.count_in_mem;
    assert!(ratio >= 1);

    LEVEL_TO_ORAM_INST
        .lock()
        .unwrap()
        .remove(&storage.level)
        .unwrap();
    assert!(
        storage.critical_section_flag.swap(false, Ordering::SeqCst),
        "Could not leave critical section when persisting storage"
    );
    #[cfg(debug_assertions)]
    debug_checks::remove_id(allocation_id);

    let shuffle_manager = Box::new(ShuffleManager::new(
        allocation_id,
        data_item_size,
        meta_item_size,
        num_bins,
        ratio,
    ));
    let id = Box::into_raw(shuffle_manager) as u64;
    SHUFFLE_MANAGER_ID.store(id, Ordering::SeqCst);
    unsafe {
        *shuffle_id = id;
    }
}

#[no_mangle]
pub extern "C" fn build_oram_from_shuffle_manager(shuffle_id: u64, allocation_id: u64) {
    assert_eq!(shuffle_id, SHUFFLE_MANAGER_ID.load(Ordering::SeqCst));
    SHUFFLE_MANAGER_ID.store(0, Ordering::SeqCst);
    let manager =
        unsafe { Box::from_raw(core::mem::transmute::<_, *mut ShuffleManager>(shuffle_id)) };
    assert_eq!(manager.storage_id, allocation_id);
    let storage = unsafe {
        core::mem::transmute::<_, *mut UntrustedAllocation>(allocation_id)
            .as_mut()
            .unwrap()
    };
    //release the space,
    drop(manager);
    //delete shuffle files
    {
        //read from disk
        let path = Path::new("./");
        for entry in fs::read_dir(path).expect("reading directory fails") {
            if let Ok(entry) = entry {
                let file = entry.path();
                let filename = file.to_str().unwrap();

                if filename.contains("shuffle_") && !filename.contains("nonce") {
                    remove_file(file).unwrap();
                }
            }
        }
    }
    let _release_mem = unsafe { libc::malloc_trim(0) };

    //no need to deal with ptr_file
    assert!(LEVEL_TO_ORAM_INST
        .lock()
        .unwrap()
        .insert(storage.level, allocation_id)
        .is_none());
    #[cfg(debug_assertions)]
    debug_checks::add_id(allocation_id);
}

/// # Safety
///
/// id must be a valid id previously returned by allocate_oram_storage
pub unsafe fn release_oram_storage(id: u64) {
    let ptr: *mut UntrustedAllocation = core::mem::transmute(id);
    assert!(
        !(*ptr).critical_section_flag.swap(true, Ordering::SeqCst),
        "Could not enter critical section when releasing storage"
    );
    LEVEL_TO_ORAM_INST.lock().unwrap().remove(&(*ptr).level);
    let _get_dropped = Box::<UntrustedAllocation>::from_raw(ptr);
    #[cfg(debug_assertions)]
    debug_checks::remove_id(id);
}

pub fn release_all_oram_storage() {
    let ids = LEVEL_TO_ORAM_INST
        .lock()
        .unwrap()
        .values()
        .cloned()
        .collect::<Vec<_>>();
    for id in ids {
        unsafe {
            release_oram_storage(id);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn persist_oram_storage(
    level: u32,
    snapshot_id: u64,
    is_volatile: u8,
    id: u64,
) {
    println!("begin persist_oram_storage");
    #[cfg(debug_assertions)]
    debug_checks::check_id(id);
    let ptr: *mut UntrustedAllocation = core::mem::transmute(id);
    assert!(
        !(*ptr).critical_section_flag.swap(true, Ordering::SeqCst),
        "Could not enter critical section when checking out storage"
    );
    assert_eq!(level, (*ptr).level);

    //because we do not persist various versions
    let data_item_size = (*ptr).data_item_size;
    let meta_item_size = (*ptr).meta_item_size;
    let count_in_mem = (*ptr).count_in_mem;
    let data = core::slice::from_raw_parts((*ptr).data_pointer, 2 * count_in_mem * data_item_size);
    let meta = core::slice::from_raw_parts((*ptr).meta_pointer, 2 * count_in_mem * meta_item_size);
    let ptrs_m_mut = &mut (*ptr).ptrs;

    //on disk persistence
    if is_volatile == 0 {
        let ptrs_m = ptrs_m_mut.clone(); //avoid blocking access
        clear_all_sts(ptrs_m_mut, usize::MAX);
        let old_snapshot_id = LAST_ON_DISK_SNAPSHOT_ID.load(Ordering::SeqCst);
        let old_ptr_file_name = PathBuf::from(
            "ptr_".to_string() + &old_snapshot_id.to_string() + "-" + &level.to_string(),
        );
        let mut old_ptr_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&old_ptr_file_name)
            .unwrap();
        let mut ptrs_d = Vec::new();
        old_ptr_file.read_to_end(&mut ptrs_d).unwrap(); //TODO: optimization

        for idx in 1..count_in_mem {
            let (sts_on_disk, _, p_src) = get_sts_ptr(&ptrs_m, idx);
            let (old_sts_on_disk, old_sts_in_mem, mut p_dst) = get_sts_ptr(&ptrs_d, idx);
            assert!(p_src < 2 && p_dst < 2); //for in memory part, p cannot be larger than 1
            assert_eq!(old_sts_on_disk || old_sts_in_mem, false);
            if sts_on_disk {
                // update exists
                p_dst = p_dst ^ 1;
                set_ptr(&mut ptrs_d, idx, p_dst);

                let b_idx_m = 2 * idx + p_src;
                let e_idx_m = 2 * idx + p_src + 1;
                let b_idx_d = 2 * idx + p_dst;
                (*ptr)
                    .data_file
                    .write_all_at(
                        &data[b_idx_m * data_item_size..e_idx_m * data_item_size],
                        (b_idx_d * data_item_size) as u64,
                    )
                    .unwrap();
                (*ptr)
                    .meta_file
                    .write_all_at(
                        &meta[b_idx_m * meta_item_size..e_idx_m * meta_item_size],
                        (b_idx_d * meta_item_size) as u64,
                    )
                    .unwrap();
            }
        }

        //on disk snapshot switch
        (&mut ptrs_d[count_in_mem / 2..]).copy_from_slice(&ptrs_m[count_in_mem / 2..]);
        clear_all_sts(&mut ptrs_d, usize::MAX);

        let new_ptr_file_name =
            PathBuf::from("ptr_".to_string() + &snapshot_id.to_string() + "-" + &level.to_string());
        let mut new_ptr_file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .open(&new_ptr_file_name)
            .unwrap();

        new_ptr_file.write_all(&ptrs_d).unwrap();

        (*ptr).data_file.sync_all().unwrap();
        (*ptr).meta_file.sync_all().unwrap();
        new_ptr_file.sync_all().unwrap();
        posix_fadvise(
            (*ptr).data_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
        posix_fadvise(
            (*ptr).meta_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
        posix_fadvise(
            new_ptr_file.as_raw_fd(),
            0,
            0,
            PosixFadviseAdvice::POSIX_FADV_DONTNEED,
        )
        .unwrap();
    } else {
        //in mem snapshot switch
        clear_all_in_mem_sts(ptrs_m_mut, count_in_mem - 1);
    }

    assert!(
        (*ptr).critical_section_flag.swap(false, Ordering::SeqCst),
        "Could not leave critical section when persisting storage"
    );
    println!("end persist_oram_storage");
}

/// # Safety
///
/// idx must point to a buffer of length idx_len
/// databuf must point to a buffer of length databuf_len
/// metabuf must point to a buffer of length metabuf_len
///
/// id must be a valid id previously returned by allocate_oram_storage
///
/// databuf_len must be equal to idx_len * data_item_size,
/// where data_item_size was passed when allocating storage.
///
/// metabuf_len must be equal to idx_len * meta_item_size,
/// where meta_item_size was passed when allocating storage.
///
/// All indices must be in bounds, less than count that was passed when
/// allocaitng.
#[no_mangle]
pub unsafe extern "C" fn checkout_oram_storage(
    id: u64,
    idx: *const u64,
    idx_len: usize,
    databuf: *mut u8,
    databuf_len: usize,
    metabuf: *mut u8,
    metabuf_len: usize,
) {
    #[cfg(debug_assertions)]
    debug_checks::check_id(id);
    let ptr: *const UntrustedAllocation = core::mem::transmute(id);
    assert!(
        !(*ptr).critical_section_flag.swap(true, Ordering::SeqCst),
        "Could not enter critical section when checking out storage"
    );
    assert!(
        !(*ptr).checkout_flag.swap(true, Ordering::SeqCst),
        "illegal checkout"
    );

    let count_in_mem = (*ptr).count_in_mem;
    let data_item_size = (*ptr).data_item_size;
    let meta_item_size = (*ptr).meta_item_size;
    assert!(idx_len * data_item_size == databuf_len);
    assert!(idx_len * meta_item_size == metabuf_len);

    let ptrs = &(*ptr).ptrs;
    let indices = core::slice::from_raw_parts(idx, idx_len);
    let databuf_s = core::slice::from_raw_parts_mut(databuf, databuf_len);
    let metabuf_s = core::slice::from_raw_parts_mut(metabuf, metabuf_len);

    let first_treetop_index = indices
        .iter()
        .position(|idx| idx < &(*ptr).treetop_max_count)
        .expect("should be unreachable, at least one thing should be in the treetop");

    for (count, index) in indices.iter().enumerate() {
        let index = *index as usize;
        let p = get_sts_ptr(ptrs, index).2;
        if count < first_treetop_index {
            (*ptr)
                .data_file
                .read_exact_at(
                    &mut databuf_s[data_item_size * count..data_item_size * (count + 1)],
                    (data_item_size * (count_in_mem * 2 + (index - count_in_mem) * 3 + p)) as u64,
                )
                .unwrap();
        } else {
            assert!(p < 2);
            core::ptr::copy_nonoverlapping(
                (*ptr).data_pointer.add(data_item_size * (index * 2 + p)),
                databuf.add(data_item_size * count),
                data_item_size,
            );
        }
    }

    for (count, index) in indices.iter().enumerate() {
        let index = *index as usize;
        let p = get_sts_ptr(ptrs, index).2;
        if count < first_treetop_index {
            (*ptr)
                .meta_file
                .read_exact_at(
                    &mut metabuf_s[meta_item_size * count..meta_item_size * (count + 1)],
                    (meta_item_size * (count_in_mem * 2 + (index - count_in_mem) * 3 + p)) as u64,
                )
                .unwrap();
        } else {
            assert!(p < 2);
            core::ptr::copy_nonoverlapping(
                (*ptr).meta_pointer.add(meta_item_size * (index * 2 + p)),
                metabuf.add(meta_item_size * count),
                meta_item_size,
            );
        }
    }

    assert!(
        (*ptr).critical_section_flag.swap(false, Ordering::SeqCst),
        "Could not leave critical section when checking out storage"
    );

    posix_fadvise(
        (*ptr).data_file.as_raw_fd(),
        0,
        0,
        PosixFadviseAdvice::POSIX_FADV_DONTNEED,
    )
    .unwrap();
    posix_fadvise(
        (*ptr).meta_file.as_raw_fd(),
        0,
        0,
        PosixFadviseAdvice::POSIX_FADV_DONTNEED,
    )
    .unwrap();
}

/// # Safety
///
/// idx must point to a buffer of length idx_len
/// databuf must point to a buffer of length databuf_len
/// metabuf must point to a buffer of length metabuf_len
///
/// id must be a valid id previously returned by allocate_oram_storage
///
/// databuf_len must be equal to idx_len * data_item_size,
/// where data_item_size was passed when allocating storage.
///
/// metabuf_len must be equal to idx_len * meta_item_size,
/// where meta_item_size was passed when allocating storage.
///
/// All indices must be in bounds, less than count that was passed when
/// allocaitng.
#[no_mangle]
pub unsafe extern "C" fn checkin_oram_storage(
    id: u64,
    idx: *const u64,
    idx_len: usize,
    databuf: *const u8,
    databuf_len: usize,
    metabuf: *const u8,
    metabuf_len: usize,
) {
    #[cfg(debug_assertions)]
    debug_checks::check_id(id);
    let ptr: *mut UntrustedAllocation = core::mem::transmute(id);
    assert!(
        !(*ptr).critical_section_flag.swap(true, Ordering::SeqCst),
        "Could not enter critical section when checking in storage"
    );
    assert!(
        (*ptr).checkout_flag.swap(false, Ordering::SeqCst),
        "illegal checkin"
    );

    let count_in_mem = (*ptr).count_in_mem;
    let data_item_size = (*ptr).data_item_size;
    let meta_item_size = (*ptr).meta_item_size;
    assert!(idx_len * data_item_size == databuf_len);
    assert!(idx_len * meta_item_size == metabuf_len);

    let ptrs = &mut (*ptr).ptrs;
    let indices = core::slice::from_raw_parts(idx, idx_len);
    let databuf_s = core::slice::from_raw_parts(databuf, databuf_len);
    let metabuf_s = core::slice::from_raw_parts(metabuf, metabuf_len);

    let first_treetop_index = indices
        .iter()
        .position(|idx| idx < &(*ptr).treetop_max_count)
        .expect("should be unreachable, at least one thing should be in the treetop");

    // First step: Do the part that's on the disk
    // Second step: Do the part that's in the treetop
    for (count, index) in indices.iter().enumerate() {
        let index = *index as usize;
        if count < first_treetop_index {
            let (sts_on_disk, _, mut p) = get_sts_ptr(ptrs, index);
            if !sts_on_disk {
                p = (p + 1) % 3;
            }
            set_ptr(ptrs, index, p);
            (*ptr)
                .data_file
                .write_all_at(
                    &databuf_s[data_item_size * count..data_item_size * (count + 1)],
                    (data_item_size * (count_in_mem * 2 + (index - count_in_mem) * 3 + p)) as u64,
                )
                .unwrap();
        } else {
            let (_, sts_in_mem, mut p) = get_sts_ptr(ptrs, index);
            assert!(p < 2);
            if !sts_in_mem {
                p = p ^ 1;
            }
            set_ptr(ptrs, index, p);
            core::ptr::copy_nonoverlapping(
                databuf.add(data_item_size * count),
                (*ptr).data_pointer.add(data_item_size * (index * 2 + p)),
                data_item_size,
            );
        }
    }

    for (count, index) in indices.iter().enumerate() {
        let index = *index as usize;
        let (sts_on_disk, sts_in_mem, p) = get_sts_ptr(ptrs, index);
        assert_eq!(sts_in_mem && sts_on_disk, true);
        if count < first_treetop_index {
            (*ptr)
                .meta_file
                .write_all_at(
                    &metabuf_s[meta_item_size * count..meta_item_size * (count + 1)],
                    (meta_item_size * (count_in_mem * 2 + (index - count_in_mem) * 3 + p)) as u64,
                )
                .unwrap();
        } else {
            core::ptr::copy_nonoverlapping(
                metabuf.add(meta_item_size * count),
                (*ptr).meta_pointer.add(meta_item_size * (index * 2 + p)),
                meta_item_size,
            );
        }
    }

    assert!(
        (*ptr).critical_section_flag.swap(false, Ordering::SeqCst),
        "Could not leave critical section when checking in storage"
    );
    (*ptr).data_file.sync_all().unwrap();
    (*ptr).meta_file.sync_all().unwrap();
    posix_fadvise(
        (*ptr).data_file.as_raw_fd(),
        0,
        0,
        PosixFadviseAdvice::POSIX_FADV_DONTNEED,
    )
    .unwrap();
    posix_fadvise(
        (*ptr).meta_file.as_raw_fd(),
        0,
        0,
        PosixFadviseAdvice::POSIX_FADV_DONTNEED,
    )
    .unwrap();
}

#[no_mangle]
pub extern "C" fn get_valid_snapshot_id(
    size_triv_pos_map: u64,
    snapshot_id: *mut u64,
    lifetime_id_from_meta: *mut u64,
) {
    let mut snapshot_to_tposmap = SNAPSHOT_TO_TPOSMAP.lock().unwrap();

    let mut id_to_len = BTreeMap::new();
    //read from disk
    let path = Path::new("./");
    let mut id_to_files = BTreeMap::new();
    for entry in fs::read_dir(path).expect("reading directory fails") {
        if let Ok(entry) = entry {
            let file = entry.path();
            let filename = file.to_str().unwrap();

            if filename.contains("trivial_posmap_") {
                lazy_static! {
                    static ref RE: Regex = Regex::new(r"trivial_posmap_(\d+)").unwrap();
                }
                for cap in RE.captures_iter(filename) {
                    let id = (&cap[1]).parse::<u64>().unwrap();
                    id_to_len.insert(id, entry.metadata().unwrap().len());
                    let files = id_to_files.entry(id).or_insert(vec![]);
                    files.push(file.clone());
                }
            } else {
                lazy_static! {
                    static ref RE: Regex = Regex::new(r"(\d+)-(\d+)").unwrap();
                }
                for cap in RE.captures_iter(filename) {
                    let id = (&cap[1]).parse::<u64>().unwrap();
                    let files = id_to_files.entry(id).or_insert(vec![]);
                    files.push(file.clone());
                }
            }
        }
    }

    println!("id_to_files = {:?}", id_to_files);
    let mut not_found = true;
    while not_found {
        let (id_mem, tposmap_len_mem) = {
            match snapshot_to_tposmap.last_key_value() {
                Some(pair) => (*pair.0, pair.1.len() as u64),
                None => (0, 0),
            }
        };

        let (id_disk, tposmap_len_disk) = {
            match id_to_len.last_key_value() {
                Some(pair) => (*pair.0, *pair.1),
                None => (0, 0),
            }
        };

        if id_mem != 0 && id_mem >= id_disk {
            if tposmap_len_mem == size_triv_pos_map {
                not_found = false;
                let tposmap = snapshot_to_tposmap.get(&id_mem).unwrap();
                //Notice: should be consistent with enclave/src/oram_manager/position_map
                let lifetime_id_buf: [u8; 8] = (&tposmap[48..56]).try_into().unwrap();
                println!("id_mem = {:?}", id_mem);
                unsafe {
                    *snapshot_id = id_mem;
                    *lifetime_id_from_meta = u64::from_ne_bytes(lifetime_id_buf);
                }
            } else {
                snapshot_to_tposmap.remove(&id_mem);
                SNAPSHOT_TO_STASH.lock().unwrap().remove(&id_mem);
                SNAPSHOT_TO_TREETOP.lock().unwrap().remove(&id_mem);
                SNAPSHOT_TO_MROOTS.lock().unwrap().remove(&id_mem);
            }
        } else if id_disk != 0 {
            if tposmap_len_disk == size_triv_pos_map {
                not_found = false;
                let mut tposmap = Vec::new();
                if let Ok(mut f) = File::open(&PathBuf::from(
                    "trivial_posmap_".to_string() + &id_disk.to_string(),
                )) {
                    f.read_to_end(&mut tposmap).unwrap();
                };
                //Notice: should be consistent with enclave/src/oram_manager/position_map
                let lifetime_id_buf: [u8; 8] = (&tposmap[48..56]).try_into().unwrap();
                println!("id_disk = {:?}", id_disk);
                unsafe {
                    *snapshot_id = id_disk;
                    *lifetime_id_from_meta = u64::from_ne_bytes(lifetime_id_buf);
                }
                //remove other files
                id_to_files.remove(&id_disk);
                for (_, files) in id_to_files.iter() {
                    for file in files {
                        remove_file(file).unwrap();
                    }
                }
            } else {
                id_to_len.remove(&id_disk);
                //remove related files
                for file in id_to_files.get(&id_disk).unwrap() {
                    remove_file(file).unwrap();
                }
            }
        } else {
            unsafe { *snapshot_id = 0 };
            not_found = false;
        }
    }
    LAST_ON_DISK_SNAPSHOT_ID.store(unsafe { *snapshot_id }, Ordering::SeqCst);
}

#[no_mangle]
pub extern "C" fn persist_stash(
    new_stash_data: *const u8,
    new_stash_data_len: usize,
    new_stash_meta: *const u8,
    new_stash_meta_len: usize,
    level: u32,
    new_snapshot_id: u64,
    is_volatile: u8,
) {
    println!(
        "begin persist_stash level = {:?}, new_snapshot_id = {:?}",
        level, new_snapshot_id
    );
    let new_stash_data = unsafe { slice::from_raw_parts(new_stash_data, new_stash_data_len) };
    let new_stash_meta = unsafe { slice::from_raw_parts(new_stash_meta, new_stash_meta_len) };
    let mut state_map = SNAPSHOT_TO_STASH.lock().unwrap();
    let state = state_map.entry(new_snapshot_id).or_insert(BTreeMap::new());
    state.insert(level, (new_stash_data.to_vec(), new_stash_meta.to_vec()));

    if is_volatile == 0 {
        let stash_data_file_name = PathBuf::from(
            "stash_data_".to_string() + &new_snapshot_id.to_string() + "-" + &level.to_string(),
        );
        let stash_meta_file_name = PathBuf::from(
            "stash_meta_".to_string() + &new_snapshot_id.to_string() + "-" + &level.to_string(),
        );
        if let Ok(mut f) = File::create(&stash_data_file_name) {
            f.write_all(new_stash_data).unwrap();
        };
        if let Ok(mut f) = File::create(&stash_meta_file_name) {
            f.write_all(new_stash_meta).unwrap();
        };
    }
    println!("end persist_stash");
}

#[no_mangle]
pub extern "C" fn recover_stash(
    stash_data: *mut u8,
    stash_data_len: usize,
    stash_meta: *mut u8,
    stash_meta_len: usize,
    level: u32,
    snapshot_id: u64,
) {
    println!(
        "begin recover_stash level = {:?}, new_snapshot_id = {:?}",
        level, snapshot_id
    );
    let stash_data = unsafe { slice::from_raw_parts_mut(stash_data, stash_data_len) };
    let stash_meta = unsafe { slice::from_raw_parts_mut(stash_meta, stash_meta_len) };
    let state_map = SNAPSHOT_TO_STASH.lock().unwrap();
    let state = state_map.get(&snapshot_id);
    match state {
        Some(all_stash) => {
            let (data, meta) = all_stash.get(&level).unwrap();
            stash_data.copy_from_slice(data);
            stash_meta.copy_from_slice(meta);
        }
        None => {
            let stash_data_file_name = PathBuf::from(
                "stash_data_".to_string() + &snapshot_id.to_string() + "-" + &level.to_string(),
            );
            let stash_meta_file_name = PathBuf::from(
                "stash_meta_".to_string() + &snapshot_id.to_string() + "-" + &level.to_string(),
            );

            if let Ok(mut f) = File::open(&stash_data_file_name) {
                f.read_exact(stash_data).unwrap();
            };
            if let Ok(mut f) = File::open(&stash_meta_file_name) {
                f.read_exact(stash_meta).unwrap();
            };
        }
    }
    println!("end recover_stash");
}

#[no_mangle]
pub extern "C" fn persist_treetop(
    new_data: *const u8,
    new_data_len: usize,
    new_meta: *const u8,
    new_meta_len: usize,
    level: u32,
    new_snapshot_id: u64,
    is_volatile: u8,
) {
    println!(
        "begin persist_treetop level = {:?}, new_snapshot_id = {:?}",
        level, new_snapshot_id
    );
    let new_data = unsafe { slice::from_raw_parts(new_data, new_data_len) };
    let new_meta = unsafe { slice::from_raw_parts(new_meta, new_meta_len) };
    let mut state_map = SNAPSHOT_TO_TREETOP.lock().unwrap();
    let state = state_map.entry(new_snapshot_id).or_insert(BTreeMap::new());
    state.insert(level, (new_data.to_vec(), new_meta.to_vec()));

    if is_volatile == 0 {
        let data_file_name = PathBuf::from(
            "treetop_data_".to_string() + &new_snapshot_id.to_string() + "-" + &level.to_string(),
        );
        let meta_file_name = PathBuf::from(
            "treetop_meta_".to_string() + &new_snapshot_id.to_string() + "-" + &level.to_string(),
        );
        if let Ok(mut f) = File::create(&data_file_name) {
            f.write_all(new_data).unwrap();
        };
        if let Ok(mut f) = File::create(&meta_file_name) {
            f.write_all(new_meta).unwrap();
        };
    }
    println!("end persist_treetop");
}

#[no_mangle]
pub extern "C" fn recover_treetop(
    data: *mut u8,
    data_len: usize,
    meta: *mut u8,
    meta_len: usize,
    level: u32,
    snapshot_id: u64,
) {
    println!(
        "begin recover_treetop level = {:?}, new_snapshot_id = {:?}",
        level, snapshot_id
    );
    let data = unsafe { slice::from_raw_parts_mut(data, data_len) };
    let meta = unsafe { slice::from_raw_parts_mut(meta, meta_len) };
    let state_map = SNAPSHOT_TO_TREETOP.lock().unwrap();
    let state = state_map.get(&snapshot_id);
    match state {
        Some(all_treetop) => {
            let (d, m) = all_treetop.get(&level).unwrap();
            data.copy_from_slice(d);
            meta.copy_from_slice(m);
        }
        None => {
            let data_file_name = PathBuf::from(
                "treetop_data_".to_string() + &snapshot_id.to_string() + "-" + &level.to_string(),
            );
            let meta_file_name = PathBuf::from(
                "treetop_meta_".to_string() + &snapshot_id.to_string() + "-" + &level.to_string(),
            );

            if let Ok(mut f) = File::open(&data_file_name) {
                f.read_exact(data).unwrap();
            };
            if let Ok(mut f) = File::open(&meta_file_name) {
                f.read_exact(meta).unwrap();
            };
        }
    }
    println!("end recover_treetop");
}

#[no_mangle]
pub extern "C" fn persist_merkle_roots(
    new_roots: *const u8,
    new_roots_len: usize,
    level: u32,
    new_snapshot_id: u64,
    is_volatile: u8,
) {
    println!(
        "begin persist_merkle_roots level = {:?}, new_snapshot_id = {:?}",
        level, new_snapshot_id
    );
    let new_roots = unsafe { slice::from_raw_parts(new_roots, new_roots_len) };
    let mut state_map = SNAPSHOT_TO_MROOTS.lock().unwrap();
    let state = state_map.entry(new_snapshot_id).or_insert(BTreeMap::new());
    state.insert(level, new_roots.to_vec());

    if is_volatile == 0 {
        let data_file_name = PathBuf::from(
            "merkle_roots_".to_string() + &new_snapshot_id.to_string() + "-" + &level.to_string(),
        );
        if let Ok(mut f) = File::create(&data_file_name) {
            f.write_all(new_roots).unwrap();
        };
    }
    println!("end persist_merkle_roots");
}

#[no_mangle]
pub extern "C" fn recover_merkle_roots(
    roots: *mut u8,
    roots_len: usize,
    level: u32,
    snapshot_id: u64,
) {
    println!(
        "begin recover_merkle_roots level = {:?}, new_snapshot_id = {:?}",
        level, snapshot_id
    );
    let roots = unsafe { slice::from_raw_parts_mut(roots, roots_len) };
    let state_map = SNAPSHOT_TO_MROOTS.lock().unwrap();
    let state = state_map.get(&snapshot_id);
    match state {
        Some(all_roots) => {
            let d = all_roots.get(&level).unwrap();
            roots.copy_from_slice(d);
        }
        None => {
            let data_file_name = PathBuf::from(
                "merkle_roots_".to_string() + &snapshot_id.to_string() + "-" + &level.to_string(),
            );
            if let Ok(mut f) = File::open(&data_file_name) {
                f.read_exact(roots).unwrap();
            };
        }
    }
    println!("end recover_merkle_roots");
}

#[no_mangle]
pub extern "C" fn persist_trivial_posmap(
    new_posmap: *const u8,
    new_posmap_len: usize,
    new_snapshot_id: u64,
    is_volatile: u8,
) {
    println!(
        "begin persist_trivial_posmap new_snapshot_id = {:?}",
        new_snapshot_id
    );
    let new_posmap = unsafe { slice::from_raw_parts(new_posmap, new_posmap_len) };
    let mut state_map = SNAPSHOT_TO_TPOSMAP.lock().unwrap();
    state_map.insert(new_snapshot_id, new_posmap.to_vec());

    // Remove stale in-memory states
    SNAPSHOT_TO_STASH
        .lock()
        .unwrap()
        .retain(|&k, _| k == new_snapshot_id);
    SNAPSHOT_TO_TREETOP
        .lock()
        .unwrap()
        .retain(|&k, _| k == new_snapshot_id);
    SNAPSHOT_TO_MROOTS
        .lock()
        .unwrap()
        .retain(|&k, _| k == new_snapshot_id);
    state_map.retain(|&k, _| k == new_snapshot_id);

    if is_volatile == 0 {
        let data_file_name =
            PathBuf::from("trivial_posmap_".to_string() + &new_snapshot_id.to_string());
        if let Ok(mut f) = File::create(&data_file_name) {
            f.write_all(new_posmap).unwrap();
        };
        LAST_ON_DISK_SNAPSHOT_ID.store(new_snapshot_id, Ordering::SeqCst);
        // Delete stale files
        // First step: read from disk
        let path = Path::new("./");
        let mut id_to_files = BTreeMap::new();
        for entry in fs::read_dir(path).expect("reading directory fails") {
            if let Ok(entry) = entry {
                let file = entry.path();
                let filename = file.to_str().unwrap();

                if filename.contains("trivial_posmap_") {
                    lazy_static! {
                        static ref RE: Regex = Regex::new(r"trivial_posmap_(\d+)").unwrap();
                    }
                    for cap in RE.captures_iter(filename) {
                        let id = (&cap[1]).parse::<u64>().unwrap();
                        let files = id_to_files.entry(id).or_insert(vec![]);
                        files.push(file.clone());
                    }
                } else {
                    lazy_static! {
                        static ref RE: Regex = Regex::new(r"(\d+)-(\d+)").unwrap();
                    }
                    for cap in RE.captures_iter(filename) {
                        let id = (&cap[1]).parse::<u64>().unwrap();
                        let files = id_to_files.entry(id).or_insert(vec![]);
                        files.push(file.clone());
                    }
                }
            }
        }
        // Second step: remove
        id_to_files.remove(&new_snapshot_id);
        for (_, files) in id_to_files.iter() {
            for file in files {
                remove_file(file).unwrap();
            }
        }
    }
    println!("end persist_trivial_posmap");
}

#[no_mangle]
pub extern "C" fn recover_trivial_posmap(posmap: *mut u8, posmap_len: usize, snapshot_id: u64) {
    println!(
        "begin recover_trivial_posmap new_snapshot_id = {:?}",
        snapshot_id
    );
    let posmap = unsafe { slice::from_raw_parts_mut(posmap, posmap_len) };
    let state_map = SNAPSHOT_TO_TPOSMAP.lock().unwrap();
    let state = state_map.get(&snapshot_id);
    match state {
        Some(d) => {
            posmap.copy_from_slice(d);
        }
        None => {
            let data_file_name =
                PathBuf::from("trivial_posmap_".to_string() + &snapshot_id.to_string());
            if let Ok(mut f) = File::open(&data_file_name) {
                f.read_exact(posmap).unwrap();
            };
        }
    }
    println!("end recover_trivial_posmap");
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
    assert_eq!(shuffle_id, SHUFFLE_MANAGER_ID.load(Ordering::SeqCst));
    let manager = unsafe {
        (core::mem::transmute::<_, *mut ShuffleManager>(shuffle_id))
            .as_mut()
            .unwrap()
    };
    //TODO: may need lock
    let data = unsafe { core::slice::from_raw_parts_mut(data, data_size) };
    let meta = unsafe { core::slice::from_raw_parts_mut(meta, meta_size) };
    manager.pull_buckets(b_idx, e_idx, data, meta);
}

#[no_mangle]
pub extern "C" fn shuffle_pull_bin(
    shuffle_id: u64,
    tid: usize,
    cur_bin_num: usize,
    bin_type: u8,
    bin_size: *mut usize,
    data_item_size: usize,
    meta_item_size: usize,
    has_data: u8,
    has_meta: u8,
    nonce_size: usize,
    hash_size: usize,
    data_ptr: *mut usize,
    meta_ptr: *mut usize,
    nonce_ptr: *mut usize,
    hash_ptr: *mut usize,
) {
    assert_eq!(shuffle_id, SHUFFLE_MANAGER_ID.load(Ordering::SeqCst));
    let manager = unsafe {
        (core::mem::transmute::<_, *mut ShuffleManager>(shuffle_id))
            .as_mut()
            .unwrap()
    };
    //TODO: may need lock
    let bin_size = unsafe { bin_size.as_mut().unwrap() };
    manager.pull_bin(
        tid,
        cur_bin_num,
        bin_type,
        bin_size,
        data_item_size,
        meta_item_size,
        has_data != 0,
        has_meta != 0,
        nonce_size,
        hash_size,
    );
    let mut tmp_buf = manager.tmp_buf[tid].lock().unwrap();
    unsafe {
        *data_ptr = (&mut tmp_buf.0) as *mut Vec<u8> as usize;
        *meta_ptr = (&mut tmp_buf.1) as *mut Vec<u8> as usize;
        *nonce_ptr = (&mut tmp_buf.2) as *mut Vec<u8> as usize;
        *hash_ptr = (&mut tmp_buf.3) as *mut Vec<u8> as usize;
    }
}

#[no_mangle]
pub extern "C" fn shuffle_push_buckets_pre(
    shuffle_id: u64,
    tid: usize,
    data_size: usize,
    meta_size: usize,
    data_ptr: *mut usize,
    meta_ptr: *mut usize,
) {
    assert_eq!(shuffle_id, SHUFFLE_MANAGER_ID.load(Ordering::SeqCst));
    let manager = unsafe {
        (core::mem::transmute::<_, *mut ShuffleManager>(shuffle_id))
            .as_mut()
            .unwrap()
    };
    let mut tmp_buf = manager.tmp_buf[tid].lock().unwrap();
    *tmp_buf = (
        vec![0u8; data_size],
        vec![0u8; meta_size],
        Vec::new(),
        Vec::new(),
    );
    unsafe {
        *data_ptr = (&mut tmp_buf.0) as *mut Vec<u8> as usize;
        *meta_ptr = (&mut tmp_buf.1) as *mut Vec<u8> as usize;
    }
}

#[no_mangle]
pub extern "C" fn shuffle_push_buckets(shuffle_id: u64, tid: usize, b_idx: usize, e_idx: usize) {
    assert_eq!(shuffle_id, SHUFFLE_MANAGER_ID.load(Ordering::SeqCst));
    let manager = unsafe {
        (core::mem::transmute::<_, *mut ShuffleManager>(shuffle_id))
            .as_mut()
            .unwrap()
    };
    //TODO: may need lock
    manager.push_buckets(tid, b_idx, e_idx);
}

#[no_mangle]
pub extern "C" fn shuffle_push_bin_pre(
    shuffle_id: u64,
    tid: usize,
    data_size: usize,
    meta_size: usize,
    nonce_size: usize,
    hash_size: usize,
    data_ptr: *mut usize,
    meta_ptr: *mut usize,
    nonce_ptr: *mut usize,
    hash_ptr: *mut usize,
) {
    assert_eq!(shuffle_id, SHUFFLE_MANAGER_ID.load(Ordering::SeqCst));
    let manager = unsafe {
        (core::mem::transmute::<_, *mut ShuffleManager>(shuffle_id))
            .as_mut()
            .unwrap()
    };
    let mut tmp_buf = manager.tmp_buf[tid].lock().unwrap();
    *tmp_buf = (
        vec![0u8; data_size],
        vec![0u8; meta_size],
        vec![0u8; nonce_size],
        vec![0u8; hash_size],
    );
    unsafe {
        *data_ptr = (&mut tmp_buf.0) as *mut Vec<u8> as usize;
        *meta_ptr = (&mut tmp_buf.1) as *mut Vec<u8> as usize;
        *nonce_ptr = (&mut tmp_buf.2) as *mut Vec<u8> as usize;
        *hash_ptr = (&mut tmp_buf.3) as *mut Vec<u8> as usize;
    }
}

#[no_mangle]
pub extern "C" fn shuffle_push_bin(shuffle_id: u64, tid: usize, cur_bin_num: usize, bin_type: u8) {
    assert_eq!(shuffle_id, SHUFFLE_MANAGER_ID.load(Ordering::SeqCst));
    let manager = unsafe {
        (core::mem::transmute::<_, *mut ShuffleManager>(shuffle_id))
            .as_mut()
            .unwrap()
    };
    //TODO: may need lock
    manager.push_bin(tid, cur_bin_num, bin_type);
}

#[no_mangle]
pub extern "C" fn shuffle_push_tmp_posmap(
    data_size: usize,
    nonce_size: usize,
    hash_size: usize,
    data_ptr: *mut usize,
    nonce_ptr: *mut usize,
    hash_ptr: *mut usize,
) {
    let mut t = TMP_POS_MAP.lock().unwrap();
    t.0.resize(data_size, 0);
    t.1.resize(nonce_size, 0);
    t.2.resize(hash_size, 0);
    unsafe {
        *data_ptr = (&mut t.0) as *mut Vec<u8> as usize;
        *nonce_ptr = (&mut t.1) as *mut Vec<u8> as usize;
        *hash_ptr = (&mut t.2) as *mut Vec<u8> as usize;
    }
}
#[no_mangle]
pub extern "C" fn shuffle_pull_tmp_posmap(
    data_ptr: *mut usize,
    nonce_ptr: *mut usize,
    hash_ptr: *mut usize,
) {
    let t = TMP_POS_MAP.lock().unwrap();
    unsafe {
        *data_ptr = (&t.0) as *const Vec<u8> as usize;
        *nonce_ptr = (&t.1) as *const Vec<u8> as usize;
        *hash_ptr = (&t.2) as *const Vec<u8> as usize;
    }
}

#[no_mangle]
pub extern "C" fn shuffle_release_tmp_posmap() {
    *TMP_POS_MAP.lock().unwrap() = (Vec::new(), Vec::new(), Vec::new());
}

#[no_mangle]
pub extern "C" fn bin_switch(shuffle_id: u64) {
    assert_eq!(shuffle_id, SHUFFLE_MANAGER_ID.load(Ordering::SeqCst));
    let manager = unsafe {
        (core::mem::transmute::<_, *mut ShuffleManager>(shuffle_id))
            .as_mut()
            .unwrap()
    };
    //TODO: may need lock
    manager.bin_switch();
}

#[no_mangle]
pub extern "C" fn set_fixed_bin_size(
    shuffle_id: u64,
    data_bin_size: u64,
    meta_bin_size: u64,
    src_bin_size: u64,
    dst_bin_size: u64,
) {
    assert_eq!(shuffle_id, SHUFFLE_MANAGER_ID.load(Ordering::SeqCst));
    let manager = unsafe {
        (core::mem::transmute::<_, *mut ShuffleManager>(shuffle_id))
            .as_mut()
            .unwrap()
    };
    //TODO: may need lock
    manager.set_fixed_bin_size(data_bin_size, meta_bin_size, src_bin_size, dst_bin_size);
}

#[no_mangle]
pub extern "C" fn clear_content(shuffle_id: u64) {
    assert_eq!(shuffle_id, SHUFFLE_MANAGER_ID.load(Ordering::SeqCst));
    let manager = unsafe {
        (core::mem::transmute::<_, *mut ShuffleManager>(shuffle_id))
            .as_mut()
            .unwrap()
    };
    //TODO: may need lock
    manager.clear_content();
}

// This module is only used in debug builds, it allows us to ensure that an id
// is valid before we cast it to a pointer, and give nicer asserts if it isn't
#[cfg(debug_assertions)]
mod debug_checks {
    use std::{collections::BTreeSet, sync::Mutex};

    pub fn add_id(id: u64) {
        let mut lk = VALID_IDS.lock().unwrap();
        assert!(!lk.contains(&id), "id already exists");
        lk.insert(id);
    }
    pub fn remove_id(id: u64) {
        let mut lk = VALID_IDS.lock().unwrap();
        assert!(lk.contains(&id), "can't remove non-existant id");
        lk.remove(&id);
    }
    pub fn check_id(id: u64) {
        let lk = VALID_IDS.lock().unwrap();
        assert!(lk.contains(&id), "invalid id passed from enclave");
    }

    lazy_static::lazy_static! {
        static ref VALID_IDS: Mutex<BTreeSet<u64>> = Mutex::new(Default::default());
    }
}
