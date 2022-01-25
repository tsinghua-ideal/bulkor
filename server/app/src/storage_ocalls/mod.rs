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

use regex::Regex;
use std::{
    alloc::{alloc, alloc_zeroed, dealloc, Layout},
    boxed::Box,
    cmp::max,
    collections::BTreeMap,
    fs::{self, remove_file, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    slice,
    string::ToString,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        Mutex,
    },
};

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
    pub static ref TREETOP_CACHING_THRESHOLD_LOG2: AtomicU32 = AtomicU32::new(33u32); // 8 GB

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
}

/// Resources held on untrusted side in connection to an allocation request by
/// enclave
///
/// This is not actually part of the public interface of the crate, the only
/// thing exported by the crate is the enclave EDL functions
struct UntrustedAllocation {
    /// The level of this ORAM
    level: u32,
    /// The number of data and meta items stored in this allocation
    count_in_mem: usize,
    // The maximum count for the treetop storage in untrusted memory,
    // based on what we loaded from TREETOP_CACHING_THRESHOLD_LOG2 at construction time
    // This must never change after construction.
    treetop_max_count: u64,
    /// The size of a data item in bytes
    data_item_size: usize,
    /// The size of a meta item in bytes
    meta_item_size: usize,
    /// The pointer to the data items
    data_pointer: *mut u8,
    /// The pointer to the meta items
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
    let num_bytes = (count * (data_item_size + meta_item_size)) as u64;
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
        count: usize,
        data_item_size: usize,
        meta_item_size: usize,
    ) -> Self {
        let treetop_max_count: u64 = max(
            2u64,
            (1u64 << TREETOP_CACHING_THRESHOLD_LOG2.load(Ordering::SeqCst)) / data_item_size as u64,
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

        let data_file_name = PathBuf::from("data_".to_string() + &level.to_string());
        let meta_file_name = PathBuf::from("meta_".to_string() + &level.to_string());
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

        let data_pointer =
            unsafe { alloc(Layout::from_size_align(count_in_mem * data_item_size, 8).unwrap()) };
        if data_pointer.is_null() {
            panic!(
                "Could not allocate memory for data segment: {}",
                count_in_mem * data_item_size
            )
        }
        let meta_pointer = unsafe {
            alloc_zeroed(Layout::from_size_align(count_in_mem * meta_item_size, 8).unwrap())
        };
        if meta_pointer.is_null() {
            panic!(
                "Could not allocate memory for meta segment: {}",
                count_in_mem * meta_item_size
            )
        }
        if snapshot_id == 0 {
            let mut data_file_len = data_file.metadata().unwrap().len() as usize;
            let mut meta_file_len = meta_file.metadata().unwrap().len() as usize;
            // Initialization is not finished, so do the initialization
            let zeros = vec![0u8; data_item_size];
            while data_file_len < count * data_item_size {
                data_file.write_all(&zeros).unwrap();
                data_file_len += data_item_size;
            }
            while meta_file_len < count * meta_item_size {
                meta_file.write_all(&zeros[..meta_item_size]).unwrap();
                meta_file_len += meta_item_size;
            }
            data_file.flush().unwrap();
            meta_file.flush().unwrap();
        } else {
            let data = unsafe {
                core::slice::from_raw_parts_mut(data_pointer, count_in_mem * data_item_size)
            };
            let meta = unsafe {
                core::slice::from_raw_parts_mut(meta_pointer, count_in_mem * meta_item_size)
            };
            //naive recovery
            //TODO: optimization is needed
            data_file.read_exact_at(data, 0).unwrap();
            meta_file.read_exact_at(meta, 0).unwrap();
        }

        let critical_section_flag = AtomicBool::new(false);
        let checkout_flag = AtomicBool::new(false);

        Self {
            level,
            count_in_mem,
            treetop_max_count,
            data_item_size,
            meta_item_size,
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
                Layout::from_size_align_unchecked(self.count_in_mem * self.data_item_size, 8),
            );
            dealloc(
                self.meta_pointer as *mut u8,
                Layout::from_size_align_unchecked(self.count_in_mem * self.meta_item_size, 8),
            );
            let mem_kb =
                compute_mem_kb(self.count_in_mem, self.data_item_size, self.meta_item_size);
            TOTAL_MEM_FOOTPRINT_KB.fetch_sub(mem_kb, Ordering::SeqCst);
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
    count: u64,
    data_size: u64,
    meta_size: u64,
    id_out: *mut u64,
) {
    let mut m = LEVEL_TO_ORAM_INST.lock().unwrap();
    let id = m.entry(level).or_insert(0);
    println!("id = {:?}", id);
    if *id == 0 {
        println!(
            "before allocing, level = {:?}, snapshot_id = {:?}",
            level, snapshot_id
        );
        let result = Box::new(UntrustedAllocation::new(
            level,
            snapshot_id,
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
pub unsafe extern "C" fn persist_oram_storage(level: u32, snapshot_id: u64, id: u64) {
    println!("begin persist_oram_storage");
    #[cfg(debug_assertions)]
    debug_checks::check_id(id);
    let ptr: *const UntrustedAllocation = core::mem::transmute(id);
    assert!(
        !(*ptr).critical_section_flag.swap(true, Ordering::SeqCst),
        "Could not enter critical section when checking out storage"
    );
    assert_eq!(level, (*ptr).level);

    //TODO: snapshot id is not used yet
    //because we do not persist various versions
    let data_item_size = (*ptr).data_item_size;
    let meta_item_size = (*ptr).meta_item_size;
    let count_in_mem = (*ptr).count_in_mem;
    let data = core::slice::from_raw_parts((*ptr).data_pointer, count_in_mem * data_item_size);
    let meta = core::slice::from_raw_parts((*ptr).meta_pointer, count_in_mem * meta_item_size);

    //naive persistence
    //TODO: optimization is needed
    (*ptr).data_file.write_all_at(data, 0).unwrap();
    (*ptr).meta_file.write_all_at(meta, 0).unwrap();

    assert!(
        (*ptr).critical_section_flag.swap(false, Ordering::SeqCst),
        "Could not leave critical section when checking out storage"
    );
    println!("end persist_oram_storage");
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

    let data_item_size = (*ptr).data_item_size;
    let meta_item_size = (*ptr).meta_item_size;
    assert!(idx_len * data_item_size == databuf_len);
    assert!(idx_len * meta_item_size == metabuf_len);

    let indices = core::slice::from_raw_parts(idx, idx_len);
    let databuf_s = core::slice::from_raw_parts_mut(databuf, databuf_len);
    let metabuf_s = core::slice::from_raw_parts_mut(metabuf, metabuf_len);

    let first_treetop_index = indices
        .iter()
        .position(|idx| idx < &(*ptr).treetop_max_count)
        .expect("should be unreachable, at least one thing should be in the treetop");

    for (count, index) in indices.iter().enumerate() {
        let index = *index as usize;
        if count < first_treetop_index {
            (*ptr)
                .data_file
                .read_exact_at(
                    &mut databuf_s[data_item_size * count..data_item_size * (count + 1)],
                    (data_item_size * index) as u64,
                )
                .unwrap();
        } else {
            core::ptr::copy_nonoverlapping(
                (*ptr).data_pointer.add(data_item_size * index),
                databuf.add(data_item_size * count),
                data_item_size,
            );
        }
    }

    for (count, index) in indices.iter().enumerate() {
        let index = *index as usize;
        if count < first_treetop_index {
            (*ptr)
                .meta_file
                .read_exact_at(
                    &mut metabuf_s[meta_item_size * count..meta_item_size * (count + 1)],
                    (meta_item_size * index) as u64,
                )
                .unwrap();
        } else {
            core::ptr::copy_nonoverlapping(
                (*ptr).meta_pointer.add(meta_item_size * index),
                metabuf.add(meta_item_size * count),
                meta_item_size,
            );
        }
    }

    assert!(
        (*ptr).critical_section_flag.swap(false, Ordering::SeqCst),
        "Could not leave critical section when checking out storage"
    );
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
    let ptr: *const UntrustedAllocation = core::mem::transmute(id);
    assert!(
        !(*ptr).critical_section_flag.swap(true, Ordering::SeqCst),
        "Could not enter critical section when checking in storage"
    );
    assert!(
        (*ptr).checkout_flag.swap(false, Ordering::SeqCst),
        "illegal checkin"
    );

    let data_item_size = (*ptr).data_item_size;
    let meta_item_size = (*ptr).meta_item_size;
    assert!(idx_len * data_item_size == databuf_len);
    assert!(idx_len * meta_item_size == metabuf_len);

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
            (*ptr)
                .data_file
                .write_all_at(
                    &databuf_s[data_item_size * count..data_item_size * (count + 1)],
                    (data_item_size * index) as u64,
                )
                .unwrap();
        } else {
            core::ptr::copy_nonoverlapping(
                databuf.add(data_item_size * count),
                (*ptr).data_pointer.add(data_item_size * index),
                data_item_size,
            );
        }
    }

    for (count, index) in indices.iter().enumerate() {
        let index = *index as usize;
        if count < first_treetop_index {
            (*ptr)
                .meta_file
                .write_all_at(
                    &metabuf_s[meta_item_size * count..meta_item_size * (count + 1)],
                    (meta_item_size * index) as u64,
                )
                .unwrap();
        } else {
            core::ptr::copy_nonoverlapping(
                metabuf.add(meta_item_size * count),
                (*ptr).meta_pointer.add(meta_item_size * index),
                meta_item_size,
            );
        }
    }

    assert!(
        (*ptr).critical_section_flag.swap(false, Ordering::SeqCst),
        "Could not leave critical section when checking in storage"
    );
}

#[no_mangle]
pub extern "C" fn get_valid_snapshot_id(size_triv_pos_map: u64, snapshot_id: *mut u64) {
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
                unsafe { *snapshot_id = id_mem };
                println!("id_mem = {:?}", id_mem);
                not_found = false;
            } else {
                snapshot_to_tposmap.remove(&id_mem);
                SNAPSHOT_TO_STASH.lock().unwrap().remove(&id_mem);
                SNAPSHOT_TO_TREETOP.lock().unwrap().remove(&id_mem);
                SNAPSHOT_TO_MROOTS.lock().unwrap().remove(&id_mem);
            }
        } else if id_disk != 0 {
            if tposmap_len_disk == size_triv_pos_map {
                unsafe { *snapshot_id = id_disk };
                println!("id_disk = {:?}", id_disk);
                not_found = false;
                id_to_files.split_off(&id_disk);
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
            // TODO: the deletion of stale file
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
            // TODO: the deletion of stale file
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
            // TODO: the deletion of stale file
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

    if is_volatile == 0 {
        let data_file_name =
            PathBuf::from("trivial_posmap_".to_string() + &new_snapshot_id.to_string());
        if let Ok(mut f) = File::create(&data_file_name) {
            f.write_all(new_posmap).unwrap();
        };
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
            // TODO: the deletion of stale file
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
pub extern "C" fn release_states(snapshot_id: u64) {
    unimplemented!()
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
