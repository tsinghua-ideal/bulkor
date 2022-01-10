// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License..
#![crate_name = "oramenclave"]
#![crate_type = "staticlib"]
#![allow(non_snake_case)]
#![cfg_attr(not(target_env = "sgx"), no_std)]
#![cfg_attr(target_env = "sgx", feature(rustc_private))]
#![feature(box_syntax)]
#![feature(cell_update)]

#[cfg(not(target_env = "sgx"))]
#[macro_use]
extern crate sgx_tstd as std;
extern crate sgx_libc as libc;
#[macro_use]
extern crate lazy_static;

use std::slice;
use std::string::{String, ToString};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc::{channel, Receiver, Sender},
    Arc, SgxMutex as Mutex, SgxRwLock as RwLock,
};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use std::untrusted::time::InstantEx;
use std::vec::Vec;

use aes::{
    cipher::{NewCipher, StreamCipher},
    Aes256Ctr,
};
use aligned_cmov::{
    typenum::{Unsigned, U1024, U24, U4, U4096, U96},
    CMov,
};

mod allocator;
use allocator::Allocator;
mod atomicptr_wrapper;
use atomicptr_wrapper::AtomicPtrWrapper;
use sgx_types::sgx_status_t;
mod oram_manager;
use oram_manager::{PathORAM, PathORAM4096Z4Creator};
mod oram_storage;
use oram_storage::{OcallORAMStorage, OcallORAMStorageCreator};
mod oram_traits;
use oram_traits::{rng_maker, ORAMCreator, ORAM};
mod query;
use query::Query;
use rand_hc::Hc128Rng;
mod test_helper;
use test_helper::get_seeded_rng;

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

/// Cipher type. Anything implementing StreamCipher and NewCipher at 128
/// bit security should be acceptable
type CipherType = Aes256Ctr;
/// Parameters of the cipher as typedefs (which eases syntax)
type NonceSize = <CipherType as NewCipher>::NonceSize;
type KeySize = <CipherType as NewCipher>::KeySize;
/// Sometimes you need to have the type in scope to call trait functions
type RngType = Hc128Rng;
/// Parameters that correspond to PathORAM4096Z4Creator, should be consist with crate::oram_manager (DataMetaSize)
type StorageBlockSize = U1024;
type StorageBlockMetaSize = U24;
type StorageBucketSize = U4096;
type StorageBucketMetaSize = U96; //MetaSize(24)*Z(4)
type StorageZ = U4;
const STASH_SIZE: usize = 16;
/// Bucket size = U4096, Z = U4, MetaSize = U96
type ORAMCreatorClass = PathORAM4096Z4Creator<RngType, OcallORAMStorageCreator>;
type ORAMClass = PathORAM<
    StorageBlockSize,
    StorageBlockMetaSize,
    StorageZ,
    OcallORAMStorage<StorageBucketSize, StorageBucketMetaSize>,
    RngType,
>;

enum Message {
    Sync,
    Query(Vec<u8>),
    PosMap(Vec<u8>),
    Stash(Vec<u8>),
}

lazy_static! {
    /// Initialize ORAM if not yet
    static ref ORAM_OBJ: Mutex<Option<ORAMClass>> = Mutex::new(None);
    /// Counter
    static ref COUNTER: AtomicU64 = AtomicU64::new(0u64);
    /// Backend thread
    static ref HANDLE: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);
    /// Sender for logging
    static ref SENDER: Mutex<Option<Sender<Message>>> = Mutex::new(None);
}

/// Create new oram if not exists
/// TODO: recovery
#[no_mangle]
pub extern "C" fn ecall_create_oram(n: u64) -> u64 {
    let rng = get_seeded_rng();
    let new_oram = ORAMCreatorClass::create(n, STASH_SIZE, &mut rng_maker(rng));
    let mut lk = ORAM_OBJ.lock().unwrap();
    let is_some = lk.is_some();
    let mut cur_n = n;
    if is_some {
        cur_n = lk.as_ref().map(|cur_oram| cur_oram.len()).unwrap();
    } else {
        *lk = Some(new_oram);
    }
    let (sender, receiver) = channel();
    let join_handle = thread::spawn(move || {
        do_backend(receiver);
    });

    *HANDLE.lock().unwrap() = Some(join_handle);
    *SENDER.lock().unwrap() = Some(sender);
    return cur_n;
}

#[no_mangle]
pub extern "C" fn ecall_destroy_oram() {
    let mut lk = ORAM_OBJ.lock().unwrap();
    if lk.is_some() {
        let _old_oram = lk.take(); //call the destructor when leaving the scope
    }
    let sender = SENDER.lock().unwrap().take();
    if sender.is_some() {
        let _sender = sender.unwrap();
    }
    let hd = HANDLE.lock().unwrap().take();
    if hd.is_some() {
        hd.unwrap().join().unwrap();
    }
}

#[no_mangle]
pub extern "C" fn ecall_access(
    query: *mut u8,
    query_len: usize,
    resp: *mut u8,
    resp_len: usize,
) -> sgx_status_t {
    let bytes = unsafe { slice::from_raw_parts_mut(query, query_len) };
    let resp_slice = unsafe { slice::from_raw_parts_mut(resp, resp_len) };
    let query = Query::<StorageBlockSize>::decrypt_from(bytes);

    let cur_counter = COUNTER.fetch_add(1, Ordering::SeqCst);
    //logging
    let buf = query.encrypt_with_counter(cur_counter);
    let sender = get_sender();
    sender.send(Message::Query(buf)).unwrap();
    sender.send(Message::Sync).unwrap(); //ensure the completion of logging

    //processing
    let Query {
        op_type,
        idx,
        new_val,
    } = query;

    let mut lk = ORAM_OBJ.lock().unwrap();
    let cur_oram = lk.as_mut().unwrap();
    let mut data = cur_oram.access(idx, |val, counter| {
        let retval = val.clone();
        val.cmov(op_type, &new_val);
        counter.cmov(op_type, &(*counter + 1));
        retval
    });

    let nonce = query::encrypt_res(&mut data);
    if resp_len < nonce.len() + data.len() {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }
    let (resp_nonce, resp_data) = resp_slice.split_at_mut(NonceSize::USIZE);
    resp_nonce.copy_from_slice(nonce.as_slice());
    resp_data.copy_from_slice(data.as_slice());

    return sgx_status_t::SGX_SUCCESS;
}

fn get_sender() -> Sender<Message> {
    let lk = SENDER.lock().unwrap();
    lk.clone().unwrap()
}

fn do_backend(rx: Receiver<Message>) {
    let mut res;
    let mut cur_log_id = 1;

    while {
        res = rx.recv();
        res.is_ok()
    } {
        match res.unwrap() {
            Message::Sync => (),
            Message::PosMap(pos_map) => {}
            Message::Query(query) => {
                unsafe { log_query(cur_log_id, query.as_ptr(), query.len()) };
                ()
            }
            Message::Stash(stash) => {}
        }
    }
}

// This stuff must match edl file
extern "C" {
    fn log_query(id: u64, query: *const u8, query_size: usize);
    //fn persist_pos_map(pos_map: *const u8, pos_map_size: usize);
    //fn persist_stash(stash: *const u8, stash_size: usize);
}
