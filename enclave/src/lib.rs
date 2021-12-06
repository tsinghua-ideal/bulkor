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
use std::sync::{atomic::Ordering, Arc, SgxMutex as Mutex, SgxRwLock as RwLock};
use std::thread;
use std::time::Instant;
use std::untrusted::time::InstantEx;
use std::vec::Vec;


use aes::{
    cipher::{NewCipher, StreamCipher},
    Aes256Ctr,
};
use aligned_cmov::{typenum::{U1024, U4, U4096, U64, Unsigned}, CMov};

mod allocator;
use allocator::Allocator;
mod atomicptr_wrapper;
use atomicptr_wrapper::AtomicPtrWrapper;
use sgx_types::sgx_status_t;
mod custom_thread;
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
/// Parameters that correspond to PathORAM4096Z4Creator
type StorageBlockSize = U1024;
type StorageBucketSize = U4096;
type StorageZ = U4;
type StorageMetaSize = U64;
const STASH_SIZE: usize = 16;
/// Bucket size = U4096, Z = U4, MetaSize = U64
type ORAMCreatorClass = PathORAM4096Z4Creator<RngType, OcallORAMStorageCreator>;
type ORAMClass = PathORAM<
    StorageBlockSize,
    StorageZ,
    OcallORAMStorage<StorageBucketSize, StorageMetaSize>,
    RngType,
>;

lazy_static! {
    /// Initialize ORAM if not yet
    static ref ORAM_OBJ: Mutex<Option<ORAMClass>> = Mutex::new(None);
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
    return cur_n;
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
    let Query {
        op_type,
        idx,
        new_val,
    } = Query::<StorageBlockSize>::decrypt_from(bytes);
    let mut lk = ORAM_OBJ.lock().unwrap();
    let cur_oram = lk.as_mut().unwrap();
    let mut data = cur_oram.access(idx, |val| {
        let retval = val.clone();
        val.cmov(op_type, &new_val);
        retval
    });

    let nonce = query::encrypt_res(&mut data);
    if resp_len  < nonce.len() + data.len() {
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }
    let (resp_nonce, resp_data) = resp_slice.split_at_mut(NonceSize::USIZE);
    resp_nonce.copy_from_slice(nonce.as_slice());
    resp_data.copy_from_slice(data.as_slice());

    return sgx_status_t::SGX_SUCCESS;
}
