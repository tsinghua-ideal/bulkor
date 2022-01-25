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

use std::boxed::Box;
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sgxfs::{OpenOptions, SgxFile};
use std::slice;
use std::string::{String, ToString};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, SgxMutex as Mutex, SgxRwLock as RwLock,
};
use std::time::Instant;
use std::untrusted::time::InstantEx;
use std::vec::Vec;

use aes::{
    cipher::{NewCipher, StreamCipher},
    Aes128Ctr,
};
use aes_gcm::Aes128Gcm;
use aligned_cmov::{
    typenum::{Unsigned, U1024, U12, U24, U4, U4096, U96},
    CMov,
};

mod allocator;
use allocator::Allocator;
use sgx_types::sgx_status_t;
mod oram_manager;
use oram_manager::{PathORAM, PathORAM4096Z4Creator, POS_MAP_THRESHOLD};
mod oram_storage;
use oram_storage::{
    get_valid_snapshot_id, s_decrypt, s_encrypt_with_nonce, OcallORAMStorage,
    OcallORAMStorageCreator, ORAM_KEY,
};
mod oram_traits;
use oram_traits::{log2_ceil, rng_maker, ORAMCreator, ORAM};
mod query;
use query::{Query, ANSWER_SIZE, QUERY_KEY, QUERY_SIZE};
use rand_hc::Hc128Rng;
mod test_helper;
use test_helper::get_seeded_rng;

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

/// First level persistence: every 10 log batches
const FL_PERSIST_SIZE: u64 = 5;
/// Second level persistence: every 100 log batches
const SL_PERSIST_SIZE: u64 = 10;

/// Cipher type. Anything implementing StreamCipher and NewCipher at 128
/// bit security should be acceptable
type CipherType = Aes128Ctr;
type AuthCipherType = Aes128Gcm;
/// Parameters of the cipher as typedefs (which eases syntax)
type NonceSize = <CipherType as NewCipher>::NonceSize;
type AuthNonceSize = U12;
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

lazy_static! {
    /// Initialize ORAM if not yet
    static ref ORAM_OBJ: Mutex<Option<ORAMClass>> = Mutex::new(None);
    /// Batch counter
    static ref BATCH_CNT: AtomicU64 = AtomicU64::new(1u64);
    /// Query counter, aka, new snapshot id
    static ref QUERY_CNT: AtomicU64 = AtomicU64::new(0u64);
    /// Snapshot id, based on which the ORAMs are recovered or created
    static ref SNAPSHOT_ID: AtomicU64 = AtomicU64::new(0u64);
    /// Snapshot id to log position
    static ref ID_TO_LOG_POS: Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());
    /// Logging lock
    static ref LOG_LOCK: Mutex<()> = Mutex::new(());
}

/// Create new oram if not exists
#[no_mangle]
pub extern "C" fn ecall_create_oram(n: u64, results_ptr: *mut usize) -> u64 {
    let mut lk = ORAM_OBJ.lock().unwrap();

    let mut log = OpenOptions::new()
        .read(true)
        .open_ex("log", &ORAM_KEY.clone().into())
        .ok();
    let mut log_len = 0;
    let mut query_cnt = 0;
    log.as_mut().map(|l| {
        log_len = l.seek(SeekFrom::End(-8)).unwrap() + 8;
        let mut query_cnt_buf = [0; 8];
        l.read_exact(&mut query_cnt_buf).unwrap();
        query_cnt = u64::from_le_bytes(query_cnt_buf);
    });
    //TODO: check the log counter with the monotonic counter
    QUERY_CNT.store(query_cnt, Ordering::SeqCst);
    println!("loaded query_cnt = {:?}", query_cnt);

    let mut snapshot_id: u64 = 0;
    //Compute the size of trivial position map
    let mut size_triv_pos_map = n;
    while size_triv_pos_map > POS_MAP_THRESHOLD {
        size_triv_pos_map >>= log2_ceil(StorageBlockSize::U64) - 2;
    }
    //NOTE: should be consistent with `posmap_len` in position_map.rs
    size_triv_pos_map = NonceSize::U64 + 16 + 8 + 8 + size_triv_pos_map * 4;
    unsafe {
        //The last persisted state should be the trivial position map
        //So if the file is not broken, the newest valid snapshot id can be gotten
        //It is secure, because we further check the snapshot_id when loading metadata.
        //And it does not matter whether the snapshot id is the newest.
        get_valid_snapshot_id(size_triv_pos_map, &mut snapshot_id);
    }
    SNAPSHOT_ID.store(snapshot_id, Ordering::SeqCst);
    let rng = get_seeded_rng();
    let mut new_oram = ORAMCreatorClass::create(n, STASH_SIZE, &mut rng_maker(rng));

    //replay the log since the loaded snapshot id
    let log_pos = ID_TO_LOG_POS
        .lock()
        .unwrap()
        .remove(&snapshot_id)
        .unwrap_or(0);
    let queries_buf_cap = (log_len - log_pos) as usize;
    println!("queries_buf_cap = {:?}", queries_buf_cap);
    let mut queries_buf = Vec::with_capacity(queries_buf_cap);
    log.as_mut().map(|l| {
        l.seek(SeekFrom::Start(log_pos - 8)).unwrap();
        let mut snapshot_id_buf = [0; 8];
        l.read_exact(&mut snapshot_id_buf).unwrap();
        assert_eq!(snapshot_id, u64::from_le_bytes(snapshot_id_buf));
        l.read_exact(&mut queries_buf).unwrap();
    });
    assert_eq!(queries_buf_cap, queries_buf.len());

    ALLOCATOR.set_switch(true);
    let mut results = vec![];
    ALLOCATOR.set_switch(false);

    let mut cur_q = 0;
    while cur_q < queries_buf.len() {
        //read batch size
        let mut batch_size_buf = [0; 8];
        batch_size_buf.copy_from_slice(&queries_buf[cur_q..(cur_q + 8)]);
        let batch_size = u64::from_le_bytes(batch_size_buf) as usize;
        cur_q += 8;
        //process batch
        let mut answers = vec![0; batch_size * ANSWER_SIZE];
        process_batch(
            &mut new_oram,
            &queries_buf[cur_q..(cur_q + batch_size * QUERY_SIZE)],
            &mut answers,
        );
        cur_q += batch_size * QUERY_SIZE;
        // no need to store QUERY_CNT, since it has the newest cnt
        // let mut query_cnt_buf = [0; 8];
        // query_cnt_buf.copy_from_slice(&queries_buf[cur_q..(cur_q + 8)]);
        // QUERY_CNT.store(u64::from_le_bytes(query_cnt_buf), Ordering::SeqCst);
        cur_q += 8;

        //send results back
        ALLOCATOR.set_switch(true);
        results.extend_from_slice(&answers);
        ALLOCATOR.set_switch(false);
    }

    ALLOCATOR.set_switch(true);
    let ptr = Box::into_raw(Box::new(results)) as usize;
    ALLOCATOR.set_switch(false);
    unsafe { *results_ptr = ptr };

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
pub extern "C" fn ecall_release_results_space(results_ptr: usize) {
    ALLOCATOR.set_switch(true);
    unsafe {
        let _results = Box::from_raw(results_ptr as *mut Vec<u8>);
    }
    ALLOCATOR.set_switch(false);
}

#[no_mangle]
pub extern "C" fn ecall_destroy_oram() {
    let mut lk = ORAM_OBJ.lock().unwrap();
    if lk.is_some() {
        let _old_oram = lk.take(); //call the destructor when leaving the scope
    }
}

#[no_mangle]
pub extern "C" fn ecall_access(
    batch_size: usize,
    queries: *mut u8,
    queries_len: usize,
    answers: *mut u8,
    answers_len: usize,
) -> sgx_status_t {
    let bytes = unsafe { slice::from_raw_parts_mut(queries, queries_len) };
    let answers_slice = unsafe { slice::from_raw_parts_mut(answers, answers_len) };
    assert_eq!(
        batch_size,
        bytes
            .chunks_mut(QUERY_SIZE)
            .into_iter()
            .map(|b| s_decrypt(&QUERY_KEY, b, 16))
            .count()
    );
    let cur_batch_cnt = BATCH_CNT.fetch_add(1, Ordering::SeqCst);
    //logging, think about hold the log somewhere, avoid opening it every time
    let llk = LOG_LOCK.lock().unwrap();
    let mut cur_query_cnt = QUERY_CNT.load(Ordering::SeqCst);
    cur_query_cnt += batch_size as u64;
    QUERY_CNT.store(cur_query_cnt, Ordering::SeqCst);
    println!("cur_query_cnt = {:?}", cur_query_cnt);

    let cur_log_pos = {
        let mut log = OpenOptions::new()
            .append(true)
            .open_ex("log", &ORAM_KEY.clone().into())
            .unwrap();
        log.write_all(&batch_size.to_le_bytes()).unwrap();
        log.write_all(bytes).unwrap(); //bytes have been decrypted
        log.write_all(&cur_query_cnt.to_le_bytes()).unwrap();

        //TODO: some other information may be logged per batch
        log.stream_position().unwrap()
    };
    //Bound the pos with new snapshot id
    ID_TO_LOG_POS
        .lock()
        .unwrap()
        .insert(cur_query_cnt, cur_log_pos);
    let mut qlk = ORAM_OBJ.lock().unwrap();
    drop(llk); //release the loglock
    {
        let cur_oram = qlk.as_mut().unwrap();
        process_batch(cur_oram, bytes, answers_slice);
        if cur_batch_cnt % SL_PERSIST_SIZE == 0 {
            println!("begin SL_PERSIST, cur_query_cnt = {:?}", cur_query_cnt);
            cur_oram.persist(cur_query_cnt, false);
        } else if cur_batch_cnt % FL_PERSIST_SIZE == 0 {
            println!("begin FL_PERSIST, cur_query_cnt = {:?}", cur_query_cnt);
            cur_oram.persist(cur_query_cnt, true);
        }
    };
    drop(qlk); //release the querylock

    return sgx_status_t::SGX_SUCCESS;
}

fn process_batch(oram: &mut ORAMClass, batch_queries: &[u8], batch_answers: &mut [u8]) {
    let ms = NonceSize::USIZE + 16 + 8 + 8; //meta size
    let mut cur_queries_pos = 0;
    let mut cur_answers_pos = 0;
    let queries = batch_queries
        .chunks(QUERY_SIZE)
        .into_iter()
        .map(|b| Query::<StorageBlockSize>::from_slice(b))
        .collect::<Vec<_>>();
    for query in queries {
        let Query {
            op_type,
            idx,
            new_val,
        } = query;
        let data = oram.access(idx, |val, counter| {
            let retval = val.clone();
            val.cmov(op_type, &new_val);
            counter.cmov(op_type, &(*counter + 1));
            retval
        });
        (&mut batch_answers[cur_answers_pos..(cur_answers_pos + ms)]) //fill the nonce, hash (to be replaced later), client id and query per client id
            .copy_from_slice(&batch_queries[cur_queries_pos..(cur_queries_pos + ms)]);
        (&mut batch_answers[(cur_answers_pos + ms)..(cur_answers_pos + ANSWER_SIZE)]) //fill the data
            .copy_from_slice(&data);
        s_encrypt_with_nonce(
            &QUERY_KEY,
            &mut batch_answers[cur_answers_pos..(cur_answers_pos + ANSWER_SIZE)],
            16,
        ); //encrypt the data and compute the hash
        cur_queries_pos += QUERY_SIZE;
        cur_answers_pos += ANSWER_SIZE;
    }
    assert_eq!(cur_answers_pos, batch_answers.len());
}