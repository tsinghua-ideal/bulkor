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

extern crate sgx_types;
extern crate sgx_urts;
use sgx_types::*;
use sgx_urts::SgxEnclave;
#[macro_use]
extern crate lazy_static;
use aes::{cipher::NewCipher, Aes256Ctr};
use aligned_cmov::{typenum::{Unsigned, U1024}, A64Bytes};
use rand_core::{CryptoRng, RngCore, SeedableRng};
use rand_hc::Hc128Rng;

use std::collections::BTreeMap;
use std::time::Instant;

mod allocator;
mod logger;
use logger::LogLevel;
mod query;
use query::{a64_bytes, Query};
mod storage_ocalls;

/// Cipher type. Anything implementing StreamCipher and NewCipher at 128
/// bit security should be acceptable
type CipherType = Aes256Ctr;
/// Parameters of the cipher as typedefs (which eases syntax)
type NonceSize = <CipherType as NewCipher>::NonceSize;
type KeySize = <CipherType as NewCipher>::KeySize;
type RngType = Hc128Rng;

type StorageBlockSize = U1024;

static ENCLAVE_FILE: &'static str = "enclave.signed.so";

extern "C" {
    fn ecall_create_oram(eid: sgx_enclave_id_t, retval: *mut u64, n: u64) -> sgx_status_t;
    fn ecall_destroy_oram(eid: sgx_enclave_id_t) -> sgx_status_t;
    fn ecall_access(
        eid: sgx_enclave_id_t,
        retval: *mut sgx_status_t,
        query: *mut u8,
        query_len: usize,
        resp: *mut u8,
        resp_len: usize,
    ) -> sgx_status_t;
}

fn init_enclave() -> SgxResult<SgxEnclave> {
    let mut launch_token: sgx_launch_token_t = [0; 1024];
    let mut launch_token_updated: i32 = 0;
    // call sgx_create_enclave to initialize an enclave instance
    // Debug Support: set 2nd parameter to 1
    let debug = 1;
    let mut misc_attr = sgx_misc_attribute_t {
        secs_attr: sgx_attributes_t { flags: 0, xfrm: 0 },
        misc_select: 0,
    };
    SgxEnclave::create(
        ENCLAVE_FILE,
        debug,
        &mut launch_token,
        &mut launch_token_updated,
        &mut misc_attr,
    )
}

fn main() {
    let base_dir = std::env::current_dir().expect("not found path");
    logger::initialize_loggers(base_dir.join("running.log"), LogLevel::Info);

    log::info!("Initializing enclave...");
    let enclave = match init_enclave() {
        Ok(r) => {
            println!("[+] Init Enclave Successful {}!", r.geteid());
            r
        }
        Err(x) => {
            println!("[-] Init Enclave Failed {}!", x.as_str());
            return;
        }
    };
    log::info!("Finished.");

    let n = 8<<20;  //8M*1KB
    let mut retval = n;
    let result = unsafe { ecall_create_oram(enclave.geteid(), &mut retval, n) };
    match result {
        sgx_status_t::SGX_SUCCESS => {}
        _ => {
            println!("[-] ECALL Enclave Failed {}!", result.as_str());
            return;
        }
    }

    let mut rng = RngType::from_seed([7u8; 32]);
    exercise_oram(10000, &mut rng, n, enclave.geteid());

    enclave.destroy();
}

/// Exercise an ORAM by writing, reading, and rewriting, a progressively larger
/// set of random locations
pub fn exercise_oram<R>(num_rounds: usize, rng: &mut R, len: u64, eid: sgx_enclave_id_t)
where
    R: RngCore + CryptoRng,
{
    let mut cur_num_rounds = num_rounds;
    assert!(len != 0, "len is zero");
    assert_eq!(len & (len - 1), 0, "len is not a power of two");
    let mut expected = BTreeMap::<u64, A64Bytes<StorageBlockSize>>::default();
    let mut probe_positions = Vec::<u64>::new();
    let mut probe_idx = 0usize;

    let now = Instant::now();
    while cur_num_rounds > 0 {
        if probe_idx >= probe_positions.len() {
            probe_positions.push(rng.next_u64() & (len - 1));
            probe_idx = 0;
        }
        let idx = probe_positions[probe_idx];
        let mut expected_ent = expected.entry(idx).or_default();
        rng.fill_bytes(expected_ent);

        let mut query = Query::<StorageBlockSize> {
            op_type: 1,
            idx,
            new_val: expected_ent.clone(),
        }
        .encrypt_to();
        let query_len = query.len();
        let resp_len = StorageBlockSize::USIZE + NonceSize::USIZE;
        let mut resp = vec![0u8; resp_len];

        let mut retval = sgx_status_t::SGX_SUCCESS;
        let result = unsafe {
            ecall_access(
                eid,
                &mut retval,
                query.as_mut_ptr(),
                query_len,
                resp.as_mut_ptr(),
                resp_len,
            )
        };
        match result {
            sgx_status_t::SGX_SUCCESS => {}
            _ => {
                println!("[-] ECALL Enclave Failed {}!", result.as_str());
                return;
            }
        }
        //println!("result = {:?}", query::decrypt_res(&mut resp));

        probe_idx += 1;
        cur_num_rounds -= 1;
    }

    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    let per_dur = dur / (num_rounds as f64);
    println!("total time = {:?}s, time per query = {:?}s", dur, per_dur);
}
