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
#![feature(drain_filter)]
#![feature(map_first_last)]
#![feature(int_log)]

extern crate sgx_types;
extern crate sgx_urts;
use flume::{Receiver, Sender};
use sgx_types::*;
use sgx_urts::SgxEnclave;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate libc;
use aes::{cipher::NewCipher, Aes128Ctr};
use aligned_cmov::{
    typenum::{Unsigned, U1024, U64},
    A64Bytes,
};
use rand_core::{CryptoRng, RngCore, SeedableRng};
use rand_hc::Hc128Rng;
use tokio::io::{self, AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use std::collections::BTreeMap;
use std::thread;
use std::time::Instant;

mod allocator;
mod logger;
use logger::LogLevel;
mod query;
use query::{
    a64_bytes, extract_client_id, s_decrypt, s_encrypt, Query, ANSWER_SIZE, QUERY_KEY, QUERY_SIZE,
};
mod storage_ocalls;
use storage_ocalls::release_all_oram_storage;

/// Cipher type. Anything implementing StreamCipher and NewCipher at 128
/// bit security should be acceptable
type CipherType = Aes128Ctr;
/// Parameters of the cipher as typedefs (which eases syntax)
type NonceSize = <CipherType as NewCipher>::NonceSize;
type KeySize = <CipherType as NewCipher>::KeySize;
type RngType = Hc128Rng;

type StorageBlockSize = U1024;

static ENCLAVE_FILE: &'static str = "enclave.signed.so";

enum Message {
    Qry((u64, Vec<u8>)),
    CID((u64, Sender<Vec<u8>>)),
}

extern "C" {
    fn ecall_create_oram(
        eid: sgx_enclave_id_t,
        retval: *mut u64,
        n: u64,
        results_ptr: *mut usize,
    ) -> sgx_status_t;
    fn ecall_release_results_space(eid: sgx_enclave_id_t, results_ptr: usize) -> sgx_status_t;
    fn ecall_destroy_oram(eid: sgx_enclave_id_t) -> sgx_status_t;
    fn ecall_access(
        eid: sgx_enclave_id_t,
        retval: *mut sgx_status_t,
        batch_size: usize,
        queries: *mut u8,
        queries_len: usize,
        answers: *mut u8,
        answers_len: usize,
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

fn init_enclave_and_oram(n: u64) -> (SgxEnclave, u64, Vec<Vec<u8>>) {
    log::info!("Initializing enclave...");
    let enclave = match init_enclave() {
        Ok(r) => {
            println!("[+] Init Enclave Successful {}!", r.geteid());
            r
        }
        Err(x) => {
            panic!("[-] Init Enclave Failed {}!", x.as_str());
        }
    };
    log::info!("Enclave intialization finished.");
    let eid = enclave.geteid();

    let now = Instant::now();
    let mut retval = n;
    let mut results_ptr = 0;
    let result_code = unsafe { ecall_create_oram(eid, &mut retval, n, &mut results_ptr) };
    match result_code {
        sgx_status_t::SGX_SUCCESS => {}
        _ => {
            panic!("[-] ECALL Enclave Failed {}!", result_code.as_str());
        }
    }
    let results_in_enclave = unsafe { Box::from_raw(results_ptr as *mut Vec<u8>) };
    assert_eq!(results_in_enclave.len() % ANSWER_SIZE, 0);
    let results = results_in_enclave
        .chunks(ANSWER_SIZE)
        .map(|x| x.to_vec())
        .collect::<Vec<_>>();
    std::mem::forget(results_in_enclave);
    unsafe { ecall_release_results_space(eid, results_ptr) };
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("init n = {:?} 1KB blocks cost {:?} s", n, dur);

    (enclave, eid, results)
}

async fn handle_client(req_tx: Sender<Message>, mut stream: TcpStream) {
    let mut id_buf: [u8; 8] = Default::default();
    stream.read_exact(&mut id_buf).await.unwrap();
    let id = u64::from_ne_bytes(id_buf);
    let (resp_tx, resp_rx) = flume::unbounded();
    req_tx
        .send_async(Message::CID((id, resp_tx)))
        .await
        .unwrap();
    let mut query = vec![0 as u8; QUERY_SIZE];

    let (mut rd, mut wt) = io::split(stream);
    //send result back to client in the background
    let write_task = tokio::spawn(async move {
        loop {
            let resp = resp_rx.recv_async().await.unwrap();
            let r = wt.write_all(&resp).await;
            if r.is_err() {
                println!("close write half");
                break;
            }
        }
    });

    loop {
        let r = rd.read_exact(&mut query).await;

        if r.is_err() {
            println!("close read half");
            break;
        }

        req_tx
            .send_async(Message::Qry((id, query.clone())))
            .await
            .unwrap();
    }

    write_task.abort();
    //assert!(write_task.await.unwrap_err().is_cancelled());
}

async fn listen_to_clients(req_tx: Sender<Message>, shutdown_rx: Receiver<()>) {
    let listener = TcpListener::bind("127.0.0.1:3333").await.unwrap();
    // accept connections and process them, spawning a new thread for each one
    println!("Server listening on port 3333");
    let shutdown_rx = shutdown_rx.into_recv_async();

    tokio::select! {
        _ = async {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                println!("New connection: {}", stream.peer_addr().unwrap());
                let req_tx_c = req_tx.clone();
                tokio::spawn(async move {
                    // connection succeeded
                    handle_client(req_tx_c, stream).await;
                    println!("client exits");
                });
            }
        } => {}
        _ = shutdown_rx => {
            println!("terminating accept loop");
        }
    }

    // close the socket server
    // drop(listener);
}

fn serve(mut enclave: SgxEnclave, mut eid: u64, n: u64, mut remaining_results: Vec<Vec<u8>>) {
    println!("remaining_results = {:?}", remaining_results);

    let (shutdown_tx, shutdown_rx) = flume::unbounded();
    let (req_tx, req_rx) = flume::unbounded();
    let hd = thread::spawn(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            listen_to_clients(req_tx, shutdown_rx).await;
        })
    });
    let mut counter = 0;
    let limit_enclave_crash = 1_000_000;
    let limit_machine_crash = 100_000_000;
    let batch_size = 1;
    let mut id_to_sender: BTreeMap<u64, Sender<Vec<u8>>> = BTreeMap::new();

    let mut queries = Vec::with_capacity(QUERY_SIZE * batch_size);
    let answers_len = ANSWER_SIZE * batch_size;
    let mut answers = vec![0u8; answers_len];
    let mut ids = Vec::with_capacity(batch_size);
    loop {
        let req = req_rx.recv().unwrap();
        //send the remaining answers back
        let to_be_replied = remaining_results
            .drain_filter(|res| id_to_sender.contains_key(&extract_client_id(res)))
            .collect::<Vec<_>>();
        for reply in to_be_replied {
            let sender = id_to_sender.get(&extract_client_id(&reply)).unwrap();
            sender.send(reply).unwrap();
        }

        //receive the query
        match req {
            Message::Qry((id, mut query)) => {
                ids.push(id);
                queries.append(&mut query);
                counter += 1;
                if counter % batch_size == 0 {
                    let queries_len = queries.len();
                    assert_eq!(queries_len, QUERY_SIZE * batch_size);
                    let mut retval = sgx_status_t::SGX_SUCCESS;
                    let result = unsafe {
                        ecall_access(
                            eid,
                            &mut retval,
                            batch_size,
                            queries.as_mut_ptr(),
                            queries_len,
                            answers.as_mut_ptr(),
                            answers_len,
                        )
                    };
                    match result {
                        sgx_status_t::SGX_SUCCESS => {}
                        _ => {
                            panic!("[-] ECALL Enclave Failed {}!", result.as_str());
                        }
                    }
                    for (idx, resp) in answers.chunks(ANSWER_SIZE).into_iter().enumerate() {
                        let id = ids[idx];
                        id_to_sender.get(&id).unwrap().send(resp.to_vec()).unwrap();
                    }
                    queries.clear();
                    ids.clear();
                }
            }
            Message::CID((id, resp_tx)) => {
                println!("insert client id {:?}", id);
                id_to_sender.insert(id, resp_tx);
            }
        };

        if counter % limit_machine_crash == 0 && counter != 0 {
            break;
        } else if counter % limit_enclave_crash == 0 && counter != 0 {
            enclave.destroy();
            let mut t = init_enclave_and_oram(n);
            enclave = t.0;
            eid = t.1;
            remaining_results.append(&mut t.2);
        }
    }

    shutdown_tx.send(()).unwrap();
    hd.join().unwrap();
    enclave.destroy();
    release_all_oram_storage();
}

fn main() {
    let base_dir = std::env::current_dir().expect("not found path");
    logger::initialize_loggers(base_dir.join("running.log"), LogLevel::Info);

    let n = 16 << 10; //16K*1KB
    let (enclave, eid, remaining_results) = init_enclave_and_oram(n);
    let now = Instant::now();
    //serve(enclave, eid, n, remaining_results);
    //exercise_oram(10000, n, eid);
    exercise_oram_consecutive(n as usize, n, eid);
    //sanity_check(eid);
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("insert n = {:?} 1KB blocks cost {:?} s", n, dur);

    enclave.destroy();
    release_all_oram_storage();

    //SGX_ERROR_INVALID_ENCLAVE_ID
}

/// Exercise an ORAM by writing, reading, and rewriting, a progressively larger
/// set of random locations
pub fn exercise_oram(num_rounds: usize, len: u64, eid: sgx_enclave_id_t) {
    let mut rng = RngType::from_seed([7u8; 32]);
    let mut cur_num_rounds = num_rounds;
    assert!(len != 0, "len is zero");
    assert_eq!(len & (len - 1), 0, "len is not a power of two");
    let mut expected = BTreeMap::<u64, A64Bytes<StorageBlockSize>>::default();
    let mut probe_positions = Vec::<u64>::new();
    let mut probe_idx = 0usize;

    let mut acc_dur = 0f64;
    while cur_num_rounds > 0 {
        if probe_idx >= probe_positions.len() {
            probe_positions.push(rng.next_u64() & (len - 1));
            probe_idx = 0;
        }
        let idx = probe_positions[probe_idx];
        let expected_ent = expected.entry(idx).or_default();
        rng.fill_bytes(expected_ent);

        let _res = simple_access_wrapper(idx, expected_ent.clone(), eid, &mut rng, &mut acc_dur);
        probe_idx += 1;
        cur_num_rounds -= 1;
    }

    let per_dur = acc_dur / (num_rounds as f64);
    println!(
        "total time = {:?}s, time per query = {:?}s",
        acc_dur, per_dur
    );
}

/// Exercise an ORAM by writing, reading, and rewriting, all locations
/// consecutively
pub fn exercise_oram_consecutive(num_rounds: usize, len: u64, eid: sgx_enclave_id_t) {
    let mut cur_num_rounds = num_rounds;
    let mut rng = RngType::from_seed([7u8; 32]);
    assert!(len != 0, "len is zero");
    assert_eq!(len & (len - 1), 0, "len is not a power of two");
    //let mut expected = BTreeMap::<u64, A64Bytes<StorageBlockSize>>::default();

    let mut acc_dur = 0f64;
    while cur_num_rounds > 0 {
        let query = cur_num_rounds as u64 & (len - 1);
        //let expected_ent = expected.entry(query).or_default();
        let mut val: A64Bytes<StorageBlockSize> = Default::default();
        rng.fill_bytes(&mut val);
        let _res = simple_access_wrapper(query, val, eid, &mut rng, &mut acc_dur);
        cur_num_rounds -= 1;
    }
    let per_dur = acc_dur / (num_rounds as f64);
    println!(
        "total time = {:?}s, time per query = {:?}s",
        acc_dur, per_dur
    );
}

pub fn sanity_check(eid: sgx_enclave_id_t) {
    let mut rng = RngType::from_seed([7u8; 32]);
    let mut acc_dur = 0f64;
    assert_eq!(
        a64_bytes::<StorageBlockSize>(0).as_slice(),
        &simple_access_wrapper(0, a64_bytes(1), eid, &mut rng, &mut acc_dur)[..]
    );
    assert_eq!(
        a64_bytes::<StorageBlockSize>(1).as_slice(),
        &simple_access_wrapper(0, a64_bytes(2), eid, &mut rng, &mut acc_dur)[..]
    );
    assert_eq!(
        a64_bytes::<StorageBlockSize>(2).as_slice(),
        &simple_access_wrapper(0, a64_bytes(3), eid, &mut rng, &mut acc_dur)[..]
    );
    assert_eq!(
        a64_bytes::<StorageBlockSize>(0).as_slice(),
        &simple_access_wrapper(2, a64_bytes(4), eid, &mut rng, &mut acc_dur)[..]
    );
    assert_eq!(
        a64_bytes::<StorageBlockSize>(4).as_slice(),
        &simple_access_wrapper(2, a64_bytes(5), eid, &mut rng, &mut acc_dur)[..]
    );
    assert_eq!(
        a64_bytes::<StorageBlockSize>(3).as_slice(),
        &simple_access_wrapper(0, a64_bytes(6), eid, &mut rng, &mut acc_dur)[..]
    );
    assert_eq!(
        a64_bytes::<StorageBlockSize>(6).as_slice(),
        &simple_access_wrapper(0, a64_bytes(7), eid, &mut rng, &mut acc_dur)[..]
    );
    assert_eq!(
        a64_bytes::<StorageBlockSize>(0).as_slice(),
        &simple_access_wrapper(9, a64_bytes(8), eid, &mut rng, &mut acc_dur)[..]
    );
    assert_eq!(
        a64_bytes::<StorageBlockSize>(5).as_slice(),
        &simple_access_wrapper(2, a64_bytes(10), eid, &mut rng, &mut acc_dur)[..]
    );
    assert_eq!(
        a64_bytes::<StorageBlockSize>(7).as_slice(),
        &simple_access_wrapper(0, a64_bytes(11), eid, &mut rng, &mut acc_dur)[..]
    );
    assert_eq!(
        a64_bytes::<StorageBlockSize>(8).as_slice(),
        &simple_access_wrapper(9, a64_bytes(12), eid, &mut rng, &mut acc_dur)[..]
    );
    assert_eq!(
        a64_bytes::<StorageBlockSize>(12).as_slice(),
        &simple_access_wrapper(9, a64_bytes(13), eid, &mut rng, &mut acc_dur)[..]
    );
}

pub fn simple_access_wrapper(
    idx: u64,
    data: A64Bytes<StorageBlockSize>,
    eid: sgx_enclave_id_t,
    rng: &mut RngType,
    acc_dur: &mut f64,
) -> Vec<u8> {
    let query = Query::<StorageBlockSize> {
        op_type: 1,
        idx,
        new_val: data.clone(),
    };
    let mut bytes = vec![0; QUERY_SIZE];
    query.to_slice(&mut bytes);
    let skip_enc = 16;
    s_encrypt(&QUERY_KEY, &mut bytes, skip_enc, rng);
    let mut answer = vec![0u8; ANSWER_SIZE];
    let mut retval = sgx_status_t::SGX_SUCCESS;
    let now = Instant::now();
    let result = unsafe {
        ecall_access(
            eid,
            &mut retval,
            1,
            bytes.as_mut_ptr(),
            QUERY_SIZE,
            answer.as_mut_ptr(),
            ANSWER_SIZE,
        )
    };
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    *acc_dur += dur;
    match result {
        sgx_status_t::SGX_SUCCESS => {}
        _ => {
            println!("[-] ECALL Enclave Failed {}!", result.as_str());
            return Vec::new();
        }
    }
    s_decrypt(&QUERY_KEY, &mut answer, skip_enc);
    (&answer[(NonceSize::USIZE + 16 + skip_enc)..]).to_vec()
}
