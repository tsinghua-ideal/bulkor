// Copyright (c) 2018-2021 The MobileCoin Foundation

#![deny(missing_docs)]

//! ORAM storage is arranged as a complete balanced binary tree, with each node
//! holding a fixed-size block of size roughly a linux page. Each node also has
//! an associated fixed-size metadata block, about 100 times smaller.
//!
//! It is possible to store all of this on the heap inside of SGX, up to the
//! limits of the enclave heap size. There are also performance consequences of
//! exceeding the EPC size (enclave page cache).
//!
//! In ORAM implementations such as ZeroTrace, OCALL's are used to allow the
//! enclave to store this data outside of SGX. This data must be encrypted when
//! it leaves, and decrypted and authenticated when it returns. From trusted's
//! point of view, it doesn't matter much how untrusted chooses to actually
//! store the blocks, as long as it returns the correct ones -- if
//! authentication fails, the enclave is expected to panic.
//!
//! Tree-top caching means that the top of the tree is on the heap in SGX and
//! only the bottom part is across the OCALL boundary. This can result in
//! significant perf improvements especially when using a recursive ORAM
//! strategy.
//!
//! In this impementation, the tree-top caching size is configurable via a
//! global variable.
//!
//! For an overview and analysis of the authentication scheme implemented here,
//! the reader is directed to README.md for this crate.

use std::cmp::max;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::SgxMutex as Mutex;
use std::vec::Vec;

use aes::cipher::{NewCipher, StreamCipher};
use aes_gcm::aead::{Aead, NewAead};
use aligned_cmov::{
    subtle::{Choice, ConstantTimeEq},
    typenum::{PartialDiv, PowerOfTwo, Prod, Quot, Sum, Unsigned, U8},
    A64Bytes, A8Bytes, ArrayLength, GenericArray,
};
use balanced_tree_index::TreeIndex;
use displaydoc::Display;
use rand_core::{CryptoRng, RngCore};

use crate::oram_storage::shuffle_manager::manage;
use crate::oram_traits::{HeapORAMStorage, ORAMStorage, ORAMStorageCreator};
use crate::{
    AuthCipherType, AuthNonceSize, CipherType, KeySize, NonceSize, IS_LATEST, LIFETIME_ID,
    SNAPSHOT_ID,
};

mod extra_meta;
pub use extra_meta::{compute_block_hash, compute_slices_hash, ExtraMeta, ExtraMetaSize, Hash};
pub mod shuffle_manager;
use shuffle_manager::BIN_SIZE_IN_BLOCK;

lazy_static! {
    /// The tree-top caching threshold, specified as log2 of a number of bytes.
    ///
    /// This is the approximate number of bytes that can be stored on the heap in the enclave
    /// for a single ORAM storage object.
    ///
    /// This is expected to be tuned as a function of
    /// (1) number of (recursive) ORAM's needed
    /// (2) enclave heap size, set at build time
    ///
    /// Changing this number influences any ORAM storage objects created after the change,
    /// but not before. So, it should normally be changed during enclave init, if at all.
    /// NOTE: it influences the setting of StackMaxSize

    //pub static ref TREETOP_CACHING_THRESHOLD_LOG2: AtomicU32 = AtomicU32::new(25u32); // 32 MB
    pub static ref TREETOP_CACHING_THRESHOLD_LOG2: AtomicU32 = AtomicU32::new(24u32); // 16 MB
    pub static ref TREEMID_CACHING_THRESHOLD_LOG2: AtomicU32 = AtomicU32::new(33u32-1u32.log2()); // 8 GB
    pub static ref IN_ENCLAVE_RATIO: usize = (TREEMID_CACHING_THRESHOLD_LOG2.load(Ordering::SeqCst)/TREETOP_CACHING_THRESHOLD_LOG2.load(Ordering::SeqCst)) as usize;

    // An extra mutex which we lock across our OCALLs, this is done to detect if the untrusted
    // attacker does strange things.
    //
    // The purpose of this mutex is to try to guard against re-entrancy into the enclave.
    // Re-entrancy could occur if the untrusted side handles the OCALL by turning around and
    // making more ECALLs, never returning from first OCALL and setting up unexpected state in enclave.
    // Another worrisome scenario is, they could never return from first OCALL and let other threads
    // continue to proceed and evenutally make more OCALLs.
    // Then the first OCALL could be responded to adaptively etc. in a way that the enclave designer
    // might not expect. Using a mutex to serialize the OCALLs would prevent that.
    //
    // Yogseh Swami "Intel SGX Remote Attestation is not sufficient" discusses such issues in detail:
    // https://www.blackhat.com/docs/us-17/thursday/us-17-Swami-SGX-Remote-Attestation-Is-Not-Sufficient-wp.pdf
    //
    // A simliar measure is in place for the trusted side fog-recovery-db-iface.
    //
    // It is not clear that this is actually needed to prevent problems,
    // we just haven't done a detailed anaysis.
    //
    // If you do such an analysis and conclude it's safe, you could remove this and leave a code comment.
    // However, the cost of this is likely very low since the mutex is uncontended -- our real enclaves
    // only have one OMAP object, and its API is &mut, so the caller must wrap it in a mutex anyways,
    // and this is unlikely to change.
    static ref OCALL_REENTRANCY_MUTEX: Mutex<()> = Mutex::new(());

    // The maximum level of created ORAM, up to now.
    pub static ref MAX_LEVEL: AtomicU32 = AtomicU32::new(0u32);

    // The key used to encrypt whatever to be stored outside, and it should be set by attestation
    pub static ref ORAM_KEY: GenericArray<u8, KeySize> = GenericArray::<u8, KeySize>::default();
}

// Make an aes nonce per the docu
fn make_aes_nonce(block_idx: u64, block_ctr: u64) -> GenericArray<u8, NonceSize> {
    let mut result = GenericArray::<u8, NonceSize>::default();
    result[0..8].copy_from_slice(&block_idx.to_ne_bytes());
    result[8..16].copy_from_slice(&block_ctr.to_ne_bytes());
    result
}

//block cipher
pub fn b_encrypt<Rng: RngCore + CryptoRng>(pt: &[u8], rng: &mut Rng) -> Vec<u8> {
    let cipher = AuthCipherType::new(&ORAM_KEY);
    let mut nonce = GenericArray::<u8, AuthNonceSize>::default();
    rng.fill_bytes(nonce.as_mut_slice());
    let mut res = nonce.to_vec();
    res.append(&mut cipher.encrypt(&nonce, pt).expect("encryption failure"));
    res
}

//block decipher
pub fn b_decrypt(ct: &[u8]) -> Vec<u8> {
    let cipher = AuthCipherType::new(&ORAM_KEY);
    let nonce = GenericArray::from_slice(&ct[0..AuthNonceSize::USIZE]);
    cipher
        .decrypt(nonce, &ct[AuthNonceSize::USIZE..])
        .expect("decryption failure")
}

//stream cipher, reserve place for nonce and tag
pub fn s_encrypt<Rng: RngCore + CryptoRng>(
    key: &GenericArray<u8, KeySize>,
    pt: &mut [u8],
    skip_enc: usize,
    rng: &mut Rng,
) {
    let mut nonce = GenericArray::<u8, NonceSize>::default();
    let ns = NonceSize::USIZE;
    rng.fill_bytes(nonce.as_mut_slice());
    let mut cipher = CipherType::new(key, &nonce);
    cipher.apply_keystream(&mut pt[(ns + 16 + skip_enc)..]);
    let h = compute_slices_hash(key, &pt[(ns + 16)..]);
    (&mut pt[..ns]).copy_from_slice(&nonce);
    (&mut pt[ns..(ns + 16)]).copy_from_slice(&h);
}

//stream cipher for nonce has been prepared, reserve place for nonce and tag,
pub fn s_encrypt_with_nonce(key: &GenericArray<u8, KeySize>, pt: &mut [u8], skip_enc: usize) {
    let ns = NonceSize::USIZE;
    let nonce = GenericArray::<u8, NonceSize>::from_slice(&pt[..ns]);
    let mut cipher = CipherType::new(key, nonce);
    cipher.apply_keystream(&mut pt[(ns + 16 + skip_enc)..]);
    let h = compute_slices_hash(key, &pt[(ns + 16)..]);
    (&mut pt[ns..(ns + 16)]).copy_from_slice(&h);
}

//stream decipher, ct including nonce and tag
pub fn s_decrypt(key: &GenericArray<u8, KeySize>, ct: &mut [u8], skip_enc: usize) {
    let ns = NonceSize::USIZE;
    let h: Hash = (&ct[ns..(ns + 16)]).try_into().unwrap();
    assert!(h == compute_slices_hash(key, &ct[(ns + 16)..]));
    let nonce = GenericArray::from_slice(&ct[..ns]);
    let mut cipher = CipherType::new(key, nonce);
    cipher.apply_keystream(&mut ct[(ns + 16 + skip_enc)..]);
}

/// An ORAMStorage type which stores data with untrusted storage, over an OCALL.
/// This must encrypt the data which is stored, and authenticate the data when
/// it returns.
pub struct OcallORAMStorage<DataSize, MetaSize, Z>
where
    DataSize: ArrayLength<u8> + PowerOfTwo + PartialDiv<U8> + Div<Z>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8> + Add<ExtraMetaSize> + Add<Prod<Z, U8>> + Div<Z>,
    Z: Unsigned + Mul<U8>,
    Prod<Z, U8>: Unsigned,
    Sum<MetaSize, ExtraMetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    Quot<DataSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
    Quot<MetaSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Add<U8> + Unsigned,
    Sum<MetaSize, Prod<Z, U8>>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
    Sum<Quot<MetaSize, Z>, U8>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
{
    /// The current level of recursive ORAM
    level: u32,
    // The id returned from untrusted for the untrusted-side storage if any, or 0 if none.
    allocation_id: u64,
    // The size of the binary tree the caller asked us to provide storage for, must be a power of
    // two
    count: u64,
    // The maximum count for the treetop storage in trusted memory,
    // based on what we loaded from TREETOP_CACHING_THRESHOLD_LOG2 at construction time
    // This must never change after construction.
    treetop_max_count: u64,
    // The storage on the heap for the top of the tree
    treetop: HeapORAMStorage<DataSize, MetaSize, Z>,
    // The trusted merkle roots of trees rooted just below the treetop
    trusted_merkle_roots: Vec<Hash>,
    // A temporary scratch buffer for use when getting metadata from untrusted and validating it
    // This buffer contains metadata + extended_metadata for each checked out block (see README.md)
    meta_scratch_buffer: Vec<A8Bytes<Sum<MetaSize, ExtraMetaSize>>>,
    // An AES key
    aes_key: GenericArray<u8, KeySize>,
    // The key we use when hashing ciphertexts to make merkle tree
    // Keeping this secret makes the hash functionally a mac
    hash_key: GenericArray<u8, KeySize>,
    _z: PhantomData<fn() -> Z>,
}

impl<DataSize, MetaSize, Z> OcallORAMStorage<DataSize, MetaSize, Z>
where
    DataSize: ArrayLength<u8> + PowerOfTwo + PartialDiv<U8> + Div<Z>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8> + Add<ExtraMetaSize> + Add<Prod<Z, U8>> + Div<Z>,
    Z: Unsigned + Mul<U8>,
    Prod<Z, U8>: Unsigned,
    Sum<MetaSize, ExtraMetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    Quot<DataSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
    Quot<MetaSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Add<U8> + Unsigned,
    Sum<MetaSize, Prod<Z, U8>>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
    Sum<Quot<MetaSize, Z>, U8>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
{
    /// Create a new oram storage object for count items, with particular RNG
    pub fn new<Rng: RngCore + CryptoRng>(level: u32, count: u64, rng: &mut Rng) -> Self {
        assert!(count != 0);
        assert!(count & (count - 1) == 0, "count must be a power of two");

        let treetop_max_count: u64 = max(
            2u64,
            (1u64 << TREETOP_CACHING_THRESHOLD_LOG2.load(Ordering::SeqCst)) / DataSize::U64,
        );

        let snapshot_id = SNAPSHOT_ID.load(Ordering::SeqCst);
        let is_latest = IS_LATEST.load(Ordering::SeqCst);
        let lifetime_id = LIFETIME_ID.load(Ordering::SeqCst);

        let mut trusted_merkle_roots = if count <= treetop_max_count {
            Default::default()
        } else {
            let mut v = vec![Default::default(); (treetop_max_count * 2) as usize];
            if snapshot_id > 0 && (is_latest || level == 0) {
                let ns = NonceSize::USIZE;
                let roots_len = ns + 16 + 8 + 4 + 8 + 8 + v.len() * 16; //add count to be checked
                                                                        //The correct count also ensure the correct height in pos map
                let mut roots_buf = vec![0 as u8; roots_len];
                unsafe {
                    recover_merkle_roots(roots_buf.as_mut_ptr(), roots_len, level, snapshot_id);
                }
                s_decrypt(&ORAM_KEY, &mut roots_buf, 28);
                //check the integrity
                let loaded_snapshot_id =
                    u64::from_ne_bytes((&roots_buf[(ns + 16)..(ns + 24)]).try_into().unwrap());
                assert_eq!(loaded_snapshot_id, snapshot_id);
                let loaded_level =
                    u32::from_ne_bytes((&roots_buf[(ns + 24)..(ns + 28)]).try_into().unwrap());
                assert_eq!(loaded_level, level);
                let loaded_count =
                    u64::from_ne_bytes((&roots_buf[(ns + 28)..(ns + 36)]).try_into().unwrap());
                assert_eq!(loaded_count, count);
                let loaded_lifetime_id =
                    u64::from_ne_bytes((&roots_buf[(ns + 36)..(ns + 44)]).try_into().unwrap());
                assert_eq!(loaded_lifetime_id, lifetime_id);

                let iter_data = (&roots_buf[(ns + 44)..]).chunks_exact(16);
                assert_eq!(iter_data.remainder(), []);
                v = iter_data
                    .into_iter()
                    .map(|d| d.try_into().unwrap())
                    .collect::<Vec<_>>();
            }
            v
        };

        let mut treetop = {
            if count <= treetop_max_count {
                HeapORAMStorage::new(level, count)
            } else {
                HeapORAMStorage::new(level, treetop_max_count)
            }
        };
        //recovery is included in allocation
        let mut allocation_id = 0u64;
        unsafe {
            allocate_oram_storage(
                level,
                snapshot_id,
                is_latest as u8,
                count,
                DataSize::U64,
                MetaSize::U64 + ExtraMetaSize::U64,
                &mut allocation_id,
            );
        }
        if allocation_id == 0 {
            panic!("Untrusted could not allocate storage! count = {}, data_size = {}, meta_size + extra_meta_size = {}",
                    count,
                    DataSize::U64,
                    MetaSize::U64 + ExtraMetaSize::U64)
        }

        //Recover the keys
        let aes_key: GenericArray<u8, KeySize> = ORAM_KEY.clone();
        let hash_key: GenericArray<u8, KeySize> = ORAM_KEY.clone();
        // let mut aes_key = GenericArray::<u8, KeySize>::default();
        // rng.fill_bytes(aes_key.as_mut_slice());
        // let mut hash_key = GenericArray::<u8, KeySize>::default();
        // rng.fill_bytes(hash_key.as_mut_slice());

        //shuffle the oram tree if necessary
        if snapshot_id > 0 && !is_latest {
            let mut shuffle_id = 0;
            unsafe {
                allocate_shuffle_manager(allocation_id, Z::U64, BIN_SIZE_IN_BLOCK, &mut shuffle_id);
            }
            //change the shuffle nonce
            manage(
                level,
                shuffle_id,
                allocation_id,
                count,
                treetop_max_count as usize,
                &mut treetop,
                &mut trusted_merkle_roots,
                &aes_key,
                &hash_key,
                rng,
            );
        }

        println!("Finish ORAM allocation");

        Self {
            level,
            allocation_id,
            count,
            treetop_max_count,
            treetop,
            trusted_merkle_roots,
            meta_scratch_buffer: Default::default(),
            aes_key,
            hash_key,
            _z: Default::default(),
        }
    }

    /// Get the treetop_max_count value for this storage object
    pub fn get_treetop_max_count(&self) -> u64 {
        self.treetop_max_count
    }
}

impl<DataSize, MetaSize, Z> ORAMStorage<DataSize, MetaSize, Z>
    for OcallORAMStorage<DataSize, MetaSize, Z>
where
    DataSize: ArrayLength<u8> + PowerOfTwo + PartialDiv<U8> + Div<Z>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8> + Add<ExtraMetaSize> + Add<Prod<Z, U8>> + Div<Z>,
    Z: Unsigned + Mul<U8>,
    Prod<Z, U8>: Unsigned,
    Sum<MetaSize, ExtraMetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    Quot<DataSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
    Quot<MetaSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Add<U8> + Unsigned,
    Sum<MetaSize, Prod<Z, U8>>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
    Sum<Quot<MetaSize, Z>, U8>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
{
    fn len(&self) -> u64 {
        self.count
    }

    fn checkout(
        &mut self,
        index: u64,
        dest: &mut [A64Bytes<DataSize>],
        dest_meta: &mut [A8Bytes<MetaSize>],
    ) {
        assert_eq!(dest.len(), dest_meta.len());
        assert!(index > 0, "0 is not a valid TreeIndex");
        assert!(index < self.count, "index out of bounds");

        let mut indices: Vec<u64> = index.parents().collect();

        assert_eq!(indices.len(), dest.len());

        // First step: Do the part that's in the treetop
        let first_treetop_index = indices
            .iter()
            .position(|idx| idx < &self.treetop_max_count)
            .expect("should be unreachable, at least one thing should be in the treetop");

        self.treetop.checkout(
            indices[first_treetop_index],
            &mut dest[first_treetop_index..],
            &mut dest_meta[first_treetop_index..],
        );

        // Now do the part that's not in the treetop
        // If first_treetop_index == 0 then everything is in the treetop
        if first_treetop_index > 0 {
            // For persist ORAM, it should not allow to subtract treetop_max_count
            // from indices before sending to untrusted
            // for idx in &mut indices[..first_treetop_index] {
            //     *idx -= self.treetop_max_count;
            // }
            self.meta_scratch_buffer
                .resize_with(first_treetop_index, Default::default);

            {
                let _lk = OCALL_REENTRANCY_MUTEX
                    .lock()
                    .expect("could not lock our mutex");
                helpers::checkout_ocall(
                    self.allocation_id,
                    &indices[..first_treetop_index],
                    &mut dest[..first_treetop_index],
                    &mut self.meta_scratch_buffer,
                );
            }
            // For persist ORAM, it does not need to add treetop_max_count back to
            // indices so that our calculations will be correct
            // for idx in &mut indices[..first_treetop_index] {
            //     *idx += self.treetop_max_count;
            // }

            // We have to decrypt, checking the macs in the meta scratch buffer, and
            // ultimately set dest_meta[idx]
            let mut last_hash: Option<(u64, Hash)> = None;
            for idx in 0..first_treetop_index {
                // If untrusted gave us all 0's for the metadata, then the result is all zeroes
                // Otherwise we have to decrypt
                if self.meta_scratch_buffer[idx] == Default::default() {
                    dest[idx] = Default::default();
                    dest_meta[idx] = Default::default();
                    last_hash = Some((indices[idx], Default::default()));
                } else {
                    // Compute the hash for this block
                    let this_block_hash = compute_block_hash(
                        &self.hash_key,
                        &dest[idx],
                        indices[idx],
                        &self.meta_scratch_buffer[idx],
                    );

                    // Split extra_meta out of scratch buffer
                    let (meta, extra_meta) =
                        self.meta_scratch_buffer[idx].split_at_mut(MetaSize::USIZE);
                    let extra_meta = ExtraMeta::from(&*extra_meta);

                    // If this block has a child, check if its hash that we computed before matches
                    // metadata
                    if let Some((last_idx, last_hash)) = last_hash {
                        if last_idx & 1 == 0 {
                            if last_hash != extra_meta.left_child_hash {
                                panic!("authentication failed when checking out index[{}] = {}: left child hash {:?} != expected {:?}", idx, indices[idx], last_hash, extra_meta.left_child_hash);
                            }
                        } else if last_hash != extra_meta.right_child_hash {
                            panic!("authentication failed when checking out index[{}] = {}:, right child hash {:?} != expected {:?}", idx, indices[idx], last_hash, extra_meta.right_child_hash);
                        }
                    }

                    // Check with trusted merkle root if this is the last treetop index
                    if idx == first_treetop_index - 1
                        && !bool::from(
                            self.trusted_merkle_roots[indices[idx] as usize]
                                .ct_eq(&this_block_hash),
                        )
                    {
                        panic!(
                            "authentication failed, trusted merkle root {}",
                            indices[idx]
                        );
                    }

                    // Store this hash for next round
                    last_hash = Some((indices[idx], this_block_hash));

                    // Decrypt
                    let aes_nonce = make_aes_nonce(indices[idx], extra_meta.block_ctr);
                    let mut cipher = CipherType::new(&self.aes_key, &aes_nonce);
                    cipher.apply_keystream(&mut dest[idx]);
                    cipher.apply_keystream(meta);
                    dest_meta[idx].copy_from_slice(meta);
                }
            }
        }
    }

    fn checkin(
        &mut self,
        index: u64,
        src: &mut [A64Bytes<DataSize>],
        src_meta: &mut [A8Bytes<MetaSize>],
    ) {
        assert_eq!(src.len(), src_meta.len());
        assert!(index > 0);
        assert!(index < self.count, "index out of bounds");

        let mut indices: Vec<u64> = index.parents().collect();

        assert_eq!(indices.len(), src.len());

        let first_treetop_index = indices
            .iter()
            .position(|idx| idx < &self.treetop_max_count)
            .expect("should be unreachable, at least one thing should be in the treetop");

        self.treetop.checkin(
            indices[first_treetop_index],
            &mut src[first_treetop_index..],
            &mut src_meta[first_treetop_index..],
        );

        // If first_treetop_index == 0 then everything is in the treetop
        if first_treetop_index > 0 {
            self.meta_scratch_buffer
                .resize_with(first_treetop_index, Default::default);

            // We have to update the extra metadata, then encrypt the data and metadata,
            // then compute and store hash for next round.
            let mut last_hash: Option<(u64, Hash)> = None;
            for idx in 0..first_treetop_index {
                // Update the metadata field and extract the new block_ctr value so that we can
                // encrypt
                let block_ctr = {
                    // Split extra_meta out of scratch buffer
                    let (meta, extra_meta) =
                        self.meta_scratch_buffer[idx].split_at_mut(MetaSize::USIZE);

                    // Update the meta
                    meta.copy_from_slice(&src_meta[idx]);

                    // Update the extra_meta
                    let mut extra_meta_val = ExtraMeta::from(&*extra_meta);

                    // If this block has a child, update extra_meta check if its hash that we
                    // computed before matches metadata
                    if let Some((last_idx, last_hash)) = last_hash {
                        if last_idx & 1 == 0 {
                            extra_meta_val.left_child_hash = last_hash;
                        } else {
                            extra_meta_val.right_child_hash = last_hash;
                        }
                    }

                    // Update block_ctr value by incrementing it
                    extra_meta_val.block_ctr += 1;

                    // Serialize the ExtraMeta object to bytes and store them at extra_meta
                    let extra_meta_bytes = GenericArray::<u8, ExtraMetaSize>::from(&extra_meta_val);
                    extra_meta.copy_from_slice(extra_meta_bytes.as_slice());

                    // Return the block_ctr value to use for this encryption
                    extra_meta_val.block_ctr
                };

                // Encrypt the data that is supposed to be encrypted
                {
                    // Split meta out of scratch buffer
                    let (meta, _) = self.meta_scratch_buffer[idx].split_at_mut(MetaSize::USIZE);

                    // Encrypt
                    let aes_nonce = make_aes_nonce(indices[idx], block_ctr);
                    let mut cipher = CipherType::new(&self.aes_key, &aes_nonce);
                    cipher.apply_keystream(&mut src[idx]);
                    cipher.apply_keystream(meta);
                }

                // Compute the hash for this block and store it, to go with parent next round
                let this_block_hash = compute_block_hash(
                    &self.hash_key,
                    &src[idx],
                    indices[idx],
                    &self.meta_scratch_buffer[idx],
                );
                last_hash = Some((indices[idx], this_block_hash));
            }

            // The last one from the treetop goes in self.trusted_merkle_roots
            let (last_idx, last_hash) = last_hash.expect("should not be empty at this point");
            self.trusted_merkle_roots[last_idx as usize] = last_hash;

            // All extra-metas are done, now send it to untrusted for storage
            // For persist ORAM, it should not allow to subtract treetop_max_count
            // from indices before sending to untrusted
            // for idx in &mut indices[..first_treetop_index] {
            //     *idx -= self.treetop_max_count;
            // }
            let _lk = OCALL_REENTRANCY_MUTEX
                .lock()
                .expect("could not lock our mutex");
            helpers::checkin_ocall(
                self.allocation_id,
                &indices[..first_treetop_index],
                &src[..first_treetop_index],
                &self.meta_scratch_buffer,
            );
        }
    }

    fn persist<Rng: RngCore + CryptoRng>(
        &mut self,
        lifetime_id: u64,
        new_snapshot_id: u64,
        volatile: bool,
        rng: &mut Rng,
    ) {
        unsafe {
            persist_oram_storage(
                self.level,
                new_snapshot_id,
                volatile as u8,
                self.allocation_id,
            )
        };
        self.treetop
            .persist(lifetime_id, new_snapshot_id, volatile, rng);
        //encrypt the merkle roots and send it out
        //TODO: This step can be in parallel with the following ones
        let mut roots = vec![0; NonceSize::USIZE + 16];
        roots.extend_from_slice(&new_snapshot_id.to_ne_bytes());
        roots.extend_from_slice(&self.level.to_ne_bytes());
        roots.extend_from_slice(&self.count.to_ne_bytes());
        roots.extend_from_slice(&lifetime_id.to_ne_bytes());
        for d in &self.trusted_merkle_roots {
            roots.extend_from_slice(d);
        }

        s_encrypt(&ORAM_KEY, &mut roots, 28, rng);

        // TODO: the ocall should not stall the steps after it
        unsafe {
            persist_merkle_roots(
                roots.as_ptr(),
                roots.len(),
                self.level,
                new_snapshot_id,
                volatile as u8,
            )
        }
    }
}

/// An ORAMStorageCreator for the Ocall-based storage type
pub struct OcallORAMStorageCreator;

impl<DataSize, MetaSize, Z> ORAMStorageCreator<DataSize, MetaSize, Z> for OcallORAMStorageCreator
where
    DataSize: ArrayLength<u8> + PowerOfTwo + PartialDiv<U8> + Div<Z> + 'static,
    MetaSize:
        ArrayLength<u8> + PartialDiv<U8> + Add<ExtraMetaSize> + Add<Prod<Z, U8>> + Div<Z> + 'static,
    Sum<MetaSize, ExtraMetaSize>: ArrayLength<u8> + PartialDiv<U8> + 'static,
    Z: Unsigned + Mul<U8> + 'static,
    Prod<Z, U8>: Unsigned + 'static,
    Quot<DataSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Unsigned + 'static,
    Quot<MetaSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Add<U8> + Unsigned + 'static,
    Sum<MetaSize, Prod<Z, U8>>: ArrayLength<u8> + PartialDiv<U8> + Unsigned + 'static,
    Sum<Quot<MetaSize, Z>, U8>: ArrayLength<u8> + PartialDiv<U8> + Unsigned + 'static,
{
    type Output = OcallORAMStorage<DataSize, MetaSize, Z>;
    type Error = UntrustedStorageError;

    fn create<Rng: RngCore + CryptoRng>(
        level: u32,
        size: u64,
        rng: &mut Rng,
    ) -> Result<Self::Output, Self::Error> {
        Ok(Self::Output::new(level, size, rng))
    }
}

/// An error type for when creating the OcallORAMStorage
// We actually panic on all of these errors, at least for now, because
// we can't really recover from them.
#[derive(Display, Debug)]
pub enum UntrustedStorageError {
    /// Untrusted could not allocate storage
    AllocationFailed,
}

mod helpers {
    use super::*;

    // Helper for invoking the checkout OCALL safely
    pub fn checkout_ocall<
        DataSize: ArrayLength<u8> + PartialDiv<U8>,
        MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    >(
        id: u64,
        idx: &[u64],
        data: &mut [A64Bytes<DataSize>],
        meta: &mut [A8Bytes<MetaSize>],
    ) {
        debug_assert!(idx.len() == data.len());
        debug_assert!(idx.len() == meta.len());
        unsafe {
            super::checkout_oram_storage(
                id,
                idx.as_ptr(),
                idx.len(),
                data.as_mut_ptr() as *mut u8,
                data.len() * DataSize::USIZE,
                meta.as_mut_ptr() as *mut u8,
                meta.len() * MetaSize::USIZE,
            )
        }
    }
    // Helper for invoking the checkin OCALL safely
    pub fn checkin_ocall<
        DataSize: ArrayLength<u8> + PartialDiv<U8>,
        MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    >(
        id: u64,
        idx: &[u64],
        data: &[A64Bytes<DataSize>],
        meta: &[A8Bytes<MetaSize>],
    ) {
        debug_assert!(idx.len() == data.len());
        debug_assert!(idx.len() == meta.len());
        unsafe {
            super::checkin_oram_storage(
                id,
                idx.as_ptr(),
                idx.len(),
                data.as_ptr() as *const u8,
                data.len() * DataSize::USIZE,
                meta.as_ptr() as *const u8,
                meta.len() * MetaSize::USIZE,
            )
        }
    }

    //for debug use
    pub fn pull_all_elements_ocall<
        DataSize: ArrayLength<u8> + PartialDiv<U8>,
        MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    >(
        allocation_id: u64,
        count: u64,
        aes_key: &GenericArray<u8, KeySize>,
    ) {
        {
            let mut data: Vec<GenericArray<u8, DataSize>> =
                vec![Default::default(); count as usize];
            let mut meta: Vec<GenericArray<u8, MetaSize>> =
                vec![Default::default(); count as usize];
            unsafe {
                pull_all_elements(
                    allocation_id,
                    data.as_mut_ptr() as *mut u8,
                    data.len() * DataSize::USIZE,
                    meta.as_mut_ptr() as *mut u8,
                    meta.len() * (MetaSize::USIZE + ExtraMetaSize::USIZE),
                );
            }
            let mut filtered = vec![];
            for idx in 0..count as usize {
                if meta[idx] == Default::default() {
                    continue;
                }
                let (meta_mut, extra_meta_mut) = meta[idx].split_at_mut(MetaSize::USIZE);
                let extra_meta = ExtraMeta::from(&*extra_meta_mut);
                let aes_nonce = make_aes_nonce(idx as u64, extra_meta.block_ctr);
                let mut cipher = CipherType::new(aes_key, &aes_nonce);
                cipher.apply_keystream(&mut data[idx]);
                cipher.apply_keystream(meta_mut);
                filtered.push((idx, data[idx].clone(), meta[idx].clone()));
            }
            println!("filtered = {:?}", filtered);
        }
    }
}

// This stuff must match edl file
extern "C" {
    fn pull_all_elements(
        id: u64,
        databuf: *mut u8,
        databuf_size: usize,
        metabuf: *mut u8,
        metabuf_size: usize,
    );
    fn allocate_oram_storage(
        level: u32,
        snapshot_id: u64,
        is_latest: u8,
        count: u64,
        data_size: u64,
        meta_size: u64,
        id: *mut u64,
    );
    fn allocate_shuffle_manager(
        allocation_id: u64,
        z: u64,
        bin_size_in_block: usize,
        shuffle_id: *mut u64,
    );
    fn persist_oram_storage(level: u32, snapshot_id: u64, is_volatile: u8, id: u64);
    fn checkout_oram_storage(
        id: u64,
        idx: *const u64,
        idx_len: usize,
        databuf: *mut u8,
        databuf_size: usize,
        metabuf: *mut u8,
        metabuf_size: usize,
    );
    fn checkin_oram_storage(
        id: u64,
        idx: *const u64,
        idx_len: usize,
        databuf: *const u8,
        databuf_size: usize,
        metabuf: *const u8,
        metabuf_size: usize,
    );
    pub fn get_valid_snapshot_id(
        size_triv_pos_map: u64,
        snapshot_id: *mut u64,
        lifetime_id_from_meta: *mut u64,
    );
    pub fn persist_stash(
        new_stash_data: *const u8,
        new_stash_data_len: usize,
        new_stash_meta: *const u8,
        new_stash_meta_len: usize,
        level: u32,
        new_snapshot_id: u64,
        is_volatile: u8,
    );
    pub fn recover_stash(
        stash_data: *mut u8,
        stash_data_len: usize,
        stash_meta: *mut u8,
        stash_meta_len: usize,
        level: u32,
        snapshot_id: u64,
    );
    pub fn persist_treetop(
        new_data: *const u8,
        new_data_len: usize,
        new_meta: *const u8,
        new_meta_len: usize,
        level: u32,
        new_snapshot_id: u64,
        is_volatile: u8,
    );
    pub fn recover_treetop(
        data: *mut u8,
        data_len: usize,
        meta: *mut u8,
        meta_len: usize,
        level: u32,
        snapshot_id: u64,
    );
    pub fn persist_merkle_roots(
        new_roots: *const u8,
        new_roots_len: usize,
        level: u32,
        new_snapshot_id: u64,
        is_volatile: u8,
    );
    pub fn recover_merkle_roots(roots: *mut u8, roots_len: usize, level: u32, snapshot_id: u64);
    pub fn persist_trivial_posmap(
        new_posmap: *const u8,
        new_posmap_len: usize,
        new_snapshot_id: u64,
        is_volatile: u8,
    );
    pub fn recover_trivial_posmap(posmap: *mut u8, posmap_len: usize, snapshot_id: u64);
}
