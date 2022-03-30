use crate::oram_manager::{DataMetaSize, PosMetaSize};
use crate::oram_storage::{compute_block_hash, make_aes_nonce, ExtraMeta, ExtraMetaSize, Hash};
use crate::oram_traits::{log2_ceil, HeapORAMStorage};
use crate::{AuthCipherType, AuthNonceSize, CipherType, KeySize, NonceSize, ALLOCATOR};
use aes::cipher::{NewCipher, StreamCipher};
use aligned_cmov::{
    cswap,
    subtle::{Choice, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess},
    typenum::{PartialDiv, PowerOfTwo, Prod, Quot, Sum, Unsigned, U32, U8},
    A64Bytes, A8Bytes, Aligned, ArrayLength, AsAlignedChunks, AsNeSlice, CMov, GenericArray, A8,
};
use blake2::{digest::Digest, Blake2b};
use rand_core::{CryptoRng, RngCore};
use sgx_trts::trts::rsgx_read_rand;
use std::convert::TryInto;
use std::ops::{Add, Deref, Div, Mul};
use std::time::Instant;
use std::untrusted::time::InstantEx;
use std::vec::Vec;

//The parameter in bucket oblivious sort
//For an overflow probability of 2^-80 and most reasonable values of n, Z = 512 suffices.
pub const BIN_SIZE_IN_BLOCK: usize = 512;
//assume dummy element random_key=0 or u64::MAX
const DUMMY_KEY_LEFT: u64 = 0;
const DUMMY_KEY_RIGHT: u64 = u64::MAX;
lazy_static! {
    // The key used to encrypt the tmp position map. Note that the tmp pos map is transfer across
    // levels of ORAM, so the key cannot use the aes_key or hash_key of each ORAM.
    pub static ref POSMAP_KEY: GenericArray<u8, KeySize> = GenericArray::<u8, KeySize>::default();
}

//return the hash of current node and the extrameta which keeps hashes of children
//this function is called when saving oram buckets
fn encrypt_bucket_and_pre_authenticate<DataSize, MetaSize>(
    idx: usize,
    i: usize,
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    data: &mut Vec<A64Bytes<DataSize>>,
    block_ctr: u64,
    meta_plus_extra: &mut Vec<A8Bytes<Sum<MetaSize, ExtraMetaSize>>>,
    treetop_max_count: usize,
    trusted_merkle_roots: &mut Vec<Hash>,
) -> Hash
where
    DataSize: ArrayLength<u8> + PowerOfTwo + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + Add<ExtraMetaSize>,
    Sum<MetaSize, ExtraMetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    // Split meta out of scratch buffer
    let (meta_mut, _) = meta_plus_extra[i].split_at_mut(MetaSize::USIZE);

    // Encrypt
    let aes_nonce = make_aes_nonce(idx as u64, block_ctr);
    let mut cipher = CipherType::new(aes_key, &aes_nonce);
    cipher.apply_keystream(&mut data[i]);
    cipher.apply_keystream(meta_mut);

    // Compute the hash for this block
    let this_block_hash = compute_block_hash(hash_key, &data[i], idx as u64, &meta_plus_extra[i]);

    // Check with trusted merkle root if this is the last treetop index
    if idx < 2 * treetop_max_count as usize {
        trusted_merkle_roots[idx] = this_block_hash;
    }

    this_block_hash
}

//return the hash of current node and the extrameta which keeps hashes of children
//this function is called when loading oram buckets
fn decrypt_bucket_and_pre_verify<DataSize, MetaSize>(
    idx: usize,
    i: usize,
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    data: &mut Vec<A64Bytes<DataSize>>,
    meta: &mut Vec<A8Bytes<MetaSize>>,
    meta_plus_extra: &mut Vec<A8Bytes<Sum<MetaSize, ExtraMetaSize>>>,
    treetop_max_count: usize,
    trusted_merkle_roots: &Vec<Hash>,
) -> (Hash, ExtraMeta)
where
    DataSize: ArrayLength<u8> + PowerOfTwo + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + Add<ExtraMetaSize>,
    Sum<MetaSize, ExtraMetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    // Compute the hash for this block
    let this_block_hash = compute_block_hash(hash_key, &data[i], idx as u64, &meta_plus_extra[i]);
    // Split extra_meta out of scratch buffer
    let (meta_mut, extra_meta_mut) = meta_plus_extra[i].split_at_mut(MetaSize::USIZE);
    let extra_meta = ExtraMeta::from(&*extra_meta_mut);
    // Decrypt
    let aes_nonce = make_aes_nonce(idx as u64, extra_meta.block_ctr);
    let mut cipher = CipherType::new(aes_key, &aes_nonce);
    cipher.apply_keystream(&mut data[i]);
    cipher.apply_keystream(meta_mut);
    meta[i].copy_from_slice(meta_mut);
    // Check with trusted merkle root if this is the last treetop index
    if idx < 2 * treetop_max_count as usize
        && !bool::from(trusted_merkle_roots[idx].ct_eq(&this_block_hash))
    {
        panic!("authentication failed, trusted merkle root {}", idx);
    }

    (this_block_hash, extra_meta)
}

//change the format of metadata
//current format of meta: (leaf num, block num, counter)
//after this process, the format of meta: (leaf num, block num, old idx);
//the old idx is block idx, while the new idx is first set to bucket idx
//for dummy block, new leaf = 0
//Note: the meta_item aggregate metadata from z blocks
fn reformat_metadata_first<Rng: RngCore + CryptoRng, MetaSize, Z>(
    count: u64,
    b_idx: usize,
    meta: &mut Vec<A8Bytes<MetaSize>>,
    rng: &mut Rng,
) where
    MetaSize: ArrayLength<u8> + Add<ExtraMetaSize> + Div<Z>,
    Z: Unsigned,
    Quot<MetaSize, Z>: ArrayLength<u8> + PartialDiv<U8>,
{
    assert_eq!(Quot::<MetaSize, Z>::USIZE, DataMetaSize::USIZE);
    for (i, meta_items) in meta.iter_mut().enumerate() {
        let meta_items: &mut [A8Bytes<Quot<MetaSize, Z>>] = meta_items.as_mut_aligned_chunks();
        for (idx_in_block, meta_item) in meta_items.iter_mut().enumerate() {
            //get old leaf num
            let old_leaf = meta_item.as_ne_u64_slice()[0];
            let is_vacant = old_leaf.ct_eq(&0);
            //test whether a metadata is vacant, i.e., refer to dummy block
            //assign new leaf num
            let mut new_leaf = (rng.next_u64() & ((count >> 1) - 1)) + (count >> 1);
            new_leaf.cmov(is_vacant, &0);
            let new_leaf_buf = new_leaf.to_ne_bytes();
            (&mut meta_item[0..8]).copy_from_slice(&new_leaf_buf);
            //assign MAX block num to dummy blocks
            let mut block_num = meta_item.as_ne_u64_slice()[1];
            block_num.cmov(is_vacant, &u64::MAX);
            meta_item.as_mut_ne_u64_slice()[1] = block_num;
            //assign old block idx
            let old_idx = (b_idx + i) * Z::USIZE + idx_in_block;
            (&mut meta_item[16..24]).copy_from_slice(&old_idx.to_ne_bytes());
        }
    }
}

//change the format of metadata
//current format of meta: (leaf num, block num, old idx)
//after this process, the format of meta: (leaf num, old idx, new idx);
//for dummy block, both leaf and new idx = 0
//we returned the data in next level ORAM
//Note: the meta_item aggregate metadata from z blocks
fn reformat_metadata_second<MetaSize>(meta: &mut Vec<A8Bytes<MetaSize>>)
where
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    for meta_item in meta.iter_mut() {
        //get new leaf num and old idx
        let new_leaf = meta_item.as_ne_u64_slice()[0];
        let old_idx = meta_item.as_ne_u64_slice()[2];
        //assign old block idx
        meta_item.as_mut_ne_u64_slice()[1] = old_idx;
        meta_item.as_mut_ne_u64_slice()[2] = new_leaf;
    }
}

//this interface supports separate keys for aes and hash
//this function is called when pushing bins

fn encrypt_and_authenticate_bin<Rng, DataSize, MetaSize>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    data: &mut [A64Bytes<DataSize>],
    meta: &mut [A8Bytes<MetaSize>],
    random_keys: &mut [u64],
    cur_bin_num: usize,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    rng: &mut Rng,
) -> (GenericArray<u8, NonceSize>, Hash)
where
    Rng: RngCore + CryptoRng,
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    let mut nonce = GenericArray::<u8, NonceSize>::default();
    rng.fill_bytes(nonce.as_mut_slice());
    let mut cipher = CipherType::new(aes_key, &nonce);
    let mut hasher = Blake2b::new();
    hasher.update(hash_key);
    hasher.update(&nonce);
    hasher.update(freshness_nonce);
    hasher.update(cur_bin_num.to_ne_bytes());

    for item in data {
        cipher.apply_keystream(item);
        hasher.update(item.as_ref().deref());
    }
    for item in meta {
        cipher.apply_keystream(item);
        hasher.update(item.as_ref().deref());
    }
    for item in random_keys {
        let mut item_buf = item.to_ne_bytes();
        cipher.apply_keystream(&mut item_buf);
        hasher.update(item_buf);
        *item = u64::from_ne_bytes(item_buf);
    }
    let result = hasher.finalize();
    (nonce, result[..16].try_into().unwrap())
}

//this interface supports separate keys for aes and hash
//this function is called when pulling bins
fn decrypt_and_verify_bin<DataSize, MetaSize>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    data: &mut [A64Bytes<DataSize>],
    meta: &mut [A8Bytes<MetaSize>],
    random_keys: &mut [u64],
    cur_bin_num: usize,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    nonce: &GenericArray<u8, NonceSize>,
    hash: &Hash,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    let mut cipher = CipherType::new(aes_key, nonce);
    let mut hasher = Blake2b::new();
    hasher.update(hash_key);
    hasher.update(&nonce);
    hasher.update(freshness_nonce);
    hasher.update(cur_bin_num.to_ne_bytes());

    for item in data {
        hasher.update(item.as_ref().deref());
        cipher.apply_keystream(item);
    }
    for item in meta {
        hasher.update(item.as_ref().deref());
        cipher.apply_keystream(item);
    }
    for item in random_keys {
        let mut item_buf = item.to_ne_bytes();
        hasher.update(item_buf);
        cipher.apply_keystream(&mut item_buf);
        *item = u64::from_ne_bytes(item_buf);
    }
    let loaded_hash: Hash = hasher.finalize()[..16].try_into().unwrap();
    assert_eq!(&loaded_hash, hash);
}

//bin_type: 0 for idle bin, 1 for work bin
fn push_bin<Rng, DataSize, MetaSize>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    cur_bin_num: usize,
    bin_type: u8,
    data: &mut [A64Bytes<DataSize>],
    meta: &mut [A8Bytes<MetaSize>],
    random_keys: &mut [u64],
    rng: &mut Rng,
) where
    Rng: RngCore + CryptoRng,
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    let (nonce, hash) = encrypt_and_authenticate_bin(
        aes_key,
        hash_key,
        data,
        meta,
        random_keys,
        cur_bin_num,
        &freshness_nonce,
        rng,
    );

    helpers::shuffle_push_bin_ocall(
        shuffle_id,
        cur_bin_num,
        bin_type,
        data,
        meta,
        random_keys,
        &nonce,
        &hash,
    );
}

//after pull_bin, the bin space in the untrusted domain is released
//bin_type: 0 for idle bin, 1 for work bin
fn pull_bin<DataSize, MetaSize>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    cur_bin_num: usize,
    bin_type: u8,
    bin_size: &mut usize,
    data: &mut [A64Bytes<DataSize>],
    meta: &mut [A8Bytes<MetaSize>],
    random_keys: &mut [u64],
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    let mut nonce = GenericArray::<u8, NonceSize>::default();
    let mut hash = Default::default();
    helpers::shuffle_pull_bin_ocall(
        shuffle_id,
        cur_bin_num,
        bin_type,
        bin_size,
        data,
        meta,
        random_keys,
        &mut nonce,
        &mut hash,
    );
    let has_data = (data.len() > 0) as usize;
    let has_meta = (meta.len() > 0) as usize;
    let has_random_keys = (random_keys.len() > 0) as usize;
    decrypt_and_verify_bin(
        aes_key,
        hash_key,
        &mut data[..*bin_size * has_data],
        &mut meta[..*bin_size * has_meta],
        &mut random_keys[..*bin_size * has_random_keys],
        cur_bin_num,
        &freshness_nonce,
        &nonce,
        &hash,
    );
}

fn pull_oram_buckets<Rng, DataSize, MetaSize, Z>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    seal_nonce: &GenericArray<u8, NonceSize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    count: u64,
    treetop_max_count: usize,
    treetop: &mut HeapORAMStorage<DataSize, MetaSize, Z>,
    trusted_merkle_roots: &mut Vec<Hash>,
    num_bins: usize,
    bin_size_in_bucket_real: usize,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PowerOfTwo + PartialDiv<U8> + Div<Z>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8> + Add<ExtraMetaSize> + Div<Z>,
    Z: Unsigned,
    Rng: RngCore + CryptoRng,
    Quot<MetaSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Add<U8> + Unsigned,
    Sum<MetaSize, ExtraMetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    let now = Instant::now();
    //reassign bin_size_in_bucket because we don't pad now
    let bin_size_in_bucket = bin_size_in_bucket_real;
    let mut cur_bin_num = num_bins - 1;
    let mut e_count = count as usize;
    let mut b_count = e_count >> 1;

    let mut prev_tier_hash: Hash = Default::default();

    while b_count > 0 {
        let mut e_idx = e_count;
        if e_count <= bin_size_in_bucket {
            assert_eq!(e_idx, bin_size_in_bucket);
            assert_eq!(cur_bin_num, 0);
            let mut data: Vec<A64Bytes<DataSize>> = vec![Default::default(); bin_size_in_bucket];
            let mut meta: Vec<A8Bytes<MetaSize>> = vec![Default::default(); bin_size_in_bucket];
            let mut meta_plus_extra: Vec<A8Bytes<Sum<MetaSize, ExtraMetaSize>>> =
                vec![Default::default(); bin_size_in_bucket];
            let b_idx = e_idx - bin_size_in_bucket;
            let delta = bin_size_in_bucket.saturating_sub(treetop_max_count);
            //if the bin size is larger than treetop max count
            if delta > 0 {
                let mut is_bottom_in_bucket = true;
                helpers::shuffle_pull_buckets_ocall(
                    shuffle_id,
                    e_idx - delta,
                    e_idx,
                    &mut data[bin_size_in_bucket - delta..],
                    &mut meta_plus_extra[bin_size_in_bucket - delta..],
                );
                while b_count >= treetop_max_count {
                    let mut prev_tier_hasher = Blake2b::new();
                    prev_tier_hasher.update(hash_key);
                    for i in (b_count..e_count).rev() {
                        if meta_plus_extra[i] != Default::default() {
                            let idx = b_idx + i;
                            let (this_block_hash, extra_meta) = decrypt_bucket_and_pre_verify(
                                idx,
                                i,
                                aes_key,
                                hash_key,
                                &mut data,
                                &mut meta,
                                &mut meta_plus_extra,
                                treetop_max_count,
                                trusted_merkle_roots,
                            );
                            // If this block has a child, check if its hash that we computed before matches
                            // metadata. And we should verify the prev_hashes before assertion.
                            if is_bottom_in_bucket && e_count < count as usize {
                                if extra_meta.right_child_hash != Hash::default() {
                                    prev_tier_hasher.update(&extra_meta.right_child_hash);
                                }
                                if extra_meta.left_child_hash != Hash::default() {
                                    prev_tier_hasher.update(&extra_meta.left_child_hash);
                                }
                            }
                            // No need to authenticate the block hash, no need for additionally storing hashes anymore
                            // we can just check against already-fetched metadata of parents
                            if idx >= 2 * treetop_max_count {
                                let (_, parent_extra_meta_mut) =
                                    meta_plus_extra[idx >> 1].split_at_mut(MetaSize::USIZE);
                                let parent_extra_meta = ExtraMeta::from(&*parent_extra_meta_mut);
                                if idx & 1 == 0 {
                                    assert_eq!(this_block_hash, parent_extra_meta.left_child_hash);
                                } else {
                                    assert_eq!(this_block_hash, parent_extra_meta.right_child_hash);
                                }
                            }
                        }
                    }
                    //if the current tier is not the lowest tier but is the lowest tier in the bucket
                    if is_bottom_in_bucket && e_count < count as usize {
                        let h: Hash = prev_tier_hasher.finalize()[..16].try_into().unwrap();
                        assert_eq!(prev_tier_hash, h);
                    }

                    is_bottom_in_bucket = false;
                    e_count = b_count;
                    b_count >>= 1;
                }
            }
            (&mut data[..bin_size_in_bucket - delta])
                .clone_from_slice(&treetop.data[b_idx..e_idx - delta]);
            (&mut meta[..bin_size_in_bucket - delta])
                .clone_from_slice(&treetop.metadata[b_idx..e_idx - delta]);

            //write data and meta back separately with encryption and authentication
            //in the untrusted domain, data may saved on disk, while meta almost in memory for further process
            let mut original_meta = meta.clone();

            push_bin::<_, DataSize, MetaSize>(
                aes_key,
                hash_key,
                seal_nonce,
                shuffle_id,
                cur_bin_num,
                0,
                &mut data,
                &mut Vec::new(),
                &mut Vec::new(),
                rng,
            );
            push_bin::<_, DataSize, MetaSize>(
                aes_key,
                hash_key,
                seal_nonce,
                shuffle_id,
                cur_bin_num,
                0,
                &mut Vec::new(),
                &mut original_meta,
                &mut Vec::new(),
                rng,
            );

            //compute new idx from block num
            reformat_metadata_first(count, b_idx, &mut meta, rng);
            push_bin::<_, DataSize, MetaSize>(
                aes_key,
                hash_key,
                freshness_nonce,
                shuffle_id,
                cur_bin_num,
                1,
                &mut Vec::new(),
                &mut meta,
                &mut Vec::new(),
                rng,
            );

            b_count = 0;
        } else {
            let mut cur_tier_hasher = Blake2b::new();
            let mut prev_tier_hasher = Blake2b::new();
            cur_tier_hasher.update(hash_key);
            prev_tier_hasher.update(hash_key);
            while e_idx > b_count {
                let mut data: Vec<A64Bytes<DataSize>> =
                    vec![Default::default(); bin_size_in_bucket];
                let mut meta: Vec<A8Bytes<MetaSize>> = vec![Default::default(); bin_size_in_bucket];
                let mut meta_plus_extra: Vec<A8Bytes<Sum<MetaSize, ExtraMetaSize>>> =
                    vec![Default::default(); bin_size_in_bucket];
                let b_idx = e_idx - bin_size_in_bucket;
                let mut delta = bin_size_in_bucket;
                if e_idx > treetop_max_count {
                    delta = treetop_max_count.saturating_sub(b_idx);
                    assert_eq!(delta, 0);
                    helpers::shuffle_pull_buckets_ocall(
                        shuffle_id,
                        b_idx + delta,
                        e_idx,
                        &mut data[delta..],
                        &mut meta_plus_extra[delta..],
                    );
                    //decrypt data and meta, and verify
                    for i in (delta..bin_size_in_bucket).rev() {
                        if meta_plus_extra[i] != Default::default() {
                            let idx = b_idx + i; //bucket idx
                            let (this_block_hash, extra_meta) = decrypt_bucket_and_pre_verify(
                                idx,
                                i,
                                aes_key,
                                hash_key,
                                &mut data,
                                &mut meta,
                                &mut meta_plus_extra,
                                treetop_max_count,
                                trusted_merkle_roots,
                            );
                            // If this block has a child, check if its hash that we computed before matches
                            // metadata.

                            if e_count < count as usize {
                                if extra_meta.right_child_hash != Hash::default() {
                                    prev_tier_hasher.update(&extra_meta.right_child_hash);
                                }
                                if extra_meta.left_child_hash != Hash::default() {
                                    prev_tier_hasher.update(&extra_meta.left_child_hash);
                                }
                            }
                            //authenticate the hashes because the hashes are stored in untrusted domain
                            if b_idx >= 2 * treetop_max_count && this_block_hash != Hash::default()
                            {
                                cur_tier_hasher.update(&this_block_hash);
                            }
                        }
                    }
                }
                //copy from treetop to the buffers above
                //copy is necessary because the original meta should be kept
                //no need to decrypt and verify
                if b_idx < treetop_max_count {
                    (&mut data[..delta]).clone_from_slice(&treetop.data[b_idx..b_idx + delta]);
                    (&mut meta[..delta]).clone_from_slice(&treetop.metadata[b_idx..b_idx + delta]);
                }

                //write data and meta back separately with encryption and authentication
                //in the untrusted domain, data may saved on disk, while meta almost in memory for further process
                let mut original_meta = meta.clone();

                push_bin::<_, DataSize, MetaSize>(
                    aes_key,
                    hash_key,
                    seal_nonce,
                    shuffle_id,
                    cur_bin_num,
                    0,
                    &mut data,
                    &mut Vec::new(),
                    &mut Vec::new(),
                    rng,
                );
                push_bin::<_, DataSize, MetaSize>(
                    aes_key,
                    hash_key,
                    seal_nonce,
                    shuffle_id,
                    cur_bin_num,
                    0,
                    &mut Vec::new(),
                    &mut original_meta,
                    &mut Vec::new(),
                    rng,
                );

                //compute new idx from block num
                reformat_metadata_first(count, b_idx, &mut meta, rng);
                //although the DataSize, MetaSize is not consistent with subsequent
                //pull_bin and push_bin in bucket oblivious sort, it does not matter
                push_bin::<_, DataSize, MetaSize>(
                    aes_key,
                    hash_key,
                    freshness_nonce,
                    shuffle_id,
                    cur_bin_num,
                    1,
                    &mut Vec::new(),
                    &mut meta,
                    &mut Vec::new(),
                    rng,
                );

                cur_bin_num -= 1;
                e_idx = b_idx;
            }
            if e_count < count as usize {
                let h: Hash = prev_tier_hasher.finalize()[..16].try_into().unwrap();
                assert_eq!(prev_tier_hash, h);
            }
            prev_tier_hash = cur_tier_hasher.finalize()[..16].try_into().unwrap();
            e_count = b_count;
            b_count >>= 1;
        }
    }

    unsafe {
        bin_switch(shuffle_id, 0, num_bins);
    }

    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("finish pull all oram buckets, {:?}s", dur);
}

pub fn manage<DataSize, MetaSize, Z, Rng>(
    level: u32,
    shuffle_id: u64,
    allocation_id: u64,
    count: u64,
    treetop_max_count: usize,
    treetop: &mut HeapORAMStorage<DataSize, MetaSize, Z>,
    trusted_merkle_roots: &mut Vec<Hash>,
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PowerOfTwo + PartialDiv<U8> + Div<Z>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8> + Add<ExtraMetaSize> + Add<Prod<Z, U8>> + Div<Z>,
    Z: Unsigned + Mul<U8>,
    Rng: RngCore + CryptoRng,
    Sum<MetaSize, ExtraMetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, U8>: Unsigned,
    Quot<DataSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
    Quot<MetaSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Add<U8> + Unsigned,
    Sum<MetaSize, Prod<Z, U8>>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
    Sum<Quot<MetaSize, Z>, U8>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
{
    //seal nonce is needed to store original data and metadata
    let mut seal_nonce = GenericArray::<u8, NonceSize>::default();
    //freshness nonce should change every time a round in bucket oblivious sort is finished
    let mut freshness_nonce = GenericArray::<u8, NonceSize>::default();
    //freshness nonce used for aes gcm
    let mut freshness_auth_nonce = GenericArray::<u8, AuthNonceSize>::default();

    rsgx_read_rand(&mut seal_nonce).unwrap();
    rsgx_read_rand(&mut freshness_nonce).unwrap();
    rsgx_read_rand(&mut freshness_auth_nonce).unwrap();

    //read and verify buckets and extract info from metadata
    //specifically, (old leaf num, block num, counter), we use bin
    //and bucket for buckets in bucket oblivious sort and oram, respectively
    let bin_size_in_bucket = BIN_SIZE_IN_BLOCK / Z::USIZE;
    let bin_size_in_bucket_real = bin_size_in_bucket >> 1;
    let mut num_bins = count as usize / bin_size_in_bucket_real;
    if num_bins == 0 {
        num_bins = 1;
    }
    assert!(
        num_bins & (num_bins - 1) == 0,
        "num_bins must be a power of two"
    );
    //reorganize buckets and prepare bins for subsequent shuffle
    //only if level = 0, it is possible that the inputs are buckets
    //Currently we only implement case 2, the inputs are definitely buckets for level = 0
    if level == 0 {
        println!("begin pull all oram buckets");
        pull_oram_buckets::<_, DataSize, MetaSize, Z>(
            aes_key,
            hash_key,
            &seal_nonce,
            &freshness_nonce,
            shuffle_id,
            count,
            treetop_max_count,
            treetop,
            trusted_merkle_roots,
            num_bins,
            bin_size_in_bucket_real,
            rng,
        );

        let now = Instant::now();
        //sort by logical address, i.e., block num
        bucket_oblivious_sort::<_, Quot<DataSize, Z>, Quot<MetaSize, Z>, Z>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            num_bins,
            0,
            1,
            false,
            rng,
        );
        let dur = now.elapsed().as_nanos() as f64 * 1e-9;
        println!(
            "finish bucket oblivious sort by locigal address, {:?}s",
            dur
        );
        let now = Instant::now();
        //place by logical address
        oblivious_placement::<_, Quot<DataSize, Z>, Quot<MetaSize, Z>>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            num_bins,
            1,
            u64::MAX,
            rng,
        );
        let dur = now.elapsed().as_nanos() as f64 * 1e-9;
        println!("finish oblivious placement by locigal address, {:?}s", dur);

        shuffle_core::<_, Quot<DataSize, Z>, Quot<MetaSize, Z>, Z>(
            aes_key,
            hash_key,
            &seal_nonce,
            &freshness_nonce,
            shuffle_id,
            count,
            num_bins,
            rng,
        );
    } else {
        let now = Instant::now();
        oblivious_pull_tmp_posmap::<_, Quot<DataSize, Z>, Sum<Quot<MetaSize, Z>, U8>, Z>(
            aes_key,
            hash_key,
            &seal_nonce,
            &freshness_nonce,
            shuffle_id,
            count,
            num_bins,
            rng,
        );
        let dur = now.elapsed().as_nanos() as f64 * 1e-9;
        println!("finish oblivious pull tmp posmap, {:?}s", dur);

        shuffle_core::<_, Quot<DataSize, Z>, Sum<Quot<MetaSize, Z>, U8>, Z>(
            aes_key,
            hash_key,
            &seal_nonce,
            &freshness_nonce,
            shuffle_id,
            count,
            num_bins,
            rng,
        );
    }

    let now = Instant::now();
    //authenticate blocks as in oram tree.
    {
        fn strip_counter_part<DataSize, SrcMetaSize, DstMetaSize, Z>(
            aes_key: &GenericArray<u8, KeySize>,
            hash_key: &GenericArray<u8, KeySize>,
            seal_nonce: &GenericArray<u8, NonceSize>,
            shuffle_id: u64,
            cur_bin_num: usize,
            bin_size_in_bucket: usize,
        ) -> (Vec<A64Bytes<DataSize>>, Vec<A8Bytes<DstMetaSize>>)
        where
            DataSize: ArrayLength<u8> + PowerOfTwo + PartialDiv<U8>,
            SrcMetaSize: ArrayLength<u8> + PartialDiv<U8>,
            DstMetaSize: ArrayLength<u8> + PartialDiv<U8> + Div<Z>,
            Z: Unsigned,
            Quot<DstMetaSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Add<U8> + Unsigned,
            Sum<Quot<DstMetaSize, Z>, U8>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
        {
            let mut actual_bin_size = 0;
            let mut data = vec![Default::default(); bin_size_in_bucket];
            let mut src_meta = vec![Default::default(); bin_size_in_bucket];
            pull_bin::<DataSize, SrcMetaSize>(
                aes_key,
                hash_key,
                seal_nonce,
                shuffle_id,
                cur_bin_num,
                1,
                &mut actual_bin_size,
                &mut data,
                &mut src_meta,
                &mut Vec::new(),
            );
            assert_eq!(actual_bin_size, src_meta.len());
            let mut dst_meta: Vec<A8Bytes<DstMetaSize>> =
                vec![Default::default(); bin_size_in_bucket];
            for (dst_meta_items, src_meta_items) in dst_meta.iter_mut().zip(src_meta.iter()) {
                let dst_meta_items: &mut [A8Bytes<Quot<DstMetaSize, Z>>] =
                    dst_meta_items.as_mut_aligned_chunks();
                let src_meta_items: &[A8Bytes<Sum<Quot<DstMetaSize, Z>, U8>>] =
                    src_meta_items.as_aligned_chunks();
                for (dst_meta_item, src_meta_item) in
                    dst_meta_items.iter_mut().zip(src_meta_items.iter())
                {
                    dst_meta_item.copy_from_slice(&src_meta_item[..Quot::<DstMetaSize, Z>::USIZE]);
                }
            }
            (data, dst_meta)
        }

        //clear the new idx, that is, restore the function of counter and reset it to 0
        fn reset_counter_part<DataSize, MetaSize, Z>(
            aes_key: &GenericArray<u8, KeySize>,
            hash_key: &GenericArray<u8, KeySize>,
            seal_nonce: &GenericArray<u8, NonceSize>,
            shuffle_id: u64,
            cur_bin_num: usize,
            bin_size_in_bucket: usize,
        ) -> (Vec<A64Bytes<DataSize>>, Vec<A8Bytes<MetaSize>>)
        where
            DataSize: ArrayLength<u8> + PowerOfTwo + PartialDiv<U8>,
            MetaSize: ArrayLength<u8> + PartialDiv<U8> + Div<Z>,
            Z: Unsigned,
            Quot<MetaSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Add<U8> + Unsigned,
        {
            let mut actual_bin_size = 0;
            let mut data = vec![Default::default(); bin_size_in_bucket];
            let mut meta = vec![Default::default(); bin_size_in_bucket];
            pull_bin::<DataSize, MetaSize>(
                aes_key,
                hash_key,
                seal_nonce,
                shuffle_id,
                cur_bin_num,
                1,
                &mut actual_bin_size,
                &mut data,
                &mut meta,
                &mut Vec::new(),
            );
            assert_eq!(actual_bin_size, meta.len());
            //clear the new idx
            for items_ in meta.iter_mut() {
                let items: &mut [A8Bytes<Quot<MetaSize, Z>>] = items_.as_mut_aligned_chunks();
                for item in items {
                    (&mut item[16..24]).copy_from_slice(&[0; 8]);
                }
            }
            (data, meta)
        }

        //reassign bin_size_in_bucket because no pad now
        let bin_size_in_bucket = bin_size_in_bucket_real;
        let mut cur_bin_num = num_bins - 1;
        let mut e_count = count as usize;
        let mut b_count = e_count >> 1;

        ALLOCATOR.set_switch(true);
        let mut hashes = vec![Default::default(); b_count];
        let mut prev_hashes: Vec<Hash> = Vec::new();
        ALLOCATOR.set_switch(false);

        let mut prev_tier_hash: Hash = Default::default();

        while b_count > 0 {
            let mut e_idx = e_count;
            if e_count <= bin_size_in_bucket {
                assert_eq!(cur_bin_num, 0);
                let (mut data, mut meta) = if level > 0 {
                    strip_counter_part::<DataSize, Sum<MetaSize, Prod<Z, U8>>, MetaSize, Z>(
                        aes_key,
                        hash_key,
                        &seal_nonce,
                        shuffle_id,
                        cur_bin_num,
                        bin_size_in_bucket,
                    )
                } else {
                    reset_counter_part(
                        aes_key,
                        hash_key,
                        &seal_nonce,
                        shuffle_id,
                        cur_bin_num,
                        bin_size_in_bucket,
                    )
                };
                data.truncate(e_idx);
                meta.truncate(e_idx);

                let mut meta_plus_extra: Vec<A8Bytes<Sum<MetaSize, ExtraMetaSize>>> =
                    vec![Default::default(); e_idx];
                let b_idx = 0;
                let delta = e_idx.saturating_sub(treetop_max_count);
                if delta > 0 {
                    let mut is_bottom_in_bucket = true;
                    while b_count >= treetop_max_count {
                        //encrypt data and meta, and authenticate
                        let mut prev_tier_hasher = Blake2b::new();
                        prev_tier_hasher.update(hash_key);
                        for i in (b_count..e_count).rev() {
                            let idx = b_idx + i;

                            // Update the metadata field and extract the new block_ctr value so that we can encrypt
                            // TODO: the block ctr should be larger than any previous version
                            // now for convenience, we do not obey it
                            let block_ctr = {
                                // Split extra_meta out of scratch buffer
                                let (meta_mut, extra_meta_mut) =
                                    meta_plus_extra[i].split_at_mut(MetaSize::USIZE);

                                // Update the meta
                                meta_mut.copy_from_slice(&meta[i]);

                                // Update the extra_meta
                                let mut extra_meta_val = ExtraMeta::from(&*extra_meta_mut);

                                // If this block has a child, update extra_meta
                                // If this block has a child, check if its hash that we computed before matches
                                // metadata. And we should verify the prev_hashes before assertion.
                                if is_bottom_in_bucket && e_count < count as usize {
                                    let left_child_hash = prev_hashes[(idx << 1) - e_count];
                                    let right_child_hash = prev_hashes[(idx << 1) + 1 - e_count];
                                    if right_child_hash != Hash::default() {
                                        prev_tier_hasher.update(right_child_hash);
                                    }
                                    if left_child_hash != Hash::default() {
                                        prev_tier_hasher.update(left_child_hash);
                                    }
                                    extra_meta_val.left_child_hash = left_child_hash;
                                    extra_meta_val.right_child_hash = right_child_hash;
                                }

                                // Update block_ctr value by incrementing it
                                extra_meta_val.block_ctr += 1;

                                // Serialize the ExtraMeta object to bytes and store them at extra_meta
                                let extra_meta_bytes =
                                    GenericArray::<u8, ExtraMetaSize>::from(&extra_meta_val);
                                extra_meta_mut.copy_from_slice(extra_meta_bytes.as_slice());

                                // Return the block_ctr value to use for this encryption
                                extra_meta_val.block_ctr
                            };

                            // Encrypt the data that is supposed to be encrypted
                            let this_block_hash = encrypt_bucket_and_pre_authenticate::<_, MetaSize>(
                                idx,
                                i,
                                aes_key,
                                hash_key,
                                &mut data,
                                block_ctr,
                                &mut meta_plus_extra,
                                treetop_max_count,
                                trusted_merkle_roots,
                            );
                            // No need to authenticate the block hash, no need for additionally storing hashes anymore
                            // we can just fill the metadata of parents
                            if idx >= 2 * treetop_max_count {
                                let (_, parent_extra_meta_mut) =
                                    meta_plus_extra[idx >> 1].split_at_mut(MetaSize::USIZE);
                                if idx & 1 == 0 {
                                    (&mut parent_extra_meta_mut[8..24])
                                        .clone_from_slice(&this_block_hash);
                                } else {
                                    (&mut parent_extra_meta_mut[24..40])
                                        .clone_from_slice(&this_block_hash);
                                }
                            }
                        }
                        //if the current tier is not the lowest tier but is the lowest tier in the bucket
                        if is_bottom_in_bucket && e_count < count as usize {
                            let h: Hash = prev_tier_hasher.finalize()[..16].try_into().unwrap();
                            assert_eq!(prev_tier_hash, h);
                        }

                        is_bottom_in_bucket = false;
                        e_count = b_count;
                        b_count >>= 1;
                    }
                }

                (&mut treetop.data[..e_idx - delta]).clone_from_slice(&data[..e_idx - delta]);
                (&mut treetop.metadata[..e_idx - delta]).clone_from_slice(&meta[..e_idx - delta]);
                //save the oram buckets
                helpers::shuffle_push_buckets_ocall(
                    shuffle_id,
                    b_idx,
                    e_idx,
                    &data,
                    &meta_plus_extra,
                );

                b_count = 0;
            } else {
                let mut cur_tier_hasher = Blake2b::new();
                let mut prev_tier_hasher = Blake2b::new();
                cur_tier_hasher.update(hash_key);
                prev_tier_hasher.update(hash_key);
                while e_idx > b_count {
                    let (mut data, meta) = if level > 0 {
                        strip_counter_part::<DataSize, Sum<MetaSize, Prod<Z, U8>>, MetaSize, Z>(
                            aes_key,
                            hash_key,
                            &seal_nonce,
                            shuffle_id,
                            cur_bin_num,
                            bin_size_in_bucket,
                        )
                    } else {
                        reset_counter_part(
                            aes_key,
                            hash_key,
                            &seal_nonce,
                            shuffle_id,
                            cur_bin_num,
                            bin_size_in_bucket,
                        )
                    };
                    let mut meta_plus_extra: Vec<A8Bytes<Sum<MetaSize, ExtraMetaSize>>> =
                        vec![Default::default(); bin_size_in_bucket];
                    let b_idx = e_idx - bin_size_in_bucket;
                    let mut delta = bin_size_in_bucket;
                    if e_idx > treetop_max_count {
                        delta = treetop_max_count.saturating_sub(b_idx);
                        assert_eq!(delta, 0);
                        //encrypt data and meta, and authenticate
                        for i in (delta..bin_size_in_bucket).rev() {
                            let idx = b_idx + i; //bucket_idx

                            // Update the metadata field and extract the new block_ctr value so that we can encrypt
                            // TODO: the block ctr should be larger than any previous version
                            // now for convenience, we do not obey it
                            let block_ctr = {
                                // Split extra_meta out of scratch buffer
                                let (meta_mut, extra_meta_mut) =
                                    meta_plus_extra[i].split_at_mut(MetaSize::USIZE);

                                // Update the meta
                                meta_mut.copy_from_slice(&meta[i]);

                                // Update the extra_meta
                                let mut extra_meta_val = ExtraMeta::from(&*extra_meta_mut);

                                // If this block has a child, update extra_meta
                                // If this block has a child, check if its hash that we computed before matches
                                // metadata. And we should verify the prev_hashes before assertion.
                                if e_count < count as usize {
                                    let left_child_hash = prev_hashes[(idx << 1) - e_count];
                                    let right_child_hash = prev_hashes[(idx << 1) + 1 - e_count];
                                    if right_child_hash != Hash::default() {
                                        prev_tier_hasher.update(right_child_hash);
                                    }
                                    if left_child_hash != Hash::default() {
                                        prev_tier_hasher.update(left_child_hash);
                                    }
                                    extra_meta_val.left_child_hash = left_child_hash;
                                    extra_meta_val.right_child_hash = right_child_hash;
                                }

                                // Update block_ctr value by incrementing it
                                extra_meta_val.block_ctr += 1;

                                // Serialize the ExtraMeta object to bytes and store them at extra_meta
                                let extra_meta_bytes =
                                    GenericArray::<u8, ExtraMetaSize>::from(&extra_meta_val);
                                extra_meta_mut.copy_from_slice(extra_meta_bytes.as_slice());

                                // Return the block_ctr value to use for this encryption
                                extra_meta_val.block_ctr
                            };

                            let this_block_hash = encrypt_bucket_and_pre_authenticate::<_, MetaSize>(
                                idx,
                                i,
                                aes_key,
                                hash_key,
                                &mut data,
                                block_ctr,
                                &mut meta_plus_extra,
                                treetop_max_count,
                                trusted_merkle_roots,
                            );
                            //authenticate the hashes because the hashes are stored in untrusted domain
                            if b_idx >= 2 * treetop_max_count && this_block_hash != Hash::default()
                            {
                                cur_tier_hasher.update(this_block_hash);
                                hashes[idx - b_count] = this_block_hash;
                            }
                        }
                    }
                    //copy from buffers to the treetop above
                    //no need to encrypt and authenticate
                    if b_idx < treetop_max_count {
                        (&mut treetop.data[b_idx..b_idx + delta]).clone_from_slice(&data[..delta]);
                        (&mut treetop.metadata[b_idx..b_idx + delta])
                            .clone_from_slice(&meta[..delta]);
                    }

                    //save the oram buckets
                    helpers::shuffle_push_buckets_ocall(
                        shuffle_id,
                        b_idx,
                        e_idx,
                        &data,
                        &meta_plus_extra,
                    );

                    cur_bin_num -= 1;
                    e_idx = b_idx;
                }
                if e_count < count as usize {
                    let h: Hash = prev_tier_hasher.finalize()[..16].try_into().unwrap();
                    assert_eq!(prev_tier_hash, h);
                }
                prev_tier_hash = cur_tier_hasher.finalize()[..16].try_into().unwrap();
                e_count = b_count;
                b_count >>= 1;
                ALLOCATOR.set_switch(true);
                drop(prev_hashes);
                prev_hashes = hashes;
                hashes = vec![Default::default(); b_count];
                ALLOCATOR.set_switch(false);
            }
        }
        ALLOCATOR.set_switch(true);
        drop(hashes);
        drop(prev_hashes);
        ALLOCATOR.set_switch(false);
    }
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("finish oram bucket push, {:?}s", dur);

    //build
    //handle the processed array to oram tree
    unsafe {
        build_oram_from_shuffle_manager(shuffle_id, allocation_id);
    }
    println!("finish build oram from shuffle manager");
}

fn bucket_oblivious_sort<Rng, DataSize, MetaSize, Z>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    num_bins: usize,
    first_real_bin: usize,
    key_by: usize,
    has_data: bool,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned,
    Rng: RngCore + CryptoRng,
{
    if num_bins == 1 {
        let bin_size = BIN_SIZE_IN_BLOCK / 2;
        let mut meta = vec![Default::default(); bin_size];
        let mut data = vec![Default::default(); bin_size * has_data as usize];
        if has_data {
            pull_bin::<DataSize, MetaSize>(
                aes_key,
                hash_key,
                freshness_nonce,
                shuffle_id,
                first_real_bin,
                0,
                &mut 0,
                &mut data,
                &mut Vec::new(),
                &mut Vec::new(),
            );
            pull_bin::<DataSize, MetaSize>(
                aes_key,
                hash_key,
                freshness_nonce,
                shuffle_id,
                first_real_bin,
                0,
                &mut 0,
                &mut Vec::new(),
                &mut meta,
                &mut Vec::new(),
            );
        } else {
            pull_bin::<DataSize, MetaSize>(
                aes_key,
                hash_key,
                freshness_nonce,
                shuffle_id,
                first_real_bin,
                1,
                &mut 0,
                &mut Vec::new(),
                &mut meta,
                &mut Vec::new(),
            );
        }
        bitonic_sort(&mut data, &mut meta, &mut Vec::new(), key_by, true);
        push_bin::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            first_real_bin,
            1,
            &mut data,
            &mut meta,
            &mut Vec::new(),
            rng,
        );
        unsafe {
            bin_switch(shuffle_id, first_real_bin, first_real_bin + num_bins);
        }
    } else {
        //oblivious random bin assignment
        oblivious_random_bin_assignment::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            num_bins,
            first_real_bin,
            has_data,
            rng,
        );
        //oblivious random permutation
        //note that the bins read may not be equal size
        //so in this step, we adjust the bins to the same size
        oblivious_random_permutation::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            num_bins,
            first_real_bin,
            has_data,
            rng,
        );
        //merge sort by new idx, which is more suitable for enclave setting
        non_oblivious_merge_sort::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            num_bins,
            first_real_bin,
            key_by,
            has_data,
            rng,
        );
    }
}

fn oblivious_random_bin_assignment<Rng, DataSize, MetaSize>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    num_bins: usize,
    first_real_bin: usize,
    has_data: bool,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Rng: RngCore + CryptoRng,
{
    let log_b = (num_bins as f64).log2() as usize;
    let n = num_bins * BIN_SIZE_IN_BLOCK / 2;
    let mut pow_i = 1;
    for i in 0..log_b {
        //TODO: for each i, the freshness_nonce should change
        let msb_mask = (n >> (i + 1)) as u64;
        for j in 0..num_bins / 2 {
            let j_prime = (j >> i) << i;
            //operate at block level, not bucket level
            let bin_size = BIN_SIZE_IN_BLOCK;
            //allocate space for two bins
            let mut data = vec![Default::default(); 2 * bin_size * (has_data as usize)];
            let mut meta = vec![Default::default(); 2 * bin_size];
            let mut random_keys = vec![Default::default(); 2 * bin_size];

            for c in [0, 1] {
                if i == 0 && has_data {
                    //note that data and meta are seperately encrypted and authenticated
                    //moreover, bucket oblivious sort is called only once to sort data and meta together
                    //and bin_type = 0, for that separate data and meta are fetched from idle bins
                    pull_bin::<DataSize, MetaSize>(
                        aes_key,
                        hash_key,
                        freshness_nonce,
                        shuffle_id,
                        first_real_bin + j_prime + j + c * pow_i,
                        0,
                        &mut 0,
                        //for i == 0, dummy elements are padded by ourselves, not by loading
                        &mut data[c * bin_size..(c + 1) * bin_size - bin_size / 2],
                        &mut Vec::new(),
                        &mut Vec::new(),
                    );
                    pull_bin::<DataSize, MetaSize>(
                        aes_key,
                        hash_key,
                        freshness_nonce,
                        shuffle_id,
                        first_real_bin + j_prime + j + c * pow_i,
                        0,
                        &mut 0,
                        &mut Vec::new(),
                        //for i == 0, dummy elements are padded by ourselves, not by loading
                        &mut meta[c * bin_size..(c + 1) * bin_size - bin_size / 2],
                        &mut Vec::new(),
                    );
                    //assign random keys
                    for key in random_keys[c * bin_size..c * bin_size + bin_size / 2].iter_mut() {
                        *key = (rng.next_u64() & (n as u64 - 1)) + 1;
                    }
                    for key in
                        random_keys[c * bin_size + bin_size / 2..(c + 1) * bin_size].iter_mut()
                    {
                        *key = DUMMY_KEY_LEFT;
                    }
                } else if i == 0 && !has_data {
                    pull_bin::<DataSize, MetaSize>(
                        aes_key,
                        hash_key,
                        freshness_nonce,
                        shuffle_id,
                        first_real_bin + j_prime + j + c * pow_i,
                        1,
                        &mut 0,
                        &mut Vec::new(),
                        //for i == 0, dummy elements are padded by ourselves, not by loading
                        &mut meta[c * bin_size..(c + 1) * bin_size - bin_size / 2],
                        &mut Vec::new(),
                    );
                    //assign random keys
                    for key in random_keys[c * bin_size..c * bin_size + bin_size / 2].iter_mut() {
                        *key = (rng.next_u64() & (n as u64 - 1)) + 1;
                    }
                    for key in
                        random_keys[c * bin_size + bin_size / 2..(c + 1) * bin_size].iter_mut()
                    {
                        *key = DUMMY_KEY_LEFT;
                    }
                } else if i > 0 {
                    pull_bin::<DataSize, MetaSize>(
                        aes_key,
                        hash_key,
                        freshness_nonce,
                        shuffle_id,
                        first_real_bin + j_prime + j + c * pow_i,
                        1,
                        &mut 0,
                        &mut data[c * bin_size * has_data as usize
                            ..(c + 1) * bin_size * has_data as usize],
                        &mut meta[c * bin_size..(c + 1) * bin_size],
                        &mut random_keys[c * bin_size..(c + 1) * bin_size],
                    );
                }
            }
            merge_split::<DataSize, MetaSize>(&mut data, &mut meta, &mut random_keys, msb_mask);
            //for the last i, remove dummy elements and do not store random keys
            if i == log_b - 1 {
                //this step does not need to be oblivious
                let mut search_idx_left = bin_size >> 1;
                let mut search_idx_right = bin_size + (bin_size >> 1);
                while search_idx_left <= bin_size - 1
                    && random_keys[search_idx_left] == DUMMY_KEY_LEFT
                {
                    search_idx_left += 1;
                }
                while search_idx_left != usize::MAX
                    && random_keys[search_idx_left] != DUMMY_KEY_LEFT
                {
                    search_idx_left = search_idx_left.wrapping_sub(1);
                }
                search_idx_left = search_idx_left.wrapping_add(1);
                assert!(search_idx_left < search_idx_right);
                while search_idx_right <= (bin_size << 1) - 1
                    && random_keys[search_idx_right] != DUMMY_KEY_RIGHT
                {
                    search_idx_right += 1;
                }
                if search_idx_right > (bin_size << 1) - 1 {
                    search_idx_right -= 1;
                }
                while search_idx_right != bin_size - 1
                    && random_keys[search_idx_right] == DUMMY_KEY_RIGHT
                {
                    search_idx_right = search_idx_right.wrapping_sub(1);
                }
                search_idx_right = search_idx_right.wrapping_add(1);
                assert!(search_idx_left <= bin_size);
                assert!(search_idx_right >= bin_size);
                for c in [0, 1] {
                    push_bin::<_, DataSize, MetaSize>(
                        aes_key,
                        hash_key,
                        freshness_nonce,
                        shuffle_id,
                        first_real_bin + 2 * j + c,
                        1,
                        &mut data[(c * bin_size + (1 - c) * search_idx_left) * has_data as usize
                            ..((1 - c) * bin_size + c * search_idx_right) * has_data as usize],
                        &mut meta[c * bin_size + (1 - c) * search_idx_left
                            ..(1 - c) * bin_size + c * search_idx_right],
                        &mut Vec::new(),
                        rng,
                    );
                }
                random_keys.clear();
            } else {
                for c in [0, 1] {
                    push_bin::<_, DataSize, MetaSize>(
                        aes_key,
                        hash_key,
                        freshness_nonce,
                        shuffle_id,
                        first_real_bin + 2 * j + c,
                        1,
                        &mut data[c * bin_size * has_data as usize
                            ..(c + 1) * bin_size * has_data as usize],
                        &mut meta[c * bin_size..(c + 1) * bin_size],
                        &mut random_keys[c * bin_size..(c + 1) * bin_size],
                        rng,
                    );
                }
            }
        }
        unsafe {
            bin_switch(shuffle_id, first_real_bin, first_real_bin + num_bins);
        }
        pow_i <<= 1;
    }
}

fn merge_split<DataSize, MetaSize>(
    data: &mut Vec<A64Bytes<DataSize>>,
    meta: &mut Vec<A8Bytes<MetaSize>>,
    random_keys: &mut Vec<u64>,
    msb_mask: u64,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    let n = meta.len();
    let half = n >> 1;
    let mut real_elem_cnt = (0, 0); //left: msb=0, right: msb=1;

    //The acutal key = key - 1 for non-dummy element
    for key in random_keys.iter_mut() {
        real_elem_cnt.0 += (key.saturating_sub(1) & msb_mask == 0 && *key != DUMMY_KEY_LEFT) as u64;
        real_elem_cnt.1 += (key.saturating_sub(1) & msb_mask > 0 && *key != DUMMY_KEY_RIGHT) as u64;
    }

    let mut dummy_elem_cnt = (half as u64 - real_elem_cnt.0, half as u64 - real_elem_cnt.1);

    //change the dummy key
    for key in random_keys.iter_mut() {
        let is_dummy = *key == DUMMY_KEY_LEFT || *key == DUMMY_KEY_RIGHT;
        *key -= *key * is_dummy as u64;
        let dummy_elem_left_cnt = dummy_elem_cnt.0;
        let res = dummy_elem_left_cnt.checked_sub(1);
        dummy_elem_cnt.0 -= (res.is_some() && is_dummy) as u64;
        *key += DUMMY_KEY_RIGHT * (res.is_none() && is_dummy) as u64;
    }

    bitonic_sort(data, meta, random_keys, 0, true);
}

fn oblivious_random_permutation<Rng, DataSize, MetaSize>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    num_bins: usize,
    first_real_bin: usize,
    has_data: bool,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Rng: RngCore + CryptoRng,
{
    let mut acc_data = vec![];
    let mut acc_meta = vec![];
    let mut cur_bin_num_new = 0;
    //since we remove dummy element in bin (not dummy blocks in oram), the size may not be equal
    //to BIN_SIZE_IN_BLOCK/2, but it is expected not exceeding BIN_SIZE_IN_BLOCK
    let expected_bin_size = BIN_SIZE_IN_BLOCK / 2;
    for cur_bin_num in 0..num_bins {
        let mut actual_bin_size = 0;
        let mut data = vec![Default::default(); 2 * expected_bin_size * (has_data as usize)];
        let mut meta = vec![Default::default(); 2 * expected_bin_size];
        pull_bin::<DataSize, MetaSize>(
            aes_key,
            hash_key,
            freshness_nonce,
            shuffle_id,
            first_real_bin + cur_bin_num,
            1,
            &mut actual_bin_size,
            &mut data,
            &mut meta,
            &mut Vec::new(),
        );
        data.truncate(actual_bin_size);
        meta.truncate(actual_bin_size);
        //necessary? how does it influence performance?
        data.shrink_to_fit();
        meta.shrink_to_fit();

        if meta.len() >= expected_bin_size {
            let mut remaining_meta = meta.split_off(expected_bin_size);
            acc_meta.append(&mut remaining_meta);
            if has_data {
                let mut remaining_data = data.split_off(expected_bin_size);
                acc_data.append(&mut remaining_data);
            }
            let mut random_keys = (0..expected_bin_size)
                .map(|_| rng.next_u64())
                .collect::<Vec<_>>();
            bitonic_sort(&mut data, &mut meta, &mut random_keys, 0, true);
            push_bin::<_, DataSize, MetaSize>(
                aes_key,
                hash_key,
                freshness_nonce,
                shuffle_id,
                first_real_bin + cur_bin_num_new,
                1,
                &mut data,
                &mut meta,
                &mut Vec::new(),
                rng,
            );
            cur_bin_num_new += 1;
        } else {
            acc_meta.append(&mut meta);
            //even if data is empty, the following statement is valid
            acc_data.append(&mut data);
        }

        while acc_meta.len() >= expected_bin_size {
            let meta = acc_meta.split_off(expected_bin_size);
            let data = if has_data {
                acc_data.split_off(expected_bin_size)
            } else {
                Vec::new()
            };
            push_bin::<_, DataSize, MetaSize>(
                aes_key,
                hash_key,
                freshness_nonce,
                shuffle_id,
                first_real_bin + cur_bin_num_new,
                1,
                &mut acc_data,
                &mut acc_meta,
                &mut Vec::new(),
                rng,
            );
            acc_meta = meta;
            //even if data is empty, the following statement is valid
            acc_data = data;
            cur_bin_num_new += 1;
        }
    }
    unsafe {
        bin_switch(shuffle_id, first_real_bin, first_real_bin + num_bins);
    }
    assert_eq!(acc_meta.len(), 0);
    assert_eq!(cur_bin_num_new, num_bins);
}

//either sort by random keys or something in meta indexed by key_by
fn bitonic_sort<DataSize, MetaSize>(
    data: &mut [A64Bytes<DataSize>],
    meta: &mut [A8Bytes<MetaSize>],
    random_keys: &mut [u64],
    key_by: usize,
    is_oblivious: bool,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    let n = meta.len();
    assert!(n != 0);
    let mut k = 2;
    while k <= n {
        let mut j = k >> 1;
        while j > 0 {
            for i in 0..n {
                let l = i ^ j;
                if i < l {
                    let condition = if random_keys.is_empty() {
                        !((i & k).ct_eq(&0)
                            ^ (get_key(&meta[i], key_by)).ct_gt(&get_key(&meta[l], key_by)))
                    } else {
                        !((i & k).ct_eq(&0) ^ (random_keys[i]).ct_gt(&random_keys[l]))
                    };
                    if is_oblivious {
                        if !random_keys.is_empty() {
                            let t = random_keys.split_at_mut(i + 1);
                            cswap(condition, &mut t.0[i], &mut t.1[l - i - 1]);
                        }
                        if !data.is_empty() {
                            let t = data.split_at_mut(i + 1);
                            cswap(condition, &mut t.0[i], &mut t.1[l - i - 1]);
                        }
                        let t = meta.split_at_mut(i + 1);
                        cswap(condition, &mut t.0[i], &mut t.1[l - i - 1]);
                    } else {
                        if condition.unwrap_u8() != 0 {
                            if !random_keys.is_empty() {
                                random_keys.swap(i, l);
                            }
                            if !data.is_empty() {
                                data.swap(i, l);
                            }
                            meta.swap(i, l);
                        }
                    }
                }
            }
            j >>= 1;
        }
        k <<= 1;
    }
}

//the additional function is enabled for transformation from new_bucket_idx to new_block_idx
fn non_oblivious_merge_sort<Rng, DataSize, MetaSize>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    num_bins: usize,
    first_real_bin: usize,
    key_by: usize,
    has_data: bool,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Rng: RngCore + CryptoRng,
{
    let bin_size = BIN_SIZE_IN_BLOCK >> 1; //no dummy elements are contained in bin
    let mut width = bin_size;
    let n = num_bins * bin_size;
    assert!(num_bins >= 2);
    while width < n {
        //only the last round enable the additional function
        //and no update for 0..n/2 block meta, for they are related to dummy block
        //TODO: may be the idx transformation from bucket index to block index
        //can be integrited with merge sort.

        // let additional_func = additional_func && width == n >> 1 && first_real_bin != 0;
        // let total_num_bins = first_real_bin << 1;
        // let mut cur_num_bins = first_real_bin;
        // let mut idx_update_end_bin = total_num_bins;
        // while cur_num_bins > num_bins {
        //     cur_num_bins >>= 1;
        //     idx_update_end_bin -= cur_num_bins;
        // }
        // let idx_update_begin_bin = idx_update_end_bin - (cur_num_bins >> 1);

        let mut i = 0;
        let mut cur_bin_num_new = 0;
        while i < n {
            bottom_up_merge::<_, DataSize, MetaSize>(
                aes_key,
                hash_key,
                freshness_nonce,
                shuffle_id,
                first_real_bin,
                &mut cur_bin_num_new,
                key_by,
                i,
                std::cmp::min(i + width, n),
                std::cmp::min(i + 2 * width, n),
                bin_size,
                has_data,
                rng,
            );
            i += 2 * width;
        }
        assert_eq!(cur_bin_num_new, num_bins);
        width <<= 1;
        unsafe {
            bin_switch(shuffle_id, first_real_bin, first_real_bin + num_bins);
        }
    }
}

fn bottom_up_merge<Rng, DataSize, MetaSize>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    first_real_bin: usize,
    cur_bin_num_new: &mut usize,
    key_by: usize,
    i_left: usize,
    i_right: usize,
    i_end: usize,
    bin_size: usize,
    has_data: bool,
    rng: &mut Rng,
) where
    Rng: RngCore + CryptoRng,
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    //In our case, n is the power of 2.
    assert_eq!(i_right - i_left, i_end - i_right);
    let width = i_right - i_left;

    let i_left_bin = i_left / bin_size;
    let i_right_bin = i_right / bin_size;
    let i_end_bin = i_end / bin_size;
    let mut i_bin = i_left_bin;
    let mut j_bin = i_right_bin;

    let mut i = 0;
    let mut j = 0;

    let mut meta = vec![Default::default(); 2 * bin_size];
    let mut data = vec![Default::default(); 2 * bin_size * has_data as usize];
    for (c, &cur_bin_num) in [i_bin, j_bin].iter().enumerate() {
        pull_bin::<DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            first_real_bin + cur_bin_num,
            1,
            &mut 0,
            &mut data[c * bin_size * has_data as usize..(c + 1) * bin_size * has_data as usize],
            &mut meta[c * bin_size..(c + 1) * bin_size],
            &mut Vec::new(),
        );
        //if it is the first round, sort the bin itself first
        if width == bin_size {
            bitonic_sort(
                &mut data[c * bin_size * has_data as usize..(c + 1) * bin_size * has_data as usize],
                &mut meta[c * bin_size..(c + 1) * bin_size],
                &mut Vec::new(),
                key_by,
                false,
            );
        }
    }
    i_bin += 1;
    j_bin += 1;

    let mut sorted_data = Vec::with_capacity(bin_size * (has_data as usize));
    let mut sorted_meta = Vec::with_capacity(bin_size);

    for k in i_left..i_end {
        //when exhaust one bin, pull
        if i >= bin_size && i_bin < i_right_bin {
            i = 0;
            pull_bin::<DataSize, MetaSize>(
                aes_key,
                hash_key,
                &freshness_nonce,
                shuffle_id,
                first_real_bin + i_bin,
                1,
                &mut 0,
                &mut data[..bin_size * has_data as usize],
                &mut meta[..bin_size],
                &mut Vec::new(),
            );
            i_bin += 1;
        }
        if j >= bin_size && j_bin < i_end_bin {
            j = 0;
            pull_bin::<DataSize, MetaSize>(
                aes_key,
                hash_key,
                &freshness_nonce,
                shuffle_id,
                first_real_bin + j_bin,
                1,
                &mut 0,
                &mut data[bin_size * has_data as usize..2 * bin_size * has_data as usize],
                &mut meta[bin_size..2 * bin_size],
                &mut Vec::new(),
            );
            j_bin += 1;
        }
        if i < bin_size
            && (j >= bin_size || get_key(&meta[i], key_by) < get_key(&meta[bin_size + j], key_by))
        {
            sorted_meta.push(meta[i].clone());
            if has_data {
                sorted_data.push(data[i].clone());
            }
            i += 1;
        } else {
            sorted_meta.push(meta[bin_size + j].clone());
            if has_data {
                sorted_data.push(data[bin_size + j].clone());
            }
            j += 1;
        }
        //when form a new bin, push
        if (k + 1) % bin_size == 0 {
            push_bin::<_, DataSize, MetaSize>(
                aes_key,
                hash_key,
                &freshness_nonce,
                shuffle_id,
                first_real_bin + *cur_bin_num_new,
                1,
                &mut sorted_data,
                &mut sorted_meta,
                &mut Vec::new(),
                rng,
            );
            sorted_data.clear();
            sorted_meta.clear();
            *cur_bin_num_new += 1;
        }
    }
    assert!(sorted_data.is_empty());
    assert!(sorted_meta.is_empty());
}

fn oblivious_push_tmp_posmap<Rng, DataSize, MetaSize, Z>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    count: u64,
    num_bins: usize,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned,
    Rng: RngCore + CryptoRng,
{
    //compute how many blocks are needed
    //should be consistent with crate::oram_manager::path_oram
    //note that count is not equal to size, count is the number of buckets in the tree
    //while size is the number of real blocks
    let cur_height = log2_ceil(count) - 1;
    let cur_size = 1 << (cur_height + log2_ceil(Z::U64));
    //should be consistent with crate::oram_manager::position_map
    //note that leaf is stored as u32, not u64, for some reason I don't know yet
    let l = log2_ceil(DataSize::U64) - 2;
    let next_size = cur_size >> l;

    //no dummy elements in bin (but dummy ORAM blocks exist)
    let bin_size = BIN_SIZE_IN_BLOCK >> 1;
    let mut meta = vec![Default::default(); bin_size];

    let mut rec_data = A64Bytes::<DataSize>::default();
    let mut u32_rec_data = rec_data.as_mut_ne_u32_slice();
    let mut cur_pos = 0; //pos in one data item

    //for encryption and authentication
    let mut nonce = GenericArray::<u8, NonceSize>::default();
    rng.fill_bytes(nonce.as_mut_slice());
    let mut cipher = CipherType::new(&POSMAP_KEY, &nonce);
    let mut hasher = Blake2b::new();
    hasher.update(&*POSMAP_KEY);

    //allocate untrusted memory to store the data for next level oram
    //currently we do not consider store it on disk, so it can be allocated this way
    let data_buf;
    let nonce_buf;
    let hash_buf;
    unsafe {
        let mut data_ptr = 0;
        let mut nonce_ptr = 0;
        let mut hash_ptr = 0;

        shuffle_push_tmp_posmap(
            next_size * DataSize::USIZE,
            nonce.len(),
            16,
            &mut data_ptr,
            &mut nonce_ptr,
            &mut hash_ptr,
        );

        //note that the buf is on the untrusted side
        data_buf = (data_ptr as *mut Vec<u8>).as_mut().unwrap();
        nonce_buf = (nonce_ptr as *mut Vec<u8>).as_mut().unwrap();
        hash_buf = (hash_ptr as *mut Vec<u8>).as_mut().unwrap();
    }
    let mut ch = data_buf
        .chunks_exact_mut(DataSize::USIZE)
        .collect::<Vec<_>>();
    let mut cur_idx = 0; //idx for the whole tree

    for cur_bin_num in 0..num_bins {
        pull_bin::<DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            cur_bin_num,
            1,
            &mut 0,
            &mut Vec::new(),
            &mut meta,
            &mut Vec::new(),
        );

        //build data for next level ORAM
        //the second half must be dummy block
        if cur_bin_num < num_bins / 2 {
            for meta_item in meta.iter() {
                if cur_idx >= next_size {
                    break;
                }
                let leaf = meta_item.as_ne_u64_slice()[0] as u32;
                u32_rec_data[cur_pos] = leaf;
                cur_pos += 1;
                if cur_pos >= u32_rec_data.len() {
                    cipher.apply_keystream(&mut rec_data);
                    hasher.update(rec_data.as_ref().deref());
                    ch[cur_idx].copy_from_slice(&rec_data);
                    rec_data = Default::default();
                    u32_rec_data = rec_data.as_mut_ne_u32_slice();
                    cur_pos = 0;
                    cur_idx += 1;
                }
            }
        }

        //reformat again
        reformat_metadata_second(&mut meta);
        //push back for bulk loading
        push_bin::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            cur_bin_num,
            1,
            &mut Vec::new(),
            &mut meta,
            &mut Vec::new(),
            rng,
        );
    }
    //the position map is not full
    while cur_idx < next_size {
        u32_rec_data[cur_pos] = 0;
        cur_pos += 1;
        if cur_pos >= u32_rec_data.len() {
            cipher.apply_keystream(&mut rec_data);
            hasher.update(rec_data.as_ref().deref());
            ch[cur_idx].copy_from_slice(&rec_data);
            rec_data = Default::default();
            u32_rec_data = rec_data.as_mut_ne_u32_slice();
            cur_pos = 0;
            cur_idx += 1;
        }
    }

    let hash: Hash = hasher.finalize()[..16].try_into().unwrap();
    nonce_buf.clone_from_slice(&nonce);
    hash_buf.clone_from_slice(&hash);

    unsafe {
        bin_switch(shuffle_id, 0, num_bins);
    }
}

fn oblivious_pull_tmp_posmap<Rng, DataSize, MetaSize, Z>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    seal_nonce: &GenericArray<u8, NonceSize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    count: u64,
    num_bins: usize,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned,
    Rng: RngCore + CryptoRng,
{
    //no dummy elements in bin (but dummy ORAM blocks exist)
    let bin_size = BIN_SIZE_IN_BLOCK >> 1;
    let mut data: Vec<A64Bytes<DataSize>> = vec![Default::default(); bin_size];
    let mut meta: Vec<A8Bytes<MetaSize>> = vec![Default::default(); bin_size];

    //refer to the untrusted storage
    let data_buf;
    let nonce_buf;
    let hash_buf;
    unsafe {
        let mut data_ptr = 0;
        let mut nonce_ptr = 0;
        let mut hash_ptr = 0;

        shuffle_pull_tmp_posmap(&mut data_ptr, &mut nonce_ptr, &mut hash_ptr);

        //note that the buf is on the untrusted side
        data_buf = (data_ptr as *const Vec<u8>).as_ref().unwrap();
        nonce_buf = (nonce_ptr as *const Vec<u8>).as_ref().unwrap();
        hash_buf = (hash_ptr as *const Vec<u8>).as_ref().unwrap();
    }
    assert_eq!(data_buf.len() % DataSize::USIZE, 0);
    let ch = data_buf.chunks_exact(DataSize::USIZE).collect::<Vec<_>>();
    let mut cur_idx = 0; //idx for the whole tree

    //for decryption and verification
    let mut nonce = GenericArray::<u8, NonceSize>::default();
    nonce.clone_from_slice(nonce_buf);
    let mut cipher = CipherType::new(&POSMAP_KEY, &nonce);
    let mut hasher = Blake2b::new();
    hasher.update(&*POSMAP_KEY);

    for cur_bin_num in 0..num_bins {
        let next_idx = cur_idx + bin_size;
        for idx in cur_idx..std::cmp::min(next_idx, ch.len()) {
            let idx_in_bin = idx % bin_size;
            data[idx_in_bin].copy_from_slice(ch[idx]);
            hasher.update(data[idx_in_bin].as_ref().deref());
            cipher.apply_keystream(&mut data[idx_in_bin]);
            //fill the meta for real block
            meta[idx_in_bin].as_mut_ne_u64_slice()[0] =
                (rng.next_u64() & ((count >> 1) - 1)) + (count >> 1);
            meta[idx_in_bin].as_mut_ne_u64_slice()[1] = idx as u64;
            meta[idx_in_bin].as_mut_ne_u64_slice()[2] = idx as u64;
        }
        for idx in std::cmp::min(next_idx, ch.len())..next_idx {
            let idx_in_bin = idx % bin_size;
            data[idx_in_bin] = Default::default();
            //fill the meta for dummy block
            meta[idx_in_bin].as_mut_ne_u64_slice()[0] = 0;
            meta[idx_in_bin].as_mut_ne_u64_slice()[1] = u64::MAX;
            meta[idx_in_bin].as_mut_ne_u64_slice()[2] = idx as u64;
        }
        cur_idx = next_idx;
        let mut original_meta = meta.clone();

        push_bin::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            &seal_nonce,
            shuffle_id,
            cur_bin_num,
            0,
            &mut data,
            &mut Vec::new(),
            &mut Vec::new(),
            rng,
        );
        push_bin::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            &seal_nonce,
            shuffle_id,
            cur_bin_num,
            0,
            &mut Vec::new(),
            &mut original_meta,
            &mut Vec::new(),
            rng,
        );

        push_bin::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            cur_bin_num,
            1,
            &mut Vec::new(),
            &mut meta,
            &mut Vec::new(),
            rng,
        );
    }

    let hash: Hash = hasher.finalize()[..16].try_into().unwrap();
    assert_eq!(hash_buf, &hash);

    unsafe {
        shuffle_release_tmp_posmap();
        bin_switch(shuffle_id, 0, num_bins);
    }
}

pub fn oblivious_pull_trivial_posmap(data: &mut Vec<u32>) {
    //refer to the untrusted storage
    let data_buf;
    let nonce_buf;
    let hash_buf;
    unsafe {
        let mut data_ptr = 0;
        let mut nonce_ptr = 0;
        let mut hash_ptr = 0;

        shuffle_pull_tmp_posmap(&mut data_ptr, &mut nonce_ptr, &mut hash_ptr);

        //note that the buf is on the untrusted side
        data_buf = (data_ptr as *const Vec<u8>).as_ref().unwrap();
        nonce_buf = (nonce_ptr as *const Vec<u8>).as_ref().unwrap();
        hash_buf = (hash_ptr as *const Vec<u8>).as_ref().unwrap();
    }
    //4 for u32
    let ch = data_buf.chunks_exact(4);
    println!(
        "loading trivial pos map, actual data len = {:?}, expected data len = {:?}, is legal {:?}",
        ch.len(),
        data.len(),
        ch.len() >= data.len(),
    );

    //for decryption and verification
    let mut nonce = GenericArray::<u8, NonceSize>::default();
    nonce.clone_from_slice(nonce_buf);
    let mut cipher = CipherType::new(&POSMAP_KEY, &nonce);
    let mut hasher = Blake2b::new();
    hasher.update(&*POSMAP_KEY);

    for (idx, data_item) in ch.into_iter().enumerate() {
        let mut buf: [u8; 4] = Default::default();
        buf.copy_from_slice(data_item);
        hasher.update(buf);
        cipher.apply_keystream(&mut buf);
        if idx < data.len() {
            data[idx] = u32::from_ne_bytes(buf);
        }
    }

    let hash: Hash = hasher.finalize()[..16].try_into().unwrap();
    assert_eq!(hash_buf, &hash);

    unsafe {
        shuffle_release_tmp_posmap();
    }
}

//the height is actually the standard definition plus 1
fn oblivious_push_location_upwards<Rng, DataSize, MetaSize, Z>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    height: u32,
    num_bins: usize,
    first_real_bin: usize,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned,
    Rng: RngCore + CryptoRng,
{
    let mut occupied_cnt = vec![0 as u32; height as usize];
    let mut last_new_idx = 0;
    //no dummy elements in bin (but dummy ORAM blocks exist)
    let bin_size = BIN_SIZE_IN_BLOCK >> 1;
    let mut meta = vec![Default::default(); bin_size];
    for cur_bin_num in 0..num_bins {
        pull_bin::<DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            first_real_bin + cur_bin_num,
            1,
            &mut 0,
            &mut Vec::new(),
            &mut meta,
            &mut Vec::new(),
        );

        for meta_item in meta.iter_mut() {
            let new_idx = &mut meta_item.as_mut_ne_u64_slice()[2];

            let bit_arr = last_new_idx ^ *new_idx;
            last_new_idx = *new_idx;

            let mut mask = 1 << (height - 1);
            let mut k = 0;
            let mut is_first = true;
            let mut clear_sep = height;
            while k < height {
                let cond = Choice::from(((bit_arr & mask) > 0 && is_first) as u8);
                clear_sep.cmov(cond, &k);
                is_first.cmov(cond, &false);
                mask >>= 1;
                k += 1;
            }

            for j in 0..height {
                let cond = Choice::from((j >= clear_sep) as u8);
                occupied_cnt[j as usize].cmov(cond, &0);
            }

            let mut is_first = true;
            for j in (0..height).rev() {
                let cond = Choice::from((occupied_cnt[j as usize] < Z::U32 && is_first) as u8);
                let next_idx = *new_idx >> (height - 1 - j);
                let next_cnt = occupied_cnt[j as usize] + 1;
                new_idx.cmov(cond, &next_idx);
                occupied_cnt[j as usize].cmov(cond, &next_cnt);
                is_first.cmov(cond, &false);
            }

            //TODO: assign stash addr
        }

        push_bin::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            first_real_bin + cur_bin_num,
            1,
            &mut Vec::new(),
            &mut meta,
            &mut Vec::new(),
            rng,
        );
    }
    unsafe {
        bin_switch(shuffle_id, first_real_bin, first_real_bin + num_bins);
    }
}

fn oblivious_idx_transformation<Rng, DataSize, MetaSize, Z>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    num_bins: usize,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned,
    Rng: RngCore + CryptoRng,
{
    //TODO: note the freshness_nonce issue;
    let bin_size = BIN_SIZE_IN_BLOCK >> 1;
    let mut meta = vec![Default::default(); bin_size];
    let mut last_new_bucket_idx = 0;
    let mut idx_in_block = 0;

    for cur_bin_num in 0..num_bins {
        pull_bin::<DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            cur_bin_num,
            1,
            &mut 0,
            &mut Vec::new(),
            &mut meta,
            &mut Vec::new(),
        );
        if cur_bin_num >= num_bins / 2 {
            for meta_item in meta.iter_mut() {
                let new_idx = &mut meta_item.as_mut_ne_u64_slice()[2];
                let cond_clear = new_idx.ct_gt(&last_new_bucket_idx) | new_idx.ct_eq(&0);
                last_new_bucket_idx = *new_idx;
                //if different value, reset the counter
                idx_in_block.cmov(cond_clear, &0);
                //for dummy block, (new) block idx should be 0
                *new_idx = *new_idx * Z::U64 + idx_in_block;
                idx_in_block += 1;
            }
        }
        push_bin::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            cur_bin_num,
            1,
            &mut Vec::new(),
            &mut meta,
            &mut Vec::new(),
            rng,
        );
    }
    unsafe {
        bin_switch(shuffle_id, 0, num_bins);
    }
}

fn oblivious_placement<Rng, DataSize, MetaSize>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    num_bins: usize,
    key_by: usize,
    invalid_addr: u64,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Rng: RngCore + CryptoRng,
{
    fn core_f<MetaSize: ArrayLength<u8> + PartialDiv<U8>>(
        item_left: &mut A8Bytes<MetaSize>,
        item_right: &mut A8Bytes<MetaSize>,
        sep: u64,
        key_by: usize,
        invalid_addr: u64,
    ) {
        let new_idx_left = get_key(item_left, key_by);
        let new_idx_right = get_key(item_right, key_by);
        let cond = Choice::from(
            (new_idx_left != invalid_addr && new_idx_left >= sep
                || new_idx_right != invalid_addr && new_idx_right < sep) as u8,
        );
        cswap(cond, item_left, item_right);
    }

    fn place_inside_bin<MetaSize: ArrayLength<u8> + PartialDiv<U8>>(
        bin: &mut [A8Bytes<MetaSize>],
        key_by: usize,
        invalid_addr: u64,
    ) {
        let n = bin.len();
        let mut k = n >> 1;
        while k >= 1 {
            let mut base = 0;
            for i in 0..n {
                //if i % (k*2)
                if i & ((k << 1) - 1) == 0 && i != 0 {
                    base += k << 1;
                }
                let j = i ^ k;
                if i < j {
                    let sep = (base + k) as u64;
                    let t = bin.split_at_mut(i + 1);
                    core_f(&mut t.0[i], &mut t.1[j - i - 1], sep, key_by, invalid_addr);
                }
            }
            k >>= 1;
        }
    }
    assert!(invalid_addr == 0 || invalid_addr == u64::MAX);

    let bin_size = BIN_SIZE_IN_BLOCK >> 1;
    if num_bins == 1 {
        let mut meta = vec![Default::default(); bin_size];
        pull_bin::<DataSize, MetaSize>(
            aes_key,
            hash_key,
            freshness_nonce,
            shuffle_id,
            0,
            1,
            &mut 0,
            &mut Vec::new(),
            &mut meta,
            &mut Vec::new(),
        );
        place_inside_bin(&mut meta, key_by, invalid_addr);
        push_bin::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            freshness_nonce,
            shuffle_id,
            0,
            1,
            &mut Vec::new(),
            &mut meta,
            &mut Vec::new(),
            rng,
        );
        unsafe {
            bin_switch(shuffle_id, 0, num_bins);
        }
    } else {
        let log_b = (num_bins as f64).log2() as usize;
        let n = num_bins * bin_size;
        let mut k = n >> 1;
        let mut pow_i = 1 << (log_b - 1);
        for i in (0..log_b).rev() {
            let mut base = 0;
            for j in 0..num_bins / 2 {
                let j_prime = (j >> i) << i;
                //the computation of j_elem is error-prone
                let j_elem = (j_prime + j) * bin_size;
                if j_elem & ((k << 1) - 1) == 0 && j_elem != 0 {
                    base += k << 1;
                }
                let sep = (base + k) as u64;
                let mut meta = vec![Default::default(); 2 * bin_size];
                for c in [0, 1] {
                    pull_bin::<DataSize, MetaSize>(
                        aes_key,
                        hash_key,
                        freshness_nonce,
                        shuffle_id,
                        j_prime + j + c * pow_i,
                        1,
                        &mut 0,
                        &mut Vec::new(),
                        &mut meta[c * bin_size..(c + 1) * bin_size],
                        &mut Vec::new(),
                    );
                }
                {
                    //oblivious placement across bins
                    let split_meta_mut = meta.split_at_mut(bin_size);
                    for (item_left, item_right) in
                        split_meta_mut.0.iter_mut().zip(split_meta_mut.1.iter_mut())
                    {
                        core_f(item_left, item_right, sep, key_by, invalid_addr);
                    }

                    //oblivious placement inside bins
                    if i == 0 {
                        assert_eq!(pow_i, 1);
                        place_inside_bin(split_meta_mut.0, key_by, invalid_addr);
                        place_inside_bin(split_meta_mut.1, key_by, invalid_addr);
                    }
                }

                for c in [0, 1] {
                    push_bin::<_, DataSize, MetaSize>(
                        aes_key,
                        hash_key,
                        freshness_nonce,
                        shuffle_id,
                        j_prime + j + c * pow_i,
                        1,
                        &mut Vec::new(),
                        &mut meta[c * bin_size..(c + 1) * bin_size],
                        &mut Vec::new(),
                        rng,
                    );
                }
            }
            unsafe {
                bin_switch(shuffle_id, 0, num_bins);
            }
            pow_i >>= 1;
            k >>= 1;
        }
    }
}

fn patch_meta<Rng, DataSize, MetaSize>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    seal_nonce: &GenericArray<u8, NonceSize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    num_bins: usize,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Rng: RngCore + CryptoRng,
{
    let bin_size = BIN_SIZE_IN_BLOCK >> 1;
    let mut modified_meta = vec![Default::default(); bin_size];
    let mut original_meta = vec![Default::default(); bin_size];
    for cur_bin_num in 0..num_bins {
        //And after each pull_bin, the untrusted storage should release corresponding space
        pull_bin::<DataSize, MetaSize>(
            aes_key,
            hash_key,
            &seal_nonce,
            shuffle_id,
            cur_bin_num,
            0,
            &mut 0,
            &mut Vec::new(),
            &mut original_meta,
            &mut Vec::new(),
        );
        pull_bin::<DataSize, MetaSize>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            cur_bin_num,
            1,
            &mut 0,
            &mut Vec::new(),
            &mut modified_meta,
            &mut Vec::new(),
        );

        for i in 0..bin_size {
            //update leaf
            (&mut original_meta[i][0..8]).copy_from_slice(&modified_meta[i][0..8]);
            //reuse the space of counter for convenience
            (&mut original_meta[i][16..24]).copy_from_slice(&modified_meta[i][16..24]);
        }
        //for this push_bin, bin_type = 0, because it is consistent with separated data
        push_bin::<_, DataSize, MetaSize>(
            aes_key,
            hash_key,
            &seal_nonce,
            shuffle_id,
            cur_bin_num,
            0,
            &mut Vec::new(),
            &mut original_meta,
            &mut Vec::new(),
            rng,
        );
    }
    unsafe {
        bin_switch(shuffle_id, 0, num_bins);
    }
}

fn shuffle_core<Rng, DataSize, MetaSize, Z>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    seal_nonce: &GenericArray<u8, NonceSize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    count: u64,
    num_bins: usize,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned,
    Rng: RngCore + CryptoRng,
{
    let now = Instant::now();
    oblivious_push_tmp_posmap::<_, DataSize, MetaSize, Z>(
        aes_key,
        hash_key,
        &freshness_nonce,
        shuffle_id,
        count,
        num_bins,
        rng,
    );
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("finish oblivious push tmp posmap, {:?}s", dur);

    let now = Instant::now();
    bucket_oblivious_sort::<_, DataSize, MetaSize, Z>(
        aes_key,
        hash_key,
        &freshness_nonce,
        shuffle_id,
        num_bins,
        0,
        2,
        false,
        rng,
    );
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("finish bucket oblivious sort by new idx, {:?}s", dur);

    let now = Instant::now();
    //if a bucket holds blocks more than its capacity, move excess blocks from leaf to root
    oblivious_push_location_upwards::<_, DataSize, MetaSize, Z>(
        aes_key,
        hash_key,
        &freshness_nonce,
        shuffle_id,
        log2_ceil(count),
        num_bins,
        0,
        rng,
    );
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("finish push location upwards, {:?}s", dur);

    let now = Instant::now();
    bucket_oblivious_sort::<_, DataSize, MetaSize, Z>(
        aes_key,
        hash_key,
        &freshness_nonce,
        shuffle_id,
        num_bins,
        0,
        2,
        false,
        rng,
    );
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("finish bucket oblivious sort by new idx, {:?}s", dur);

    let now = Instant::now();
    //change the new idx in bucket to new idx in block
    //TODO: if freshness nonce is changed per round, check it
    oblivious_idx_transformation::<_, DataSize, MetaSize, Z>(
        aes_key,
        hash_key,
        &freshness_nonce,
        shuffle_id,
        num_bins,
        rng,
    );
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("finish idx transformation, {:?}s", dur);

    let now = Instant::now();
    //oblivious placement according to new idx, and assign idx of vacant place to dummy blocks
    oblivious_placement::<_, DataSize, MetaSize>(
        aes_key,
        hash_key,
        &freshness_nonce,
        shuffle_id,
        num_bins,
        2,
        0,
        rng,
    );
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("finish oblivious placement by new idx, {:?}s", dur);

    let now = Instant::now();
    //bucket oblivious sort info by old idx
    bucket_oblivious_sort::<_, DataSize, MetaSize, Z>(
        aes_key,
        hash_key,
        &freshness_nonce,
        shuffle_id,
        num_bins,
        0,
        1,
        false,
        rng,
    );
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("finish bucket oblivious sort by old idx, {:?}s", dur);

    let now = Instant::now();
    //patch the sorted info to original blocks
    patch_meta::<_, DataSize, MetaSize>(
        aes_key,
        hash_key,
        &seal_nonce,
        &freshness_nonce,
        shuffle_id,
        num_bins,
        rng,
    );
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("finish patch meta, {:?}s", dur);

    let now = Instant::now();
    //bucket oblivious sort blocks (combined with meta) by new idx
    //TODO: maybe both seal_nonce and freshness nonce are needed
    //because loading the seperate data and meta bins needs seal_nonce
    //while loading the intermediate bins needs freshness_nonce
    bucket_oblivious_sort::<_, DataSize, MetaSize, Z>(
        aes_key,
        hash_key,
        &seal_nonce,
        shuffle_id,
        num_bins,
        0,
        2,
        true,
        rng,
    );
    unsafe {
        clear_content(shuffle_id);
    }
    let dur = now.elapsed().as_nanos() as f64 * 1e-9;
    println!("finish bucket oblivious sort data and meta, {:?}s", dur);
}

fn get_key<MetaSize: ArrayLength<u8> + PartialDiv<U8>>(
    meta: &A8Bytes<MetaSize>,
    key_by: usize,
) -> u64 {
    meta.as_ne_u64_slice()[key_by]
}

mod helpers {
    use super::*;

    // Helper for invoking the pull ORAM buckets OCALL safely
    // Combined means data and meta are encrypted and authenticated together
    // Merkle tree is involved and data are encrypted per bucket
    pub fn shuffle_pull_buckets_ocall<
        DataSize: ArrayLength<u8> + PartialDiv<U8>,
        MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    >(
        shuffle_id: u64,
        b_idx: usize,
        e_idx: usize,
        data: &mut [A64Bytes<DataSize>],
        meta: &mut [A8Bytes<MetaSize>],
    ) {
        debug_assert!(e_idx - b_idx == data.len());
        debug_assert!(e_idx - b_idx == meta.len());
        unsafe {
            super::shuffle_pull_buckets(
                shuffle_id,
                b_idx,
                e_idx,
                data.as_mut_ptr() as *mut u8,
                data.len() * DataSize::USIZE,
                meta.as_mut_ptr() as *mut u8,
                meta.len() * MetaSize::USIZE,
            )
        }
    }

    // Helper for invoking the pull shuffle bin OCALL safely
    pub fn shuffle_pull_bin_ocall<DataSize, MetaSize>(
        shuffle_id: u64,
        cur_bin_num: usize,
        bin_type: u8,
        bin_size: &mut usize,
        data: &mut [A64Bytes<DataSize>],
        meta: &mut [A8Bytes<MetaSize>],
        random_keys: &mut [u64],
        nonce: &mut GenericArray<u8, NonceSize>,
        hash: &mut Hash,
    ) where
        DataSize: ArrayLength<u8> + PartialDiv<U8>,
        MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    {
        use std::cmp::min;
        let mut data_ptr = 0;
        let mut meta_ptr = 0;
        let mut random_key_ptr = 0;
        let mut nonce_ptr = 0;
        let mut hash_ptr = 0;
        unsafe {
            super::shuffle_pull_bin(
                shuffle_id,
                cur_bin_num,
                bin_type,
                bin_size,
                DataSize::USIZE,
                MetaSize::USIZE,
                (data.len() > 0) as u8,
                (meta.len() > 0) as u8,
                (random_keys.len() > 0) as u8,
                nonce.len(),
                hash.len(),
                &mut data_ptr,
                &mut meta_ptr,
                &mut random_key_ptr,
                &mut nonce_ptr,
                &mut hash_ptr,
            );

            let src_data_buf = (data_ptr as *mut Vec<u8>).as_ref().unwrap();
            let src_meta_buf = (meta_ptr as *mut Vec<u8>).as_ref().unwrap();
            let src_random_key_buf = (random_key_ptr as *mut Vec<u8>).as_ref().unwrap();
            let src_nonce_buf = (nonce_ptr as *mut Vec<u8>).as_ref().unwrap();
            let src_hash_buf = (hash_ptr as *mut Vec<u8>).as_ref().unwrap();

            let data_size = min(data.len(), *bin_size) * DataSize::USIZE;
            let meta_size = min(meta.len(), *bin_size) * MetaSize::USIZE;
            let random_key_size = min(random_keys.len(), *bin_size) * 8;

            core::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data_size)
                .copy_from_slice(src_data_buf);
            core::slice::from_raw_parts_mut(meta.as_mut_ptr() as *mut u8, meta_size)
                .copy_from_slice(src_meta_buf);
            core::slice::from_raw_parts_mut(random_keys.as_mut_ptr() as *mut u8, random_key_size)
                .copy_from_slice(src_random_key_buf);
            nonce.copy_from_slice(src_nonce_buf);
            hash.copy_from_slice(src_hash_buf);
        }
    }

    // Helper for invoking the push ORAM buckets OCALL safely
    pub fn shuffle_push_buckets_ocall<
        DataSize: ArrayLength<u8> + PartialDiv<U8>,
        MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    >(
        shuffle_id: u64,
        b_idx: usize,
        e_idx: usize,
        data: &[A64Bytes<DataSize>],
        meta: &[A8Bytes<MetaSize>],
    ) {
        debug_assert!(e_idx - b_idx == data.len());
        debug_assert!(e_idx - b_idx == meta.len());
        let mut data_ptr = 0;
        let mut meta_ptr = 0;
        let data_size = data.len() * DataSize::USIZE;
        let meta_size = meta.len() * MetaSize::USIZE;

        unsafe {
            super::shuffle_push_buckets_pre(
                shuffle_id,
                data_size,
                meta_size,
                &mut data_ptr,
                &mut meta_ptr,
            );

            let dst_data_buf = (data_ptr as *mut Vec<u8>).as_mut().unwrap();
            let dst_meta_buf = (meta_ptr as *mut Vec<u8>).as_mut().unwrap();
            dst_data_buf.copy_from_slice(core::slice::from_raw_parts(
                data.as_ptr() as *mut u8,
                data_size,
            ));
            dst_meta_buf.copy_from_slice(core::slice::from_raw_parts(
                meta.as_ptr() as *mut u8,
                meta_size,
            ));

            super::shuffle_push_buckets(shuffle_id, b_idx, e_idx);
        }
    }

    // Helper for invoking the push shuffle bin OCALL safely
    pub fn shuffle_push_bin_ocall<DataSize, MetaSize>(
        shuffle_id: u64,
        cur_bin_num: usize,
        bin_type: u8,
        data: &[A64Bytes<DataSize>],
        meta: &[A8Bytes<MetaSize>],
        random_keys: &[u64],
        nonce: &GenericArray<u8, NonceSize>,
        hash: &Hash,
    ) where
        DataSize: ArrayLength<u8> + PartialDiv<U8>,
        MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    {
        let mut data_ptr = 0;
        let mut meta_ptr = 0;
        let mut random_key_ptr = 0;
        let mut nonce_ptr = 0;
        let mut hash_ptr = 0;
        let data_size = data.len() * DataSize::USIZE;
        let meta_size = meta.len() * MetaSize::USIZE;
        let random_key_size = random_keys.len() * 8;
        let nonce_size = nonce.len();
        let hash_size = hash.len();
        unsafe {
            super::shuffle_push_bin_pre(
                shuffle_id,
                data_size,
                meta_size,
                random_key_size,
                nonce_size,
                hash_size,
                &mut data_ptr,
                &mut meta_ptr,
                &mut random_key_ptr,
                &mut nonce_ptr,
                &mut hash_ptr,
            );

            let dst_data_buf = (data_ptr as *mut Vec<u8>).as_mut().unwrap();
            let dst_meta_buf = (meta_ptr as *mut Vec<u8>).as_mut().unwrap();
            let dst_random_key_buf = (random_key_ptr as *mut Vec<u8>).as_mut().unwrap();
            let dst_nonce_buf = (nonce_ptr as *mut Vec<u8>).as_mut().unwrap();
            let dst_hash_buf = (hash_ptr as *mut Vec<u8>).as_mut().unwrap();
            dst_data_buf.copy_from_slice(core::slice::from_raw_parts(
                data.as_ptr() as *mut u8,
                data_size,
            ));
            dst_meta_buf.copy_from_slice(core::slice::from_raw_parts(
                meta.as_ptr() as *mut u8,
                meta_size,
            ));
            dst_random_key_buf.copy_from_slice(core::slice::from_raw_parts(
                random_keys.as_ptr() as *mut u8,
                random_key_size,
            ));
            dst_nonce_buf.copy_from_slice(nonce);
            dst_hash_buf.copy_from_slice(hash);

            super::shuffle_push_bin(shuffle_id, cur_bin_num, bin_type);
        }
    }
}

extern "C" {
    fn shuffle_pull_buckets(
        shuffle_id: u64,
        b_idx: usize,
        e_idx: usize,
        data: *mut u8,
        data_size: usize,
        meta: *mut u8,
        meta_size: usize,
    );
    //since only shuffle_pull_bin occurs unexpected bugs
    //i.e., the ocall is not called acutally, so we don't
    //use the stash
    fn shuffle_pull_bin(
        shuffle_id: u64,
        cur_bin_num: usize,
        bin_type: u8,
        bin_size: *mut usize,
        data_item_size: usize,
        meta_item_size: usize,
        has_data: u8,
        has_meta: u8,
        has_random_key: u8,
        nonce_size: usize,
        hash_size: usize,
        data_ptr: *mut usize,
        meta_ptr: *mut usize,
        random_key_ptr: *mut usize,
        nonce_ptr: *mut usize,
        hash_ptr: *mut usize,
    );
    fn shuffle_push_buckets_pre(
        shuffle_id: u64,
        data_size: usize,
        meta_size: usize,
        data_ptr: *mut usize,
        meta_ptr: *mut usize,
    );
    fn shuffle_push_buckets(shuffle_id: u64, b_idx: usize, e_idx: usize);
    //since only shuffle_push_bin occurs unexpected bugs
    //i.e., the ocall is not called acutally, so we split
    //the function into two parts
    //allocate buffer
    fn shuffle_push_bin_pre(
        shuffle_id: u64,
        data_size: usize,
        meta_size: usize,
        random_key_size: usize,
        nonce_size: usize,
        hash_size: usize,
        data_ptr: *mut usize,
        meta_ptr: *mut usize,
        random_key_ptr: *mut usize,
        nonce_ptr: *mut usize,
        hash_ptr: *mut usize,
    );
    fn shuffle_push_bin(shuffle_id: u64, cur_bin_num: usize, bin_type: u8);
    fn shuffle_push_tmp_posmap(
        data_size: usize,
        nonce_size: usize,
        hash_size: usize,
        data_ptr: *mut usize,
        nonce_ptr: *mut usize,
        hash_ptr: *mut usize,
    );
    fn shuffle_pull_tmp_posmap(data_ptr: *mut usize, nonce_ptr: *mut usize, hash_ptr: *mut usize);
    fn shuffle_release_tmp_posmap();

    //move the dst bin to src bin, and assert that the src bin is empty now
    fn bin_switch(shuffle_id: u64, begin_bin_idx: usize, end_bin_idx: usize);
    //clear all unneccessary content
    fn clear_content(shuffle_id: u64);

    //the sequential representation of tree is built, we just need to tranfer
    //the ownership to oram manager
    fn build_oram_from_shuffle_manager(shuffle_id: u64, allocation_id: u64);
}
