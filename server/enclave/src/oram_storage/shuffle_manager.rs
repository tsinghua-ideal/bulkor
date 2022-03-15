use crate::oram_storage::{compute_block_hash, make_aes_nonce, ExtraMeta, ExtraMetaSize, Hash};
use crate::oram_traits::HeapORAMStorage;
use crate::{AuthCipherType, AuthNonceSize, CipherType, KeySize, NonceSize, ALLOCATOR};
use aes::cipher::{NewCipher, StreamCipher};
use aligned_cmov::{
    cswap,
    subtle::{Choice, ConstantTimeEq, ConstantTimeGreater, ConstantTimeLess},
    typenum::{PartialDiv, PowerOfTwo, Prod, Quot, Sum, Unsigned, U8},
    A64Bytes, A8Bytes, Aligned, ArrayLength, AsAlignedChunks, AsNeSlice, CMov, GenericArray, A8,
};
use blake2::{digest::Digest, Blake2b};
use rand_core::{CryptoRng, RngCore};
use sgx_trts::trts::rsgx_read_rand;
use std::boxed::Box;
use std::convert::TryInto;
use std::ops::{Add, Deref, Div};
use std::vec::Vec;

//The parameter in bucket oblivious sort
//For an overflow probability of 2^-80 and most reasonable values of n, Z = 512 suffices.
pub const BIN_SIZE_IN_BLOCK: usize = 512;
//assume dummy element random_key=0 or u64::MAX
const DUMMY_KEY_LEFT: u64 = 0;
const DUMMY_KEY_RIGHT: u64 = u64::MAX;

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
//after this process, the format of meta: (leaf num, old idx, new idx);
//the old idx is block idx, while the new idx is first set to bucket idx
//for dummy block, both new leaf and new idx = 0
//Note: the meta_item aggregate metadata from z blocks
fn reformat_metadata<MetaSize, Z>(
    aes_key: &GenericArray<u8, KeySize>,
    shuffle_nonce: &GenericArray<u8, NonceSize>,
    count: u64,
    b_idx: usize,
    meta: &mut Vec<A8Bytes<MetaSize>>,
) where
    MetaSize: ArrayLength<u8> + Add<ExtraMetaSize> + Div<Z>,
    Z: Unsigned,
    Quot<MetaSize, Z>: ArrayLength<u8> + PartialDiv<U8>,
{
    for (i, meta_items) in meta.iter_mut().enumerate() {
        let meta_items: &mut [A8Bytes<Quot<MetaSize, Z>>] = meta_items.as_mut_aligned_chunks();
        for (idx_in_block, meta_item) in meta_items.iter_mut().enumerate() {
            //get old leaf num
            let old_leaf = meta_item.as_ne_u64_slice()[0];
            let is_vacant = old_leaf.ct_eq(&0);
            //test whether a metadata is vacant, i.e., refer to dummy block
            //assign new leaf num
            let mut new_leaf_buf: [u8; 8] = (&meta_item[8..16]).try_into().unwrap();
            let mut cipher = CipherType::new(aes_key, shuffle_nonce);
            cipher.apply_keystream(&mut new_leaf_buf);
            let mut new_leaf =
                (u64::from_le_bytes(new_leaf_buf) & ((count >> 1) - 1)) + (count >> 1);
            new_leaf.cmov(is_vacant, &0);
            new_leaf_buf = new_leaf.to_le_bytes();
            (&mut meta_item[0..8]).copy_from_slice(&new_leaf_buf);
            //assign old block idx
            let old_idx = (b_idx + i) * Z::USIZE + idx_in_block;
            (&mut meta_item[8..16]).copy_from_slice(&old_idx.to_le_bytes());
            //assign new bucket idx
            let new_idx = new_leaf;
            (&mut meta_item[16..24]).copy_from_slice(&new_idx.to_le_bytes());
        }
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
    hasher.update(cur_bin_num.to_le_bytes());

    for item in data {
        cipher.apply_keystream(item);
        hasher.update(item.as_ref().deref());
    }
    for item in meta {
        cipher.apply_keystream(item);
        hasher.update(item.as_ref().deref());
    }
    for item in random_keys {
        let mut item_buf = item.to_le_bytes();
        cipher.apply_keystream(&mut item_buf);
        hasher.update(item_buf);
        *item = u64::from_le_bytes(item_buf);
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
    hasher.update(cur_bin_num.to_le_bytes());

    for item in data {
        hasher.update(item.as_ref().deref());
        cipher.apply_keystream(item);
    }
    for item in meta {
        hasher.update(item.as_ref().deref());
        cipher.apply_keystream(item);
    }
    for item in random_keys {
        let mut item_buf = item.to_le_bytes();
        hasher.update(item_buf);
        cipher.apply_keystream(&mut item_buf);
        *item = u64::from_le_bytes(item_buf);
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

pub fn manage<DataSize, MetaSize, Z, Rng>(
    shuffle_id: u64,
    allocation_id: u64,
    count: u64,
    treetop_max_count: usize,
    treetop: &mut HeapORAMStorage<DataSize, MetaSize, Z>,
    trusted_merkle_roots: &mut Vec<Hash>,
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    shuffle_nonce: &mut GenericArray<u8, NonceSize>,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PowerOfTwo + PartialDiv<U8> + Div<Z>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8> + Add<ExtraMetaSize> + Div<Z>,
    Z: Unsigned,
    Rng: RngCore + CryptoRng,
    Sum<MetaSize, ExtraMetaSize>: ArrayLength<u8> + PartialDiv<U8>,
    Quot<DataSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
    Quot<MetaSize, Z>: ArrayLength<u8> + PartialDiv<U8> + Unsigned,
{
    //seal nonce is needed to store original data and metadata
    let mut seal_nonce = GenericArray::<u8, NonceSize>::default();
    //freshness nonce should change every time a round in bucket oblivious sort is finished
    let mut freshness_nonce = GenericArray::<u8, NonceSize>::default();
    //freshness nonce used for aes gcm
    let mut freshness_auth_nonce = GenericArray::<u8, AuthNonceSize>::default();

    rsgx_read_rand(shuffle_nonce).unwrap();
    rsgx_read_rand(&mut seal_nonce).unwrap();
    rsgx_read_rand(&mut freshness_nonce).unwrap();
    rsgx_read_rand(&mut freshness_auth_nonce).unwrap();

    //read and verify buckets and extract info from metadata
    //specifically, (old leaf num, block num, counter), we use bin
    //and bucket for buckets in bucket oblivious sort and oram, respectively
    let bin_size_in_bucket = BIN_SIZE_IN_BLOCK / Z::USIZE;
    let bin_size_in_bucket_real = bin_size_in_bucket >> 1;
    let num_bins = count as usize / bin_size_in_bucket_real;
    assert!(num_bins != 0);
    assert!(
        num_bins & (num_bins - 1) == 0,
        "num_bins must be a power of two"
    );
    println!("begin pull all oram buckets");
    //reorganize buckets and prepare bins for subsequent shuffle
    {
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
                let mut data: Vec<A64Bytes<DataSize>> =
                    vec![Default::default(); bin_size_in_bucket];
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
                                    let parent_extra_meta =
                                        ExtraMeta::from(&*parent_extra_meta_mut);
                                    if idx & 1 == 0 {
                                        assert_eq!(
                                            this_block_hash,
                                            parent_extra_meta.left_child_hash
                                        );
                                    } else {
                                        assert_eq!(
                                            this_block_hash,
                                            parent_extra_meta.right_child_hash
                                        );
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

                //compute new idx from block num
                reformat_metadata(aes_key, shuffle_nonce, count, b_idx, &mut meta);
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

                b_count = 0;
            } else {
                let mut cur_tier_hasher = Blake2b::new();
                let mut prev_tier_hasher = Blake2b::new();
                cur_tier_hasher.update(hash_key);
                prev_tier_hasher.update(hash_key);
                while e_idx > b_count {
                    let mut data: Vec<A64Bytes<DataSize>> =
                        vec![Default::default(); bin_size_in_bucket];
                    let mut meta: Vec<A8Bytes<MetaSize>> =
                        vec![Default::default(); bin_size_in_bucket];
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
                                if b_idx >= 2 * treetop_max_count
                                    && this_block_hash != Hash::default()
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
                        (&mut meta[..delta])
                            .clone_from_slice(&treetop.metadata[b_idx..b_idx + delta]);
                    }

                    //write data and meta back separately with encryption and authentication
                    //in the untrusted domain, data may saved on disk, while meta almost in memory for further process
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

                    //compute new idx from block num
                    reformat_metadata(aes_key, shuffle_nonce, count, b_idx, &mut meta);
                    //although the DataSize, MetaSize is not consistent with subsequent
                    //pull_bin and push_bin in bucket oblivious sort, it does not matter
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
    }
    println!("finish pull all oram buckets");

    //whether there is a modification to new idx
    //if so, do bucket oblivious sort (new leaf num, old idx, new idx) by new idx
    //actually, for obliviousness, a bucket oblivious sort should be performed log(n) times
    let mut cur_num_bins = num_bins;
    let mut first_real_bin = 0;
    //note that there are at least num_bins * (BIN_SIZE_IN_BLOCK / 2) / 2 dummy blocks
    //so for the rounds except the first round, the following function will skip processing
    //The first num_bins / 2 bins
    while cur_num_bins >= 2 {
        bucket_oblivious_sort::<_, Quot<DataSize, Z>, Quot<MetaSize, Z>, Z>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            cur_num_bins,
            first_real_bin,
            2,
            false,
            rng,
        );
        cur_num_bins >>= 1;
        if first_real_bin == 0 {
            first_real_bin = num_bins / 2;
        }
        //if a bucket holds blocks more than its capacity, move excess blocks from leaf to root
        oblivious_push_location_upwards::<_, Quot<DataSize, Z>, Quot<MetaSize, Z>, Z>(
            aes_key,
            hash_key,
            &freshness_nonce,
            shuffle_id,
            cur_num_bins,
            first_real_bin,
            rng,
        );
    }
    //change the new idx in bucket to new idx in block
    //TODO: if freshness nonce is changed per round, check it
    oblivious_idx_transformation::<_, Quot<DataSize, Z>, Quot<MetaSize, Z>, Z>(
        aes_key,
        hash_key,
        &freshness_nonce,
        shuffle_id,
        num_bins,
        rng,
    );

    //oblivious placement according to new idx, and assign idx of vacant place to dummy blocks
    oblivious_placement::<_, Quot<DataSize, Z>, Quot<MetaSize, Z>>(
        aes_key,
        hash_key,
        &freshness_nonce,
        shuffle_id,
        num_bins,
        rng,
    );

    //bucket oblivious sort info by old idx
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

    //patch the sorted info to original blocks
    patch_meta::<_, Quot<DataSize, Z>, Quot<MetaSize, Z>>(
        aes_key,
        hash_key,
        &seal_nonce,
        &freshness_nonce,
        shuffle_id,
        num_bins,
        rng,
    );

    //bucket oblivious sort blocks (combined with meta) by new idx
    //TODO: maybe both seal_nonce and freshness nonce are needed
    //because loading the seperate data and meta bins needs seal_nonce
    //while loading the intermediate bins needs freshness_nonce
    println!("start bucket oblivious sort data and meta");
    bucket_oblivious_sort::<_, Quot<DataSize, Z>, Quot<MetaSize, Z>, Z>(
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
    println!("finish bucket oblivious sort data and meta");

    //authenticate blocks as in oram tree. During the procedure,
    //clear the new idx, that is, restore the function of counter and reset it to 0
    {
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
                assert_eq!(e_idx, bin_size_in_bucket);
                assert_eq!(cur_bin_num, 0);
                let mut actual_bin_size = 0;
                let mut data = vec![Default::default(); bin_size_in_bucket];
                let mut meta = vec![Default::default(); bin_size_in_bucket];
                pull_bin::<DataSize, MetaSize>(
                    aes_key,
                    hash_key,
                    &seal_nonce,
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
                let mut meta_plus_extra: Vec<A8Bytes<Sum<MetaSize, ExtraMetaSize>>> =
                    vec![Default::default(); bin_size_in_bucket];
                let b_idx = e_idx - bin_size_in_bucket;
                let delta = bin_size_in_bucket.saturating_sub(treetop_max_count);
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

                (&mut treetop.data[b_idx..e_idx - delta])
                    .clone_from_slice(&data[..bin_size_in_bucket - delta]);
                (&mut treetop.metadata[b_idx..e_idx - delta])
                    .clone_from_slice(&meta[..bin_size_in_bucket - delta]);
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
                    let mut actual_bin_size = 0;
                    let mut data = vec![Default::default(); bin_size_in_bucket];
                    let mut meta = vec![Default::default(); bin_size_in_bucket];
                    pull_bin::<DataSize, MetaSize>(
                        aes_key,
                        hash_key,
                        &seal_nonce,
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
                        let items: &mut [A8Bytes<Quot<MetaSize, Z>>] =
                            items_.as_mut_aligned_chunks();
                        for item in items {
                            (&mut item[16..24]).copy_from_slice(&[0; 8]);
                        }
                    }
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

    //build
    //handle the processed array to oram tree
    unsafe {
        build_oram_from_shuffle_manager(
            shuffle_id,
            allocation_id,
            shuffle_nonce.as_ptr(),
            shuffle_nonce.len(),
        );
    }
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

fn oblivious_push_location_upwards<Rng, DataSize, MetaSize, Z>(
    aes_key: &GenericArray<u8, KeySize>,
    hash_key: &GenericArray<u8, KeySize>,
    freshness_nonce: &GenericArray<u8, NonceSize>,
    shuffle_id: u64,
    num_bins: usize,
    first_real_bin: usize,
    rng: &mut Rng,
) where
    DataSize: ArrayLength<u8> + PartialDiv<U8>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned,
    Rng: RngCore + CryptoRng,
{
    fn core_f<MetaSize: ArrayLength<u8> + PartialDiv<U8>, Z: Unsigned>(
        last_new_idx: &mut u64,
        continuous_cnt: &mut u64,
        meta: &mut [A8Bytes<MetaSize>],
    ) {
        for meta_item in meta {
            let new_idx = &mut meta_item.as_mut_ne_u64_slice()[2];
            let cond_clear_cnt = new_idx.ct_gt(last_new_idx);
            *last_new_idx = *new_idx;
            //if different value, reset the counter
            continuous_cnt.cmov(cond_clear_cnt, &0);
            *continuous_cnt += 1;
            let cond_modify_idx = continuous_cnt.ct_gt(&Z::U64);
            //assume there are more blocks exceeding the capacity of a bucket
            new_idx.cmov(cond_modify_idx, &(*new_idx >> 1));
        }
    }
    //no dummy elements in bin (but dummy ORAM blocks exist)
    let bin_size = BIN_SIZE_IN_BLOCK >> 1;
    let mut meta = vec![Default::default(); bin_size];
    let mut last_new_idx = 0;
    let mut continuous_cnt = 0;
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
        core_f::<_, Z>(&mut last_new_idx, &mut continuous_cnt, &mut meta);
        //push location inside a bin
        if num_bins == 1 {
            let mut boundary = bin_size;
            while boundary >= Z::USIZE * 2 {
                last_new_idx = 0;
                continuous_cnt = 0;
                bitonic_sort::<DataSize, MetaSize>(
                    &mut Vec::new(),
                    &mut meta[0..boundary],
                    &mut Vec::new(),
                    2,
                    true,
                );
                boundary >>= 1;
                core_f::<_, Z>(
                    &mut last_new_idx,
                    &mut continuous_cnt,
                    &mut meta[0..boundary],
                );
            }
            //TODO: check abortion case
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

    for cur_bin_num in num_bins / 2..num_bins {
        let mut last_new_bucket_idx = 0;
        let mut idx_in_block = 0;
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
    ) {
        let new_idx_left = get_key(item_left, 2);
        let new_idx_right = get_key(item_right, 2);
        let cond = Choice::from(
            (new_idx_left != 0 && new_idx_left >= sep || new_idx_right != 0 && new_idx_right < sep)
                as u8,
        );
        cswap(cond, item_left, item_right);
    }

    fn place_inside_bin<MetaSize: ArrayLength<u8> + PartialDiv<U8>>(bin: &mut [A8Bytes<MetaSize>]) {
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
                    core_f(&mut t.0[i], &mut t.1[j - i - 1], sep);
                }
            }
            k >>= 1;
        }
    }

    let log_b = (num_bins as f64).log2() as usize;
    let bin_size = BIN_SIZE_IN_BLOCK >> 1;
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
                    core_f(item_left, item_right, sep);
                }

                //oblivious placement inside bins
                if i == 0 {
                    assert_eq!(pow_i, 1);
                    place_inside_bin(split_meta_mut.0);
                    place_inside_bin(split_meta_mut.1);
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

            let src_data_buf = Box::from_raw(data_ptr as *mut Vec<u8>);
            let src_meta_buf = Box::from_raw(meta_ptr as *mut Vec<u8>);
            let src_random_key_buf = Box::from_raw(random_key_ptr as *mut Vec<u8>);
            let src_nonce_buf = Box::from_raw(nonce_ptr as *mut Vec<u8>);
            let src_hash_buf = Box::from_raw(hash_ptr as *mut Vec<u8>);

            let data_size = min(data.len(), *bin_size) * DataSize::USIZE;
            let meta_size = min(meta.len(), *bin_size) * MetaSize::USIZE;
            let random_key_size = min(random_keys.len(), *bin_size) * 8;

            core::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data_size)
                .copy_from_slice(&src_data_buf);
            core::slice::from_raw_parts_mut(meta.as_mut_ptr() as *mut u8, meta_size)
                .copy_from_slice(&src_meta_buf);
            core::slice::from_raw_parts_mut(random_keys.as_mut_ptr() as *mut u8, random_key_size)
                .copy_from_slice(&src_random_key_buf);
            nonce.copy_from_slice(&src_nonce_buf);
            hash.copy_from_slice(&src_hash_buf);

            Box::into_raw(src_data_buf);
            Box::into_raw(src_meta_buf);
            Box::into_raw(src_random_key_buf);
            Box::into_raw(src_nonce_buf);
            Box::into_raw(src_hash_buf);

            super::shuffle_pull_bin_post(data_ptr, meta_ptr, random_key_ptr, nonce_ptr, hash_ptr);
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
        unsafe {
            super::shuffle_push_buckets(
                shuffle_id,
                b_idx,
                e_idx,
                data.as_ptr() as *const u8,
                data.len() * DataSize::USIZE,
                meta.as_ptr() as *const u8,
                meta.len() * MetaSize::USIZE,
            )
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

            let mut dst_data_buf = Box::from_raw(data_ptr as *mut Vec<u8>);
            let mut dst_meta_buf = Box::from_raw(meta_ptr as *mut Vec<u8>);
            let mut dst_random_key_buf = Box::from_raw(random_key_ptr as *mut Vec<u8>);
            let mut dst_nonce_buf = Box::from_raw(nonce_ptr as *mut Vec<u8>);
            let mut dst_hash_buf = Box::from_raw(hash_ptr as *mut Vec<u8>);
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

            Box::into_raw(dst_data_buf);
            Box::into_raw(dst_meta_buf);
            Box::into_raw(dst_random_key_buf);
            Box::into_raw(dst_nonce_buf);
            Box::into_raw(dst_hash_buf);

            super::shuffle_push_bin(
                shuffle_id,
                cur_bin_num,
                bin_type,
                data_ptr,
                meta_ptr,
                random_key_ptr,
                nonce_ptr,
                hash_ptr,
            );
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
    //i.e., the ocall is not called acutally, so we split
    //the function into two parts
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
    //free the buffer
    fn shuffle_pull_bin_post(
        data_ptr: usize,
        meta_ptr: usize,
        random_key_ptr: usize,
        nonce_ptr: usize,
        hash_ptr: usize,
    );
    fn shuffle_push_buckets(
        shuffle_id: u64,
        b_idx: usize,
        e_idx: usize,
        data: *const u8,
        data_size: usize,
        meta: *const u8,
        meta_size: usize,
    );
    //since only shuffle_push_bin occurs unexpected bugs
    //i.e., the ocall is not called acutally, so we split
    //the function into two parts
    //allocate buffer
    fn shuffle_push_bin_pre(
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
    fn shuffle_push_bin(
        shuffle_id: u64,
        cur_bin_num: usize,
        bin_type: u8,
        data_ptr: usize,
        meta_ptr: usize,
        random_key_ptr: usize,
        nonce_ptr: usize,
        hash_ptr: usize,
    );
    //move the dst bin to src bin, and assert that the src bin is empty now
    fn bin_switch(shuffle_id: u64, begin_bin_idx: usize, end_bin_idx: usize);

    //the sequential representation of tree is built, we just need to tranfer
    //the ownership to oram manager
    fn build_oram_from_shuffle_manager(
        shuffle_id: u64,
        allocation_id: u64,
        shuffle_nonce: *const u8,
        shuffle_nonce_size: usize,
    );
}
