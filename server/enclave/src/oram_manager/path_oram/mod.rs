//! Implements PathORAM on top of a generic ORAMStorage and a generic
//! PositionMap.
//!
//! In this implementation, the bucket size (Z in paper) is configurable.
//!
//! The storage will hold blocks of size ValueSize * Z for the data, and
//! MetaSize * Z for the metadata.
//!
//! Most papers suggest Z = 2 or Z = 4, Z = 1 probably won't work.
//!
//! It is expected that you want the block size to be 4096 (one linux page)
//!
//! Height of storage tree is set as log size - log bucket_size
//! This is informed by Gentry et al.

use std::boxed::Box;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::ops::Mul;
use std::sync::atomic::Ordering;
use std::vec::Vec;

use aligned_cmov::{
    subtle::{Choice, ConstantTimeEq, ConstantTimeLess},
    typenum::{PartialDiv, Prod, Unsigned, U64, U8},
    A64Bytes, A8Bytes, Aligned, ArrayLength, AsAlignedChunks, AsNeSlice, CMov, GenericArray,
};
use balanced_tree_index::TreeIndex;
use rand_core::{CryptoRng, RngCore};

use crate::oram_manager::{DataMetaSize, PosMetaSize};
use crate::oram_storage::{
    persist_stash, recover_stash, s_decrypt, s_encrypt, MAX_LEVEL, ORAM_KEY,
};
use crate::oram_traits::{
    log2_ceil, ORAMStorage, ORAMStorageCreator, PositionMap, PositionMapCreator, ORAM,
};
use crate::{NonceSize, IS_LATEST, LIFETIME_ID, SNAPSHOT_ID};

/// In this implementation, a value is expected to be an aligned 4096 byte page.
/// The metadata associated to a value is three u64's (block num, leaf, and counter), so 24
/// bytes. It is stored separately from the value so as not to break alignment.
/// In many cases block-num and leaf can be u32's. But I suspect that there will
/// be other stuff in this metadata as well in the end so the savings isn't
/// much.
///
// MetaSize can be U16 for position map ORAM or U24 for data ORAM
const DATA_META_SIZE: u64 = DataMetaSize::U64;
const POS_META_SIZE: u64 = PosMetaSize::U64;

// A metadata object is always associated to any Value in the PathORAM
// structure. A metadata consists of three fields: leaf_num, block_num, and counter
// A metadata has the status of being "vacant" or "not vacant".
//
// The block_num is the number in range 0..len that corresponds to the user's
// query. every block of data in the ORAM has an associated block number.
// There should be only one non-vacant data with a given block number at a time,
// if none is found then it will be initialized lazily on first query.
//
// The leaf_num is the "target" of this data in the tree, according to Path ORAM
// algorithm. It represents a TreeIndex value. In particular it is not zero.
//
// The leaf_num attached to a block_num should match pos[block_num], it is a
// cache of that value, which enables us to perform efficient eviction and
// packing in a branch.
//
// A metadata is defined to be "vacant" if leaf_num IS zero.
// This indicates that the metadata and its corresponding value can be
// overwritten with a real item.
//
// The counter is used to distinguish stale blocks and up-to-date blocks with the same
// block_num. It is needed when there exist multiple blocks with the same block_num, as
// we append the newest block to disk, instead of in-place update. Note that the position
// map ORAM does not need counter, because we do not persist the position map.

/// Get the leaf num of a metadata
fn meta_leaf_num<MetaSize>(src: &A8Bytes<MetaSize>) -> &u64
where
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    &src.as_ne_u64_slice()[0]
}
/// Get the leaf num of a mutable metadata
fn meta_leaf_num_mut<MetaSize>(src: &mut A8Bytes<MetaSize>) -> &mut u64
where
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    &mut src.as_mut_ne_u64_slice()[0]
}
/// Get the block num of a metadata
fn meta_block_num<MetaSize>(src: &A8Bytes<MetaSize>) -> &u64
where
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    &src.as_ne_u64_slice()[1]
}
/// Get the block num of a mutable metadata
fn meta_block_num_mut<MetaSize>(src: &mut A8Bytes<MetaSize>) -> &mut u64
where
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    &mut src.as_mut_ne_u64_slice()[1]
}
/// Get the counter of a metadata
fn meta_counter<MetaSize>(src: &A8Bytes<MetaSize>) -> &u64
where
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    &src.as_ne_u64_slice()[2]
}
/// Get the counter of a mutable metadata
fn meta_counter_mut<MetaSize>(src: &mut A8Bytes<MetaSize>) -> &mut u64
where
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    &mut src.as_mut_ne_u64_slice()[2]
}
/// Test if a metadata is "vacant"
fn meta_is_vacant<MetaSize>(src: &A8Bytes<MetaSize>) -> Choice
where
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    meta_leaf_num(src).ct_eq(&0)
}
/// Set a metadata to vacant, obliviously, if a condition is true
fn meta_set_vacant<MetaSize>(condition: Choice, src: &mut A8Bytes<MetaSize>)
where
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
{
    meta_leaf_num_mut(src).cmov(condition, &0);
}

/// An implementation of PathORAM, using u64 to represent leaves in metadata.
pub struct PathORAM<ValueSize, MetaSize, Z, StorageType, RngType>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
    StorageType: ORAMStorage<Prod<Z, ValueSize>, Prod<Z, MetaSize>, Z> + Send + Sync + 'static,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    /// The current level of recursive ORAM. If 0, it is data oram.
    /// Otherwise it is position map oram.
    level: u32,
    /// The height of the binary tree used for storage
    height: u32,
    /// The storage itself
    storage: StorageType,
    /// The position map
    pos: Box<dyn PositionMap + Send + Sync + 'static>,
    /// The rng
    rng: RngType,
    /// The stashed values
    stash_data: Vec<A64Bytes<ValueSize>>,
    /// The stashed metadata
    stash_meta: Vec<A8Bytes<MetaSize>>,
    /// Our currently checked-out branch if any
    branch: BranchCheckout<ValueSize, MetaSize, Z>,
}

impl<ValueSize, MetaSize, Z, StorageType, RngType>
    PathORAM<ValueSize, MetaSize, Z, StorageType, RngType>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
    StorageType: ORAMStorage<Prod<Z, ValueSize>, Prod<Z, MetaSize>, Z> + Send + Sync + 'static,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    /// New function creates this ORAM given a position map creator and a
    /// storage type creator and an Rng creator.
    /// The main thing that is going on here is, given the size, we are
    /// determining what the height will be, which will be like log(size) -
    /// log(bucket_size) Then we are making sure that all the various
    /// creators use this number.
    pub fn new<
        PMC: PositionMapCreator<RngType>,
        SC: ORAMStorageCreator<Prod<Z, ValueSize>, Prod<Z, MetaSize>, Z, Output = StorageType>,
        F: FnMut() -> RngType + 'static,
    >(
        size: u64,
        stash_size: usize,
        rng_maker: &mut F,
    ) -> Self {
        assert!(size != 0, "size cannot be zero");
        assert!(size & (size - 1) == 0, "size must be a power of two");
        // saturating_sub is used so that creating an ORAM of size 1 or 2 doesn't fail
        let height = log2_ceil(size).saturating_sub(log2_ceil(Z::U64));
        let level = MAX_LEVEL.fetch_add(1, Ordering::SeqCst);
        println!("current level = {:?}", level);

        let snapshot_id = SNAPSHOT_ID.load(Ordering::SeqCst);
        let is_latest = IS_LATEST.load(Ordering::SeqCst);
        let lifetime_id = LIFETIME_ID.load(Ordering::SeqCst);
        let mut stash_data: Vec<A64Bytes<ValueSize>> = vec![Default::default(); stash_size];
        let mut stash_meta: Vec<A8Bytes<MetaSize>> = vec![Default::default(); stash_size];
        //If it is not the latest version, global shuffle will be performed,
        //So the stashes of position ORAMs are useless, and they will be not recovered
        //TODO: For the stash of stale data ORAM, the leaves of blocks in it may need to be reassigned
        if snapshot_id > 0 && (is_latest || level == 0) {
            let ns = NonceSize::USIZE;
            let stash_data_len = ns + 16 + 8 + 4 + 8 + stash_size * ValueSize::USIZE;
            let mut stash_data_buf = vec![0 as u8; stash_data_len];
            let stash_meta_len = ns + 16 + 8 + 4 + 8 + stash_size * MetaSize::USIZE;
            let mut stash_meta_buf = vec![0 as u8; stash_meta_len];
            unsafe {
                recover_stash(
                    stash_data_buf.as_mut_ptr(),
                    stash_data_len,
                    stash_meta_buf.as_mut_ptr(),
                    stash_meta_len,
                    level,
                    snapshot_id,
                );
            }

            s_decrypt(&ORAM_KEY, &mut stash_data_buf, 20);
            s_decrypt(&ORAM_KEY, &mut stash_meta_buf, 20);

            //check the integrity
            let loaded_snapshot_id =
                u64::from_ne_bytes((&stash_data_buf[(ns + 16)..(ns + 24)]).try_into().unwrap());
            assert_eq!(loaded_snapshot_id, snapshot_id);
            let loaded_level =
                u32::from_ne_bytes((&stash_data_buf[(ns + 24)..(ns + 28)]).try_into().unwrap());
            assert_eq!(loaded_level, level);
            let loaded_lifetime_id =
                u64::from_ne_bytes((&stash_data_buf[(ns + 28)..(ns + 36)]).try_into().unwrap());
            assert_eq!(loaded_lifetime_id, lifetime_id);
            let loaded_snapshot_id =
                u64::from_ne_bytes((&stash_meta_buf[(ns + 16)..(ns + 24)]).try_into().unwrap());
            assert_eq!(loaded_snapshot_id, snapshot_id);
            let loaded_level =
                u32::from_ne_bytes((&stash_meta_buf[(ns + 24)..(ns + 28)]).try_into().unwrap());
            assert_eq!(loaded_level, level);
            let loaded_lifetime_id =
                u64::from_ne_bytes((&stash_meta_buf[(ns + 28)..(ns + 36)]).try_into().unwrap());
            assert_eq!(loaded_lifetime_id, lifetime_id);

            let iter_data = (&stash_data_buf[(ns + 36)..]).chunks_exact(ValueSize::USIZE);
            assert_eq!(iter_data.remainder(), []);
            stash_data = iter_data
                .into_iter()
                .map(|d| Aligned(GenericArray::clone_from_slice(d)))
                .collect::<Vec<_>>();
            let iter_meta = (&stash_meta_buf[(ns + 36)..]).chunks_exact(MetaSize::USIZE);
            assert_eq!(iter_meta.remainder(), []);
            stash_meta = iter_meta
                .into_iter()
                .map(|m| Aligned(GenericArray::clone_from_slice(m)))
                .collect::<Vec<_>>();
        }

        // This is 2u64 << height because it must be 2^{h+1}, we have defined
        // the height of the root to be 0, so in a tree where the lowest level
        // is h, there are 2^{h+1} nodes.
        let mut rng = rng_maker();
        let storage = SC::create(level, 2u64 << height, &mut rng).expect("Storage failed");
        let pos = PMC::create(size, height, stash_size, rng_maker);

        Self {
            level,
            height,
            storage,
            pos,
            rng,
            stash_data,
            stash_meta,
            branch: Default::default(),
        }
    }
}

impl<ValueSize, MetaSize, Z, StorageType, RngType> ORAM<ValueSize>
    for PathORAM<ValueSize, MetaSize, Z, StorageType, RngType>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    RngType: RngCore + CryptoRng + Send + Sync + 'static,
    StorageType: ORAMStorage<Prod<Z, ValueSize>, Prod<Z, MetaSize>, Z> + Send + Sync + 'static,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    fn len(&self) -> u64 {
        self.pos.len()
    }
    // TODO: We should try implementing a circuit-ORAM like approach also
    fn access<T, F: FnOnce(&mut A64Bytes<ValueSize>, &mut u64) -> T>(
        &mut self,
        key: u64,
        f: F,
    ) -> T {
        let result: T;
        // Choose what will be the next (secret) position of this item
        let new_pos = 1u64.random_child_at_height(self.height, &mut self.rng);
        // Set the new value and recover the old (current) position.
        let (current_pos, is_invalid_pos) = self.pos.write(&key, &new_pos);
        debug_assert!(current_pos != 0, "position map told us the item is at 0");
        // Get the branch where we expect to find the item.
        // NOTE: If we move to a scheme where the tree can be resized dynamically,
        // then we should checkout at `current_pos.random_child_at_height(self.height)`.
        debug_assert!(self.branch.leaf == 0);
        self.branch.checkout(&mut self.storage, current_pos);

        // Fetch the item from branch and then from stash.
        // Visit it and then insert it into the stash.
        {
            debug_assert!(self.branch.leaf == current_pos);
            let mut meta = A8Bytes::<MetaSize>::default();
            let mut data = A64Bytes::<ValueSize>::default();

            self.branch
                .ct_find_and_remove(1.into(), &key, &mut data, &mut meta);
            details::ct_find_and_remove(
                1.into(),
                &key,
                &mut data,
                &mut meta,
                &mut self.stash_data,
                &mut self.stash_meta,
            );
            debug_assert!(
                meta_block_num(&meta) == &key || meta_is_vacant(&meta).into(),
                "Hmm, we didn't find the expected item something else"
            );
            debug_assert!(self.branch.leaf == current_pos);

            // Call the callback, then store the result
            // The match here is secure because it is public information
            // that whether it is position map ORAM or data ORAM
            if MetaSize::U64 == DATA_META_SIZE {
                result = f(&mut data, meta_counter_mut(&mut meta));
            } else if MetaSize::U64 == POS_META_SIZE {
                result = f(&mut data, &mut 0);
            } else {
                panic!("MetaSize {:?} is invalid", MetaSize::U64);
            };

            // Set the block_num in case the item was not initialized yet
            *meta_block_num_mut(&mut meta) = key;
            // Set the new leaf destination for the item
            *meta_leaf_num_mut(&mut meta) = new_pos;

            // Stash the item
            details::ct_insert(
                1.into(),
                &data,
                &mut meta,
                &mut self.stash_data,
                &mut self.stash_meta,
            );
            assert!(bool::from(meta_is_vacant(&meta)), "Stash overflow!");
        }

        // Now do cleanup / eviction on this branch, before checking out
        {
            debug_assert!(self.branch.leaf == current_pos);
            self.branch.pack();
            for idx in 0..self.stash_data.len() {
                self.branch
                    .ct_insert(1.into(), &self.stash_data[idx], &mut self.stash_meta[idx]);
            }
        }

        debug_assert!(self.branch.leaf == current_pos);
        self.branch.checkin(&mut self.storage);
        debug_assert!(self.branch.leaf == 0);

        result
    }

    fn persist(&mut self, lifetime_id: u64, new_snapshot_id: u64, volatile: bool) {
        //encrypt the stash and send it out
        //TODO: This step can be in parallel with the following ones
        assert!(self.stash_data.len() == self.stash_meta.len());
        let mut stash_data = vec![0; NonceSize::USIZE + 16];
        stash_data.extend_from_slice(&new_snapshot_id.to_ne_bytes());
        stash_data.extend_from_slice(&self.level.to_ne_bytes());
        stash_data.extend_from_slice(&lifetime_id.to_ne_bytes());
        for d in &self.stash_data {
            stash_data.extend_from_slice(d);
        }

        let mut stash_meta = vec![0; NonceSize::USIZE + 16];
        stash_meta.extend_from_slice(&new_snapshot_id.to_ne_bytes());
        stash_meta.extend_from_slice(&self.level.to_ne_bytes());
        stash_meta.extend_from_slice(&lifetime_id.to_ne_bytes());
        for m in &self.stash_meta {
            stash_meta.extend_from_slice(m);
        }

        //It is unnecessary to encrypt lifetime_id，as long as it
        //is not randomly generated
        s_encrypt(&ORAM_KEY, &mut stash_data, 20, &mut self.rng);
        s_encrypt(&ORAM_KEY, &mut stash_meta, 20, &mut self.rng);

        // TODO: the ocall should not stall the steps after it
        unsafe {
            persist_stash(
                stash_data.as_ptr(),
                stash_data.len(),
                stash_meta.as_ptr(),
                stash_meta.len(),
                self.level,
                new_snapshot_id,
                volatile as u8,
            )
        }

        self.storage
            .persist(lifetime_id, new_snapshot_id, volatile, &mut self.rng);
        self.pos.persist(lifetime_id, new_snapshot_id, volatile);
    }
}

/// Struct which represents a branch which we have checked out, including its
/// leaf and the associated data.
///
/// This struct is a member of PathORAM and is long lived, so that we don't
/// call malloc with every checkout.
///
/// This is mainly just an organizational thing.
struct BranchCheckout<ValueSize, MetaSize, Z>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    /// The leaf of branch that is currently checked-out. 0 if no existing
    /// checkout.
    leaf: u64,
    /// The scratch-space for checked-out branch data
    data: Vec<A64Bytes<Prod<Z, ValueSize>>>,
    /// The scratch-space for checked-out branch metadata
    meta: Vec<A8Bytes<Prod<Z, MetaSize>>>,
    /// Phantom data for ValueSize
    _value_size: PhantomData<fn() -> ValueSize>,
}

impl<ValueSize, MetaSize, Z> Default for BranchCheckout<ValueSize, MetaSize, Z>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    fn default() -> Self {
        Self {
            leaf: 0,
            data: Default::default(),
            meta: Default::default(),
            _value_size: Default::default(),
        }
    }
}

impl<ValueSize, MetaSize, Z> BranchCheckout<ValueSize, MetaSize, Z>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
    MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    Z: Unsigned + Mul<ValueSize> + Mul<MetaSize>,
    Prod<Z, ValueSize>: ArrayLength<u8> + PartialDiv<U8>,
    Prod<Z, MetaSize>: ArrayLength<u8> + PartialDiv<U8>,
{
    /// Try to extract an item from the branch
    pub fn ct_find_and_remove(
        &mut self,
        condition: Choice,
        query: &u64,
        dest_data: &mut A64Bytes<ValueSize>,
        dest_meta: &mut A8Bytes<MetaSize>,
    ) {
        debug_assert!(self.data.len() == self.meta.len());
        for idx in 0..self.data.len() {
            let bucket_data: &mut [A64Bytes<ValueSize>] = self.data[idx].as_mut_aligned_chunks();
            let bucket_meta: &mut [A8Bytes<MetaSize>] = self.meta[idx].as_mut_aligned_chunks();
            debug_assert!(bucket_data.len() == Z::USIZE);
            debug_assert!(bucket_meta.len() == Z::USIZE);

            details::ct_find_and_remove(
                condition,
                query,
                dest_data,
                dest_meta,
                bucket_data,
                bucket_meta,
            );
        }
    }

    /// Try to insert an item into the branch, as low as it can go, consistent
    /// with the invariant.
    pub fn ct_insert(
        &mut self,
        mut condition: Choice,
        src_data: &A64Bytes<ValueSize>,
        src_meta: &mut A8Bytes<MetaSize>,
    ) {
        condition &= !meta_is_vacant(src_meta);
        let lowest_height_legal_index = self.lowest_height_legal_index(*meta_leaf_num(src_meta));
        Self::insert_into_branch_suffix(
            condition,
            src_data,
            src_meta,
            lowest_height_legal_index,
            &mut self.data,
            &mut self.meta,
        );
    }

    /// This is the Path ORAM branch packing procedure, which we implement
    /// obliviously in a naive way.
    pub fn pack(&mut self) {
        debug_assert!(self.leaf != 0);
        debug_assert!(self.data.len() == self.meta.len());
        let data_len = self.data.len();
        for bucket_num in 1..self.data.len() {
            let (lower_data, upper_data) = self.data.split_at_mut(bucket_num);
            let (lower_meta, upper_meta) = self.meta.split_at_mut(bucket_num);

            let bucket_data: &mut [A64Bytes<ValueSize>] = upper_data[0].as_mut_aligned_chunks();
            let bucket_meta: &mut [A8Bytes<MetaSize>] = upper_meta[0].as_mut_aligned_chunks();

            debug_assert!(bucket_data.len() == bucket_meta.len());
            for idx in 0..bucket_data.len() {
                let src_data: &mut A64Bytes<ValueSize> = &mut bucket_data[idx];
                let src_meta: &mut A8Bytes<MetaSize> = &mut bucket_meta[idx];

                // We use the _impl version here because we cannot borrow self
                // while self.data and self.meta are borrowed
                let lowest_height_legal_index = Self::lowest_height_legal_index_impl(
                    *meta_leaf_num(src_meta),
                    self.leaf,
                    data_len,
                );
                Self::insert_into_branch_suffix(
                    1.into(),
                    src_data,
                    src_meta,
                    lowest_height_legal_index,
                    lower_data,
                    lower_meta,
                );
            }
        }
        debug_assert!(self.leaf != 0);
    }

    /// Checkout a branch from storage into ourself
    pub fn checkout(
        &mut self,
        storage: &mut impl ORAMStorage<Prod<Z, ValueSize>, Prod<Z, MetaSize>, Z>,
        leaf: u64,
    ) {
        debug_assert!(self.leaf == 0);
        self.data
            .resize_with(leaf.height() as usize + 1, Default::default);
        self.meta
            .resize_with(leaf.height() as usize + 1, Default::default);
        storage.checkout(leaf, &mut self.data, &mut self.meta);
        self.leaf = leaf;
    }

    /// Checkin our branch to storage and clear our checkout_leaf
    pub fn checkin(
        &mut self,
        storage: &mut impl ORAMStorage<Prod<Z, ValueSize>, Prod<Z, MetaSize>, Z>,
    ) {
        debug_assert!(self.leaf != 0);
        storage.checkin(self.leaf, &mut self.data, &mut self.meta);
        self.leaf = 0;
    }

    /// Given a tree-index value (a node in the tree)
    /// Compute the lowest height (closest to the leaf) legal index of a bucket
    /// in this branch into which it can be placed. This depends on the
    /// common ancestor height of tree_index and self.leaf.
    ///
    /// This is required to give well-defined output even if tree_index is 0.
    /// It is not required to give well-defined output if self.leaf is 0.
    fn lowest_height_legal_index(&self, query: u64) -> usize {
        Self::lowest_height_legal_index_impl(query, self.leaf, self.data.len())
    }

    /// The internal logic of lowest_height_legal_index.
    /// This stand-alone version is needed to get around the borrow checker,
    /// because we cannot call functions that take &self as a parameter
    /// while data or meta are mutably borrowed.
    fn lowest_height_legal_index_impl(mut query: u64, leaf: u64, data_len: usize) -> usize {
        // Set query to point to root (1) if it is currently 0 (none / vacant)
        query.cmov(query.ct_eq(&0), &1);
        debug_assert!(
            leaf != 0,
            "this should not be called when there is not currently a checkout"
        );

        let common_ancestor_height = leaf.common_ancestor_height(&query) as usize;
        debug_assert!(data_len > common_ancestor_height);
        data_len - 1 - common_ancestor_height
    }

    /// Low-level helper function: Insert an item into (a portion of) the branch
    /// - No inspection of the src_meta is performed
    /// - The first free spot in a bucket of index >= insert_after_index is used
    /// - The destination slices need not be the whole branch, they could be a
    ///   prefix
    fn insert_into_branch_suffix(
        condition: Choice,
        src_data: &A64Bytes<ValueSize>,
        src_meta: &mut A8Bytes<MetaSize>,
        insert_after_index: usize,
        dest_data: &mut [A64Bytes<Prod<Z, ValueSize>>],
        dest_meta: &mut [A8Bytes<Prod<Z, MetaSize>>],
    ) {
        debug_assert!(dest_data.len() == dest_meta.len());
        for idx in 0..dest_data.len() {
            details::ct_insert::<ValueSize, MetaSize>(
                condition & !(idx as u64).ct_lt(&(insert_after_index as u64)),
                src_data,
                src_meta,
                dest_data[idx].as_mut_aligned_chunks(),
                dest_meta[idx].as_mut_aligned_chunks(),
            )
        }
    }
}

/// Constant time helper functions
mod details {
    use super::*;

    /// ct_find_and_remove tries to find and remove an item with a particular
    /// block num from a mutable sequence, and store it in dest_data and
    /// dest_meta.
    ///
    /// The condition value that is passed must be true or no move will actually
    /// happen. When the operation succeeds in finding an item, dest_meta
    /// will not be vacant and will have the desired block_num, and that
    /// item will be set vacant in the mutable sequence.
    ///
    /// Semantics: If dest is vacant, and condition is true,
    ///            scan across src and find the first non-vacant item with
    ///            desired block_num then cmov that to dest.
    ///            Also set source to vacant.
    ///
    /// The whole operation must be constant time.
    pub fn ct_find_and_remove<ValueSize, MetaSize>(
        mut condition: Choice,
        query: &u64,
        dest_data: &mut A64Bytes<ValueSize>,
        dest_meta: &mut A8Bytes<MetaSize>,
        src_data: &mut [A64Bytes<ValueSize>],
        src_meta: &mut [A8Bytes<MetaSize>],
    ) where
        ValueSize: ArrayLength<u8>,
        MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    {
        debug_assert!(src_data.len() == src_meta.len());
        for idx in 0..src_meta.len() {
            // XXX: Must be constant time and not optimized, may need a better barrier here
            // Maybe just use subtle::Choice
            let test = condition
                & (query.ct_eq(meta_block_num(&src_meta[idx])))
                & !meta_is_vacant(&src_meta[idx]);
            dest_meta.cmov(test, &src_meta[idx]);
            dest_data.cmov(test, &src_data[idx]);
            // Zero out the src[meta] if we moved it
            meta_set_vacant(test, &mut src_meta[idx]);
            condition &= !test;
        }
    }

    /// ct_insert tries to insert an item into a mutable sequence
    ///
    /// It takes the source data and source metadata, (the item being inserted),
    /// and slices corresponding to the destination data and metadata.
    /// It also takes a boolean "condition", if the condition is false,
    /// then all the memory accesses will be done but no side-effects will
    /// occur.
    ///
    /// Semantics: If source is not vacant, and condition is true,
    ///            scan across destination and find the first vacant slot,
    ///            then cmov the source to the slot.
    ///            Also set source to vacant.
    ///
    /// The whole operation must be constant time.
    pub fn ct_insert<ValueSize, MetaSize>(
        mut condition: Choice,
        src_data: &A64Bytes<ValueSize>,
        src_meta: &mut A8Bytes<MetaSize>,
        dest_data: &mut [A64Bytes<ValueSize>],
        dest_meta: &mut [A8Bytes<MetaSize>],
    ) where
        ValueSize: ArrayLength<u8>,
        MetaSize: ArrayLength<u8> + PartialDiv<U8>,
    {
        debug_assert!(dest_data.len() == dest_meta.len());
        condition &= !meta_is_vacant(src_meta);
        for idx in 0..dest_meta.len() {
            // XXX: Must be constant time and not optimized, may need a better barrier here
            // Maybe just use subtle::Choice
            let test = condition & meta_is_vacant(&dest_meta[idx]);
            dest_meta[idx].cmov(test, src_meta);
            dest_data[idx].cmov(test, src_data);
            meta_set_vacant(test, src_meta);
            condition &= !test;
        }
    }
}
