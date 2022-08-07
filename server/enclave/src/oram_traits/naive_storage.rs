// Copyright (c) 2018-2021 The MobileCoin Foundation

//! HeapORAMStorage just uses a Vec to provide access to block storage in the
//! simplest way possible. It does not do any memory encryption or talk to
//! untrusted. It does not have any oblivious properties itself.
//! This is suitable for tests, or ORAMs that fit entirely in the enclave.

use super::*;
use crate::oram_storage::{persist_treetop, recover_treetop, s_decrypt, s_encrypt, ORAM_KEY};
use crate::{NonceSize, IS_LATEST, LIFETIME_ID, SNAPSHOT_ID};

use aligned_cmov::{typenum::Unsigned, Aligned};
use balanced_tree_index::TreeIndex;
use std::boxed::Box;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;

/// The HeapORAMStorage is simply vector
pub struct HeapORAMStorage<BlockSize: ArrayLength<u8>, MetaSize: ArrayLength<u8>, Z: Unsigned> {
    /// The current level of recursive ORAM
    level: u32,
    /// The storage for the blocks
    pub data: Vec<A64Bytes<BlockSize>>,
    /// The storage for the metadata
    pub metadata: Vec<A8Bytes<MetaSize>>,
    /// This is here so that we can provide good debug asserts in tests,
    /// it wouldn't be needed necessarily in a production version.
    checkout_index: Option<u64>,
    _z: PhantomData<fn() -> Z>,
}

impl<BlockSize: ArrayLength<u8>, MetaSize: ArrayLength<u8>, Z: Unsigned>
    HeapORAMStorage<BlockSize, MetaSize, Z>
{
    pub fn new(level: u32, size: u64) -> Self {
        let snapshot_id = SNAPSHOT_ID.load(Ordering::SeqCst);
        let is_latest = IS_LATEST.load(Ordering::SeqCst);
        let lifetime_id = LIFETIME_ID.load(Ordering::SeqCst);
        let mut data = vec![Default::default(); size as usize];
        let mut metadata = vec![Default::default(); size as usize];
        if snapshot_id > 0 && (is_latest || level == 0) {
            let ns = NonceSize::USIZE;
            let data_len = ns + 16 + 8 + 4 + 8 + size as usize * BlockSize::USIZE;
            let meta_len = ns + 16 + 8 + 4 + 8 + size as usize * MetaSize::USIZE;
            crate::ALLOCATOR.set_switch(true);
            let mut data_buf_ut = vec![0 as u8; data_len];
            let mut meta_buf_ut = vec![0 as u8; meta_len];
            crate::ALLOCATOR.set_switch(false);

            unsafe {
                recover_treetop(
                    data_buf_ut.as_mut_ptr(),
                    data_len,
                    meta_buf_ut.as_mut_ptr(),
                    meta_len,
                    level,
                    snapshot_id,
                );
            }

            let mut data_buf = data_buf_ut.clone();
            let mut meta_buf = meta_buf_ut.clone();
            crate::ALLOCATOR.set_switch(true);
            drop(data_buf_ut);
            drop(meta_buf_ut);
            crate::ALLOCATOR.set_switch(false);

            s_decrypt(&ORAM_KEY, &mut data_buf, 20);
            s_decrypt(&ORAM_KEY, &mut meta_buf, 20);

            //check the integrity
            let loaded_snapshot_id =
                u64::from_ne_bytes((&data_buf[(ns + 16)..(ns + 24)]).try_into().unwrap());
            assert_eq!(loaded_snapshot_id, snapshot_id);
            let loaded_level =
                u32::from_ne_bytes((&data_buf[(ns + 24)..(ns + 28)]).try_into().unwrap());
            assert_eq!(loaded_level, level);
            let loaded_lifetime_id =
                u64::from_ne_bytes((&data_buf[(ns + 28)..(ns + 36)]).try_into().unwrap());
            assert_eq!(loaded_lifetime_id, lifetime_id);
            let loaded_snapshot_id =
                u64::from_ne_bytes((&meta_buf[(ns + 16)..(ns + 24)]).try_into().unwrap());
            assert_eq!(loaded_snapshot_id, snapshot_id);
            let loaded_level =
                u32::from_ne_bytes((&meta_buf[(ns + 24)..(ns + 28)]).try_into().unwrap());
            assert_eq!(loaded_level, level);
            let loaded_lifetime_id =
                u64::from_ne_bytes((&meta_buf[(ns + 28)..(ns + 36)]).try_into().unwrap());
            assert_eq!(loaded_lifetime_id, lifetime_id);

            let iter_data = (&data_buf[(ns + 36)..]).chunks_exact(BlockSize::USIZE);
            assert_eq!(iter_data.remainder(), []);
            data = iter_data
                .into_iter()
                .map(|d| Aligned(GenericArray::clone_from_slice(d)))
                .collect::<Vec<_>>();
            let iter_meta = (&meta_buf[(ns + 36)..]).chunks_exact(MetaSize::USIZE);
            assert_eq!(iter_meta.remainder(), []);
            metadata = iter_meta
                .into_iter()
                .map(|m| Aligned(GenericArray::clone_from_slice(m)))
                .collect::<Vec<_>>();
        }

        Self {
            level,
            data,
            metadata,
            checkout_index: None,
            _z: Default::default(),
        }
    }
}

impl<BlockSize: ArrayLength<u8>, MetaSize: ArrayLength<u8>, Z: Unsigned>
    ORAMStorage<BlockSize, MetaSize, Z> for HeapORAMStorage<BlockSize, MetaSize, Z>
{
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn checkout(
        &mut self,
        leaf_index: u64,
        dest: &mut [A64Bytes<BlockSize>],
        dest_meta: &mut [A8Bytes<MetaSize>],
    ) {
        debug_assert!(self.checkout_index.is_none(), "double checkout");
        debug_assert!(dest.len() == dest_meta.len(), "buffer size mismatch");
        debug_assert!(
            leaf_index.parents().count() == dest.len(),
            "leaf height doesn't match buffer sizes"
        );
        for (n, tree_index) in leaf_index.parents().enumerate() {
            dest[n] = self.data[tree_index as usize].clone();
            dest_meta[n] = self.metadata[tree_index as usize].clone();
        }
        self.checkout_index = Some(leaf_index);
    }
    fn checkin(
        &mut self,
        leaf_index: u64,
        src: &mut [A64Bytes<BlockSize>],
        src_meta: &mut [A8Bytes<MetaSize>],
    ) {
        debug_assert!(self.checkout_index.is_some(), "checkin without checkout");
        debug_assert!(
            self.checkout_index == Some(leaf_index),
            "unexpected checkin"
        );
        debug_assert!(src.len() == src_meta.len(), "buffer size mismatch");
        debug_assert!(
            leaf_index.parents().count() == src.len(),
            "leaf height doesn't match buffer sizes"
        );
        for (n, tree_index) in leaf_index.parents().enumerate() {
            self.data[tree_index as usize] = src[n].clone();
            self.metadata[tree_index as usize] = src_meta[n].clone();
        }
        self.checkout_index = None;
    }

    fn persist<Rng: RngCore + CryptoRng>(
        &mut self,
        lifetime_id: u64,
        new_snapshot_id: u64,
        volatile: bool,
        rng: &mut Rng,
    ) {
        //encrypt the treetop and send it out
        //TODO: This step can be in parallel with the following ones
        assert!(self.data.len() == self.metadata.len());
        let mut data = vec![0; NonceSize::USIZE + 16];
        data.extend_from_slice(&new_snapshot_id.to_ne_bytes());
        data.extend_from_slice(&self.level.to_ne_bytes());
        data.extend_from_slice(&lifetime_id.to_ne_bytes());
        for d in &self.data {
            data.extend_from_slice(d);
        }

        let mut meta = vec![0; NonceSize::USIZE + 16];
        meta.extend_from_slice(&new_snapshot_id.to_ne_bytes());
        meta.extend_from_slice(&self.level.to_ne_bytes());
        meta.extend_from_slice(&lifetime_id.to_ne_bytes());
        for m in &self.metadata {
            meta.extend_from_slice(m);
        }

        s_encrypt(&ORAM_KEY, &mut data, 20, rng);
        s_encrypt(&ORAM_KEY, &mut meta, 20, rng);

        crate::ALLOCATOR.set_switch(true);
        let data = data.clone();
        let meta = meta.clone();
        crate::ALLOCATOR.set_switch(false);

        // TODO: the ocall should not stall the steps after it
        unsafe {
            persist_treetop(
                data.as_ptr(),
                data.len(),
                meta.as_ptr(),
                meta.len(),
                self.level,
                new_snapshot_id,
                volatile as u8,
            )
        }

        crate::ALLOCATOR.set_switch(true);
        drop(data);
        drop(meta);
        crate::ALLOCATOR.set_switch(false);
    }
}

/// HeapORAMStorage simply allocates a vector, and requires no special
/// initialization support
pub struct HeapORAMStorageCreator {}

impl<BlockSize, MetaSize, Z> ORAMStorageCreator<BlockSize, MetaSize, Z> for HeapORAMStorageCreator
where
    BlockSize: ArrayLength<u8> + 'static,
    MetaSize: ArrayLength<u8> + 'static,
    Z: Unsigned,
{
    type Output = HeapORAMStorage<BlockSize, MetaSize, Z>;
    type Error = HeapORAMStorageCreatorError;

    fn create<R: RngCore + CryptoRng>(
        level: u32,
        size: u64,
        _rng: &mut R,
    ) -> Result<Self::Output, Self::Error> {
        Ok(Self::Output::new(level, size))
    }
}

/// There are not actually any failure modes
#[derive(Debug)]
pub enum HeapORAMStorageCreatorError {}

impl core::fmt::Display for HeapORAMStorageCreatorError {
    fn fmt(&self, _: &mut core::fmt::Formatter) -> core::fmt::Result {
        unreachable!()
    }
}
