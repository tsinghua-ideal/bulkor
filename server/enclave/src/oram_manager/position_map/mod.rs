//! Defines oblivious position map on top of a generic ORAM,
//! using strategy as described in PathORAM
//! In our representation of tree-index, a value of 0 represents
//! a "vacant" / uninitialized value.
//!
//! For correctness we must ensure that the position map appears in
//! a randomly initialized state, so we replace zeros with a random leaf
//! at the correct height before returning to caller.

use std::boxed::Box;
use std::convert::TryInto;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;
use std::vec::Vec;

use aligned_cmov::{
    subtle::ConstantTimeEq,
    typenum::{PartialDiv, Unsigned, U8},
    ArrayLength, AsNeSlice, CMov,
};
use balanced_tree_index::TreeIndex;
use rand_core::{CryptoRng, RngCore};
use subtle::Choice;

use crate::oram_storage::{
    persist_trivial_posmap, recover_trivial_posmap, s_decrypt, s_encrypt,
    shuffle_manager::oblivious_pull_trivial_posmap, ORAM_KEY,
};
use crate::oram_traits::{log2_ceil, ORAMCreator, PositionMap, PositionMapCreator, ORAM};
use crate::{NonceSize, ID_TO_LOG_POS, IS_LATEST, LIFETIME_ID, SNAPSHOT_ID};

/// This threshold is a total guess, this corresponds to four pages
pub const POS_MAP_THRESHOLD: u64 = 4096;

/// A trivial position map implemented via linear scanning.
/// Positions are represented as 32 bytes inside a page.
pub struct TrivialPositionMap<R: RngCore + CryptoRng> {
    data: Vec<u32>,
    height: u32,
    rng: R,
}

impl<R: RngCore + CryptoRng> TrivialPositionMap<R> {
    /// Create trivial position map
    pub fn new(size: u64, height: u32, rng_maker: &mut impl FnMut() -> R) -> Self {
        assert!(
            height < 32,
            "Can't use u32 position map when height of tree exceeds 31"
        );

        let mut data = vec![0u32; size as usize];
        let snapshot_id = SNAPSHOT_ID.load(Ordering::SeqCst);
        let is_latest = IS_LATEST.load(Ordering::SeqCst);
        let lifetime_id = LIFETIME_ID.load(Ordering::SeqCst);
        //if not latest, we need the loaded_log_pos
        if snapshot_id > 0 {
            let ns = NonceSize::USIZE;
            //NOTE: should be consistent with `size_triv_pos_map` in lib.rs
            let posmap_len = ns + 16 + 8 + 8 + 8 + size as usize * 4; //no level, but have log pos
            let mut posmap_buf = vec![0 as u8; posmap_len];
            unsafe {
                recover_trivial_posmap(posmap_buf.as_mut_ptr(), posmap_len, snapshot_id);
            }
            s_decrypt(&ORAM_KEY, &mut posmap_buf, 24);
            let loaded_log_pos =
                u64::from_ne_bytes((&posmap_buf[(ns + 16)..(ns + 24)]).try_into().unwrap());
            ID_TO_LOG_POS
                .lock()
                .unwrap()
                .insert(snapshot_id, loaded_log_pos);
            //check the integrity
            let loaded_snapshot_id =
                u64::from_ne_bytes((&posmap_buf[(ns + 24)..(ns + 32)]).try_into().unwrap());
            assert_eq!(loaded_snapshot_id, snapshot_id);
            if is_latest {
                let loaded_lifetime_id =
                    u64::from_ne_bytes((&posmap_buf[(ns + 32)..(ns + 40)]).try_into().unwrap());
                assert_eq!(loaded_lifetime_id, lifetime_id);

                let iter_data = (&posmap_buf[(ns + 40)..]).chunks_exact(4);
                assert_eq!(iter_data.remainder(), []);
                data = iter_data
                    .into_iter()
                    .map(|d| u32::from_ne_bytes(d.try_into().unwrap()))
                    .collect::<Vec<_>>();
            } else {
                oblivious_pull_trivial_posmap(&mut data);
            }
        }

        Self {
            data,
            height,
            rng: rng_maker(),
        }
    }
}

impl<R: RngCore + CryptoRng> PositionMap for TrivialPositionMap<R> {
    fn len(&self) -> u64 {
        self.data.len() as u64
    }
    fn write(&mut self, key: &u64, new_val: &u64) -> (u64, Choice) {
        debug_assert!(*key < self.data.len() as u64, "key was out of bounds");
        let key = *key as u32;
        let new_val = *new_val as u32;
        let mut old_val = 0u32;
        for idx in 0..self.data.len() {
            let test = (idx as u32).ct_eq(&key);
            old_val.cmov(test, &self.data[idx]);
            (&mut self.data[idx]).cmov(test, &new_val);
        }
        // if old_val is zero, sample a random leaf
        let cond = old_val.ct_eq(&0);
        old_val.cmov(
            cond,
            &1u32.random_child_at_height(self.height, &mut self.rng),
        );
        (old_val as u64, cond)
    }

    fn persist(&mut self, lifetime_id: u64, new_snapshot_id: u64, volatile: bool) {
        //encrypt the merkle roots and send it out
        //TODO: This step can be in parallel with the following ones
        let mut posmap = vec![0; NonceSize::USIZE + 16];
        let cur_log_pos = ID_TO_LOG_POS
            .lock()
            .unwrap()
            .remove(&new_snapshot_id)
            .unwrap();
        posmap.extend_from_slice(&cur_log_pos.to_ne_bytes());
        posmap.extend_from_slice(&new_snapshot_id.to_ne_bytes());
        posmap.extend_from_slice(&lifetime_id.to_ne_bytes());
        for d in &self.data {
            posmap.extend_from_slice(&d.to_ne_bytes());
        }

        s_encrypt(&ORAM_KEY, &mut posmap, 24, &mut self.rng);

        // TODO: the ocall should not stall the steps after it
        unsafe {
            persist_trivial_posmap(
                posmap.as_ptr(),
                posmap.len(),
                new_snapshot_id,
                volatile as u8,
            )
        }
    }
}

/// A position map implemented on top of an ORAM
/// Positions are represented as 32 bytes inside a page in an ORAM.
///
/// Value size represents the chunk of 32 byte values that we scan across.
pub struct ORAMU32PositionMap<
    ValueSize: ArrayLength<u8> + PartialDiv<U8>,
    O: ORAM<ValueSize> + Send + Sync + 'static,
    R: RngCore + CryptoRng + Send + Sync + 'static,
> {
    oram: O,
    height: u32,
    rng: R,
    _value_size: PhantomData<fn() -> ValueSize>,
}

impl<ValueSize, O, R> ORAMU32PositionMap<ValueSize, O, R>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8>,
    O: ORAM<ValueSize> + Send + Sync + 'static,
    R: RngCore + CryptoRng + Send + Sync + 'static,
{
    // We subtract 2 over ValueSize because u32 is 4 bytes
    const L: u32 = log2_ceil(ValueSize::U64) - 2;

    /// Create position map where all positions appear random, lazily
    pub fn new<OC: ORAMCreator<ValueSize, R, Output = O>, M: 'static + FnMut() -> R>(
        size: u64,
        height: u32,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Self {
        assert!(
            height < 32,
            "Can't use U32 position map when height of tree exceeds 31"
        );
        let rng = rng_maker();
        Self {
            oram: OC::create(size >> Self::L, stash_size, rng_maker),
            height,
            rng,
            _value_size: Default::default(),
        }
    }
}

impl<ValueSize, O, R> PositionMap for ORAMU32PositionMap<ValueSize, O, R>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8>,
    O: ORAM<ValueSize> + Send + Sync + 'static,
    R: RngCore + CryptoRng + Send + Sync + 'static,
{
    fn len(&self) -> u64 {
        self.oram.len() << Self::L
    }
    fn write(&mut self, key: &u64, new_val: &u64) -> (u64, Choice) {
        let new_val = *new_val as u32;
        let upper_key = *key >> Self::L;
        let lower_key = (*key as u32) & ((1u32 << Self::L) - 1);

        let mut old_val = self.oram.access(upper_key, |block, _counter| -> u32 {
            let mut old_val = 0u32;
            let u32_slice = block.as_mut_ne_u32_slice();
            for idx in 0..(1u32 << Self::L) {
                old_val.cmov(idx.ct_eq(&lower_key), &u32_slice[idx as usize]);
                (&mut u32_slice[idx as usize]).cmov(idx.ct_eq(&lower_key), &new_val);
            }
            old_val
        });
        // if old_val is zero, sample a random leaf
        let cond = old_val.ct_eq(&0);
        old_val.cmov(
            cond,
            &1u32.random_child_at_height(self.height, &mut self.rng),
        );
        (old_val as u64, cond)
    }

    fn persist(&mut self, lifetime_id: u64, new_snapshot_id: u64, volatile: bool) {
        self.oram.persist(lifetime_id, new_snapshot_id, volatile);
    }
}

/// Creates U32 Position Maps, either the trivial one or recursively on top of
/// ORAMs. The value size times the Z value determines the size of an ORAM
/// bucket
pub struct U32PositionMapCreator<
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + 'static,
    R: RngCore + CryptoRng + Send + Sync + 'static,
    OC: ORAMCreator<ValueSize, R>,
> {
    _value: PhantomData<fn() -> ValueSize>,
    _rng: PhantomData<fn() -> R>,
    _oc: PhantomData<fn() -> OC>,
}

impl<
        ValueSize: ArrayLength<u8> + PartialDiv<U8> + 'static,
        R: RngCore + CryptoRng + Send + Sync + 'static,
        OC: ORAMCreator<ValueSize, R>,
    > PositionMapCreator<R> for U32PositionMapCreator<ValueSize, R, OC>
{
    fn create<M: 'static + FnMut() -> R>(
        size: u64,
        height: u32,
        stash_size: usize,
        rng_maker: &mut M,
    ) -> Box<dyn PositionMap + Send + Sync + 'static> {
        if size <= POS_MAP_THRESHOLD {
            Box::new(TrivialPositionMap::<R>::new(size, height, rng_maker))
        } else if height <= 31 {
            Box::new(
                ORAMU32PositionMap::<ValueSize, OC::Output, R>::new::<OC, M>(
                    size, height, stash_size, rng_maker,
                ),
            )
        } else {
            panic!(
                "height = {}, but we didn't implement u64 position map yet",
                height
            )
        }
    }
}
