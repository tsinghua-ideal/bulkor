// Copyright (c) 2018-2021 The MobileCoin Foundation

//! Some generic tests that exercise objects implementing these traits

use super::ORAM;
use aligned_cmov::{subtle::Choice, typenum::U8, A64Bytes, A8Bytes, Aligned, ArrayLength};
use rand_core::{CryptoRng, RngCore};
use std::{
    collections::{btree_map::Entry, BTreeMap},
    vec::Vec,
};

/// Exercise an ORAM by writing, reading, and rewriting, a progressively larger
/// set of random locations
pub fn exercise_oram<BlockSize, O, R>(mut num_rounds: usize, oram: &mut O, rng: &mut R)
where
    BlockSize: ArrayLength<u8>,
    O: ORAM<BlockSize>,
    R: RngCore + CryptoRng,
{
    let len = oram.len();
    assert!(len != 0, "len is zero");
    assert_eq!(len & (len - 1), 0, "len is not a power of two");
    let mut expected = BTreeMap::<u64, A64Bytes<BlockSize>>::default();
    let mut probe_positions = Vec::<u64>::new();
    let mut probe_idx = 0usize;

    while num_rounds > 0 {
        if probe_idx >= probe_positions.len() {
            probe_positions.push(rng.next_u64() & (len - 1));
            probe_idx = 0;
        }
        let query = probe_positions[probe_idx];
        let expected_ent = expected.entry(query).or_default();

        oram.access(query, |val, counter| {
            assert_eq!(val, expected_ent);
            rng.fill_bytes(val);
            *counter += 1;
            expected_ent.clone_from_slice(val.as_slice());
        });

        probe_idx += 1;
        num_rounds -= 1;
    }
}

/// Exercise an ORAM by writing, reading, and rewriting, all locations
/// consecutively
pub fn exercise_oram_consecutive<BlockSize, O, R>(mut num_rounds: usize, oram: &mut O, rng: &mut R)
where
    BlockSize: ArrayLength<u8>,
    O: ORAM<BlockSize>,
    R: RngCore + CryptoRng,
{
    let len = oram.len();
    assert!(len != 0, "len is zero");
    assert_eq!(len & (len - 1), 0, "len is not a power of two");
    let mut expected = BTreeMap::<u64, A64Bytes<BlockSize>>::default();

    while num_rounds > 0 {
        let query = num_rounds as u64 & (len - 1);
        let expected_ent = expected.entry(query).or_default();

        oram.access(query, |val, counter| {
            assert_eq!(val, expected_ent);
            rng.fill_bytes(val);
            *counter += 1;
            expected_ent.clone_from_slice(val.as_slice());
        });

        num_rounds -= 1;
    }
}
