// Copyright (c) 2018-2021 The MobileCoin Foundation

//! Traits for different pieces of ORAM, from the level of block storage up to
//! an oblivious map.
//! These are all defined in terms of fixed-length chunks of bytes and the
//! A8Bytes object from the aligned-cmov crate.
//!
//! There is also a naive implementation of the ORAM storage object for tests.

use std::vec::Vec;

// Re-export some traits we depend on, so that downstream can ensure that they
// have the same version as us.
pub use aligned_cmov::{
    cswap, subtle, typenum::Unsigned, A64Bytes, A8Bytes, ArrayLength, CMov, GenericArray,
};
pub use rand_core::{CryptoRng, RngCore};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

mod naive_storage;
pub use naive_storage::{HeapORAMStorage, HeapORAMStorageCreator};

mod linear_scanning;
pub use linear_scanning::LinearScanningORAM;

mod creators;
pub use creators::*;

pub mod testing;

/// Represents trusted block storage holding aligned blocks of memory of a
/// certain size. This is a building block for ORAM.
///
/// This object is required to encrypt / mac the memory if it pushes things out
/// to untrusted, but it is not required to keep the indices a secret when
/// accessed. This object is not itself an oblivious data structure.
///
/// In tests this can simply be Vec.
/// In production it is planned to be an object that makes OCalls to untrusted,
/// and which encrypts and macs the memory blocks that it sends to and from
/// untrusted. This is analogous to the "Intel memory engine" in SGX.
///
/// It is anticipated that "tree-top caching" occurs at this layer, so the
/// initial portion of the storage is in the enclave and the rest is in
/// untrusted
///
/// TODO: Create an API that allows checking out from two branches
/// simultaneously.
#[allow(clippy::len_without_is_empty)]
pub trait ORAMStorage<BlockSize: ArrayLength<u8>, MetaSize: ArrayLength<u8>, Z: Unsigned> {
    /// Get the number of blocks represented by this block storage
    /// This is also the bound of the largest valid index
    fn len(&self) -> u64;

    /// Checkout all blocks on the branch leading to a particular index in the
    /// tree, copying them and their metadata into two scratch buffers.
    ///
    /// Arguments:
    /// * index: The index of the leaf, a u64 TreeIndex value.
    /// * dest: The destination data buffer
    /// * dest_meta: The destination metadata buffer
    ///
    /// Requirements:
    /// * 0 < index <= len
    /// * index.height() + 1 == dest.len() == dest_meta.len()
    /// * It is illegal to checkout while there is an existing checkout.
    fn checkout(
        &mut self,
        index: u64,
        dest: &mut [A64Bytes<BlockSize>],
        dest_meta: &mut [A8Bytes<MetaSize>],
    );

    /// Checkin a number of blocks, copying them and their metadata
    /// from two scratch buffers.
    ///
    /// It is illegal to checkin when there is not an existing checkout.
    /// It is illegal to checkin different blocks than what was checked out.
    ///
    /// Arguments:
    /// * index: The index of the leaf, a u64 TreeIndex.
    /// * src: The source data buffer
    /// * src_meta: The source metadata buffer
    ///
    /// Note: src and src_meta are mutable, because it is more efficient to
    /// encrypt them in place than to copy them and then encrypt.
    /// These buffers are left in an unspecified but valid state.
    fn checkin(
        &mut self,
        index: u64,
        src: &mut [A64Bytes<BlockSize>],
        src_meta: &mut [A8Bytes<MetaSize>],
    );

    /// This is the API for persisting the all cached parts, such as treetop
    /// Arguments:
    /// * new_snapshot_id: the newest version of snapshot would be
    /// * volatile: if the snapshot is stored to untrusted memory solely, or on disk
    /// * rng: since storages themselves do not have rng, they rely on oram engine to provide it
    fn persist<Rng: RngCore + CryptoRng>(
        &mut self,
        lifetime_id: u64,
        new_snapshot_id: u64,
        volatile: bool,
        rng: &mut Rng,
    );
}

/// An Oblivious RAM -- that is, an array like [A8Bytes<ValueSize>; N]
/// which supports access queries *without memory access patterns revealing
/// what indices were queried*. (Here, N is a runtime parameter set at
/// construction time.)
///
/// The ValueSize parameter indicates the number of bytes in a stored value.
///
/// The key-type here is always u64 even if it "could" be smaller.
/// We think that if keys are actually stored as u32 or u16 in some of the
/// recursive position maps, that conversion can happen at a different layer of
/// the system.
///
/// TODO: Should there be, perhaps, a separate trait for "resizable" ORAMs?
/// We don't have a good way for the OMAP to take advantage of that right now.
#[allow(clippy::len_without_is_empty)]
#[allow(clippy::upper_case_acronyms)]
pub trait ORAM<ValueSize: ArrayLength<u8>> {
    /// Get the number of values logically in the ORAM.
    /// This is also one more than the largest index that can be legally
    /// accessed.
    fn len(&self) -> u64;

    /// Access the ORAM at a position, calling a lambda with the recovered
    /// value, and returning the result of the lambda.
    /// This cannot fail, but will panic if index is out of bounds.
    ///
    /// This is the lowest-level API that we offer for getting data from the
    /// ORAM.
    fn access<T, F: FnOnce(&mut A64Bytes<ValueSize>, &mut u64) -> T>(
        &mut self,
        index: u64,
        func: F,
    ) -> T;

    /// High-level helper -- when you only need to read and don't need to write
    /// a new value, this is simpler than using `access`.
    /// In most ORAM there will not be a significantly faster implementation of
    /// this.
    #[inline]
    fn read(&mut self, index: u64) -> A64Bytes<ValueSize> {
        self.access(index, |val, counter| val.clone())
    }

    /// High-level helper -- when you need to write a value and want the
    /// previous value, but you don't need to see the previous value when
    /// deciding what to write, this is simpler than using `access`.
    /// In most ORAM there will not be a significantly faster implementation of
    /// this.
    #[inline]
    fn write(&mut self, index: u64, new_val: &A64Bytes<ValueSize>) -> A64Bytes<ValueSize> {
        self.access(index, |val, counter| {
            let retval = val.clone();
            *val = new_val.clone();
            *counter += 1;
            retval
        })
    }

    /// This is the API for persisting the ORAM metadata, including stash and pos map
    fn persist(&mut self, lifetime_id: u64, new_snapshot_id: u64, volatile: bool);
}

/// Trait that helps to debug ORAM.
/// This should only be used in tests.
///
/// This should never be called in production. IMO the best practice is that
/// implementations of this trait should be gated by `#[cfg(test)]`, or perhaps
/// `#[cfg(debug_assertions)]`.
pub trait ORAMDebug<ValueSize: ArrayLength<u8>> {
    /// Systematically check the data structure invariants, asserting that they
    /// hold. Also, produce an array representation of the logical state of
    /// the ORAM.
    ///
    /// This should not change the ORAM.
    ///
    /// This is returned so that recursive path ORAM can implement
    /// check_invariants by first asking recursive children to check their
    /// invariants.
    fn check_invariants(&self) -> Vec<A64Bytes<ValueSize>>;
}

/// PositionMap trait conceptually is an array of TreeIndex.
/// Each value in the map corresponds to a leaf in the complete binary tree,
/// at some common height.
///
/// PositionMap trait must be object-safe so that dyn PositionMap works.
/// It also only needs to work with integer types, and padding up to u64 is
/// fine. Therefore we make a new trait which is reduced and only exposes the
/// things that PathORAM needs from the position map.
///
/// TODO: API for resizing it? Changing height?
#[allow(clippy::len_without_is_empty)]
pub trait PositionMap {
    /// The number of keys in the map. The valid keys are in the range 0..len.
    fn len(&self) -> u64;
    /// Write a new value to a particular key.
    /// The new value should be a random nonce from a CSPRNG.
    /// Returns the old value.
    /// It is illegal to write to a key that is out of bounds.
    fn write(&mut self, key: &u64, new_val: &u64) -> (u64, Choice);

    /// This is the API for persisting the ORAM metadata, including stash and pos map
    fn persist(&mut self, lifetime_id: u64, new_snapshot_id: u64, volatile: bool);
}

/// Utility function for logs base 2 rounded up, implemented as const fn
#[inline]
pub const fn log2_ceil(arg: u64) -> u32 {
    if arg == 0 {
        return 0;
    }
    (!0u64).count_ones() - (arg - 1).leading_zeros()
}

#[cfg(test)]
mod test {
    use super::*;
    // Sanity check the log2_ceil function
    #[test]
    fn test_log2_ceil() {
        assert_eq!(0, log2_ceil(0));
        assert_eq!(0, log2_ceil(1));
        assert_eq!(1, log2_ceil(2));
        assert_eq!(2, log2_ceil(3));
        assert_eq!(2, log2_ceil(4));
        assert_eq!(3, log2_ceil(5));
        assert_eq!(3, log2_ceil(8));
        assert_eq!(4, log2_ceil(9));
        assert_eq!(4, log2_ceil(16));
        assert_eq!(5, log2_ceil(17));
    }
}
