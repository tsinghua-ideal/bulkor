use crate::{CipherType, KeySize, NonceSize, StorageBlockSize};
use std::convert::TryInto;

use aes::cipher::{generic_array::GenericArray, NewCipher, StreamCipher};
use aligned_cmov::{
    typenum::{PartialDiv, Prod, Unsigned, U16, U64, U8},
    A64Bytes, A8Bytes, ArrayLength, AsAlignedChunks, AsNeSlice, CMov,
};
use blake2::{digest::Digest, Blake2b};
use rand_core::{CryptoRng, RngCore};

lazy_static! {
    /// The key should be set by attestation, and probably be different from the key used in ORAM storage
    pub static ref QUERY_KEY: GenericArray<u8, KeySize> = GenericArray::<u8, KeySize>::default();
}

//query = nonce + hash + client_id + query_id_per_client + op_type + idx + data
pub const QUERY_SIZE: usize = NonceSize::USIZE + 16 + 8 + 8 + 1 + 8 + StorageBlockSize::USIZE;
//answer = nonce + hash + client_id + query_id_per_client + data
pub const ANSWER_SIZE: usize = NonceSize::USIZE + 16 + 8 + 8 + StorageBlockSize::USIZE;

#[derive(Clone)]
pub struct Query<ValueSize>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
{
    pub op_type: u8,
    pub idx: u64,
    pub new_val: A64Bytes<ValueSize>,
}

impl<ValueSize> Query<ValueSize>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
{
    /// encrypt the query to be sent to enclave
    pub fn to_slice(&self, bytes: &mut [u8]) {
        let ms = NonceSize::USIZE + 16 + 8 + 8; //meta size
        bytes[ms] = self.op_type;
        (&mut bytes[(ms + 1)..(ms + 9)]).copy_from_slice(&self.idx.to_ne_bytes());
        (&mut bytes[(ms + 9)..(ms + 9 + ValueSize::USIZE)]).copy_from_slice(&self.new_val);
    }
}

pub fn extract_client_id(bytes: &[u8]) -> u64 {
    assert!(bytes.len() == QUERY_SIZE || bytes.len() == ANSWER_SIZE);
    let pos = NonceSize::USIZE + 16;
    let mut client_id_buf = [0; 8];
    client_id_buf.copy_from_slice(&bytes[pos..(pos + 8)]);
    u64::from_ne_bytes(client_id_buf)
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

//stream decipher, ct including nonce and tag
pub fn s_decrypt(key: &GenericArray<u8, KeySize>, ct: &mut [u8], skip_enc: usize) {
    let ns = NonceSize::USIZE;
    let h: Hash = (&ct[ns..(ns + 16)]).try_into().unwrap();
    assert!(h == compute_slices_hash(key, &ct[(ns + 16)..]));
    let nonce = GenericArray::from_slice(&ct[..ns]);
    let mut cipher = CipherType::new(key, nonce);
    cipher.apply_keystream(&mut ct[(ns + 16 + skip_enc)..]);
}

/// A hash computed by "compute_block_hash"
pub type Hash = [u8; 16];

// Compute hash for slices
pub fn compute_slices_hash(hash_key: &GenericArray<u8, KeySize>, s: &[u8]) -> Hash {
    let mut hasher = Blake2b::new();
    hasher.update("encryption");
    hasher.update(hash_key);
    hasher.update(s);
    let result = hasher.finalize();
    result[..16].try_into().unwrap()
}

pub fn a64_bytes<N: ArrayLength<u8>>(src: u8) -> A64Bytes<N> {
    let mut result = A64Bytes::<N>::default();
    for byte in result.iter_mut() {
        *byte = src;
    }
    result
}
