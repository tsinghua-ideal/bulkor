use crate::{CipherType, KeySize, NonceSize, StorageBlockSize};
use aes::cipher::{generic_array::GenericArray as CipherGenericArray, NewCipher, StreamCipher};
use aligned_cmov::{
    typenum::{PartialDiv, Prod, Unsigned, U16, U64, U8},
    A64Bytes, A8Bytes, ArrayLength, AsAlignedChunks, AsNeSlice, CMov,
};

lazy_static! {
    /// The key should be set by attestation, and probably be different from the key used in ORAM storage
    static ref QUERY_KEY: CipherGenericArray<u8, KeySize> = CipherGenericArray::<u8, KeySize>::default();
}

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
    pub fn encrypt_to(&self) -> Vec<u8> {
        let ns = NonceSize::USIZE;
        let mut bytes = vec![1u8; ns];  //insecure, the nonce should be changed in production
        bytes.push(self.op_type);
        bytes.extend_from_slice(&self.idx.to_le_bytes());
        bytes.extend_from_slice(&self.new_val);
        let mut aes_nonce = CipherGenericArray::<u8, NonceSize>::default();
        aes_nonce[0..ns].copy_from_slice(&bytes[0..ns]);
        let mut cipher = CipherType::new(&QUERY_KEY, &aes_nonce);
        cipher.apply_keystream(&mut bytes[ns..(ns + 1)]); //op_type
        cipher.apply_keystream(&mut bytes[(ns + 1)..(ns + 9)]); //idx
        cipher.apply_keystream(&mut bytes[(ns + 9)..(ns + 9 + ValueSize::USIZE)]); // block
        bytes
    }
}

/// input: nonce + encrypted data, output: slice of decrypted data
pub fn decrypt_res(bytes: &mut [u8]) -> &[u8] {
    let mut aes_nonce = CipherGenericArray::<u8, NonceSize>::default();
    let ns = NonceSize::USIZE;
    aes_nonce[0..ns].copy_from_slice(&bytes[0..ns]);
    let mut cipher = CipherType::new(&QUERY_KEY, &aes_nonce);
    cipher.apply_keystream(&mut bytes[ns..(ns + StorageBlockSize::USIZE)]);
    &bytes[ns..]
}

pub fn a64_bytes<N: ArrayLength<u8>>(src: u8) -> A64Bytes<N> {
    let mut result = A64Bytes::<N>::default();
    for byte in result.iter_mut() {
        *byte = src;
    }
    result
}
