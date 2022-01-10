use std::convert::TryInto;
use std::vec::Vec;

use crate::{CipherType, KeySize, NonceSize};
use aes::cipher::{generic_array::GenericArray as CipherGenericArray, NewCipher, StreamCipher};
use aligned_cmov::{
    subtle::{Choice, ConstantTimeEq, ConstantTimeLess},
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
    pub op_type: Choice,
    pub idx: u64,
    pub new_val: A64Bytes<ValueSize>,
}

impl<ValueSize> Query<ValueSize>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
{
    /// decrypt the query sent by user
    pub fn decrypt_from(bytes: &mut [u8]) -> Query<ValueSize> {
        let mut aes_nonce = CipherGenericArray::<u8, NonceSize>::default();
        let ns = NonceSize::USIZE;
        aes_nonce[0..ns].copy_from_slice(&bytes[0..ns]);
        let mut cipher = CipherType::new(&QUERY_KEY, &aes_nonce);
        cipher.apply_keystream(&mut bytes[ns..(ns + 1)]); //op_type
        cipher.apply_keystream(&mut bytes[(ns + 1)..(ns + 9)]); //idx
        cipher.apply_keystream(&mut bytes[(ns + 9)..(ns + 9 + ValueSize::USIZE)]); // block
        let op_type = 1u8.ct_eq(&bytes[ns]); //if write, op_type = Choice(1u8)
        let idx = u64::from_le_bytes(bytes[(ns + 1)..(ns + 9)].try_into().unwrap());
        let mut new_val: A64Bytes<ValueSize> = Default::default();
        new_val.copy_from_slice(&bytes[(ns + 9)..(ns + 9 + ValueSize::USIZE)]);
        Query {
            op_type,
            idx,
            new_val,
        }
    }

    //encrypt the query for logging
    pub fn encrypt_with_counter(&self, cur_counter: u64) -> Vec<u8> {
        let ns = NonceSize::USIZE;
        let mut bytes = vec![1u8; ns]; //insecure, the nonce should be changed in production
        bytes.push(self.op_type.unwrap_u8());
        bytes.extend_from_slice(&self.idx.to_le_bytes());
        bytes.extend_from_slice(&self.new_val);
        bytes.extend_from_slice(&cur_counter.to_le_bytes());
        let mut aes_nonce = CipherGenericArray::<u8, NonceSize>::default();
        aes_nonce[0..ns].copy_from_slice(&bytes[0..ns]);
        let mut cipher = CipherType::new(&QUERY_KEY, &aes_nonce);
        cipher.apply_keystream(&mut bytes[ns..(ns + 1)]); //op_type
        cipher.apply_keystream(&mut bytes[(ns + 1)..(ns + 9)]); //idx
        cipher.apply_keystream(&mut bytes[(ns + 9)..(ns + 9 + ValueSize::USIZE)]); // block
        cipher
            .apply_keystream(&mut bytes[(ns + 9 + ValueSize::USIZE)..(ns + 17 + ValueSize::USIZE)]); // counter
        bytes
    }
}

/// The result is encrypted in-place, the returned value is the nonce
pub fn encrypt_res<ValueSize>(res: &mut A64Bytes<ValueSize>) -> Vec<u8>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
{
    let ns = NonceSize::USIZE;
    let bytes = vec![1u8; ns]; //insecure, the nonce should be changed in production
    let mut aes_nonce = CipherGenericArray::<u8, NonceSize>::default();
    aes_nonce[0..ns].copy_from_slice(&bytes[0..ns]);
    let mut cipher = CipherType::new(&QUERY_KEY, &aes_nonce);
    cipher.apply_keystream(res); // block
    bytes
}
