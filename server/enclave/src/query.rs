use std::convert::TryInto;

use crate::oram_storage::{s_decrypt, s_encrypt};
use crate::{CipherType, KeySize, NonceSize, StorageBlockSize};
use aes::cipher::{generic_array::GenericArray, NewCipher, StreamCipher};
use aligned_cmov::{
    subtle::{Choice, ConstantTimeEq, ConstantTimeLess},
    typenum::{PartialDiv, Prod, Unsigned, U16, U64, U8},
    A64Bytes, A8Bytes, ArrayLength, AsAlignedChunks, AsNeSlice, CMov,
};

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
    pub op_type: Choice,
    pub idx: u64,
    pub new_val: A64Bytes<ValueSize>,
}

impl<ValueSize> Query<ValueSize>
where
    ValueSize: ArrayLength<u8> + PartialDiv<U8> + PartialDiv<U64>,
{
    pub fn from_slice(bytes: &[u8]) -> Query<ValueSize> {
        assert_eq!(bytes.len(), QUERY_SIZE);
        let ms = NonceSize::USIZE + 16 + 8 + 8; //meta size
        let op_type = 1u8.ct_eq(&bytes[ms]); //if write, op_type = Choice(1u8)
        let idx = u64::from_ne_bytes(bytes[(ms + 1)..(ms + 9)].try_into().unwrap());
        let mut new_val: A64Bytes<ValueSize> = Default::default();
        new_val.copy_from_slice(&bytes[(ms + 9)..(ms + 9 + ValueSize::USIZE)]);
        Query {
            op_type,
            idx,
            new_val,
        }
    }

    /// decrypt the query sent by user
    pub fn decrypt_from(bytes: &mut [u8]) -> Query<ValueSize> {
        s_decrypt(&QUERY_KEY, bytes, 16);
        Query::from_slice(bytes)
    }
}
