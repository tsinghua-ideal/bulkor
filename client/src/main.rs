use tokio::io::{self, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
#[macro_use]
extern crate lazy_static;
use aes::{cipher::NewCipher, Aes128Ctr};
use aligned_cmov::{
    typenum::{Unsigned, U1024},
    A64Bytes,
};
use rand_core::{CryptoRng, RngCore, SeedableRng};
use rand_hc::Hc128Rng;

mod query;
use query::{a64_bytes, s_decrypt, s_encrypt, Query, ANSWER_SIZE, QUERY_KEY, QUERY_SIZE};

/// Cipher type. Anything implementing StreamCipher and NewCipher at 128
/// bit security should be acceptable
type CipherType = Aes128Ctr;
/// Parameters of the cipher as typedefs (which eases syntax)
type NonceSize = <CipherType as NewCipher>::NonceSize;
type KeySize = <CipherType as NewCipher>::KeySize;
type RngType = Hc128Rng;

type StorageBlockSize = U1024;

#[tokio::main]
async fn main() -> io::Result<()> {
    let socket = TcpStream::connect("127.0.0.1:3333").await?;
    let (mut rd, mut wr) = io::split(socket);

    //TODO: it needs to change per client
    let client_id: u64 = 1;

    // Write data in the background
    let write_task = tokio::spawn(async move {
        let mut rng = RngType::from_seed([7u8; 32]);
        let mut query_id = 0;

        let expected_queries = vec![
            (0 as u64, 1),
            (0, 2),
            (0, 3),
            (2, 4),
            (2, 5),
            (0, 6),
            (0, 7),
            (9, 8),
            (2, 10),
            (0, 11),
            (9, 12),
        ]
        .into_iter()
        .map(|(idx, data)| (idx, a64_bytes::<StorageBlockSize>(data)))
        .collect::<Vec<_>>();

        wr.write_all(&client_id.to_ne_bytes()).await?;

        for each in expected_queries {
            let query = build_query(client_id, query_id, each.0, each.1, &mut rng);
            println!("query id = {:?}", query_id);
            wr.write_all(&query).await?;
            query_id += 1;
        }

        // Sometimes, the rust type inferencer needs
        // a little help
        Ok::<_, io::Error>(())
    });

    let mut buf = vec![0; ANSWER_SIZE];
    let mut answer_id = 0;
    let expected_answers = vec![0, 1, 2, 0, 4, 3, 6, 0, 5, 7, 8]
        .into_iter()
        .map(|x| a64_bytes::<StorageBlockSize>(x))
        .collect::<Vec<_>>();

    loop {
        if answer_id >= expected_answers.len() {
            break;
        }
        let r = rd.read_exact(&mut buf).await;
        if r.is_err() {
            break;
        }

        let ms = NonceSize::USIZE + 16 + 8 + 8;
        let skip_enc = 16;
        s_decrypt(&QUERY_KEY, &mut buf, skip_enc);
        println!("answer_id = {:?}", answer_id);
        assert_eq!(expected_answers[answer_id].as_slice(), &buf[ms..]);
        answer_id += 1;
    }

    write_task.abort();

    Ok(())
}

fn build_query(
    client_id: u64,
    query_id: u64,
    idx: u64,
    data: A64Bytes<StorageBlockSize>,
    rng: &mut RngType,
) -> Vec<u8> {
    let query = Query::<StorageBlockSize> {
        op_type: 1,
        idx,
        new_val: data.clone(),
    };
    let mut bytes = vec![0; QUERY_SIZE];
    let pos = NonceSize::USIZE + 16;
    (&mut bytes[pos..(pos + 8)]).copy_from_slice(&client_id.to_ne_bytes());
    (&mut bytes[(pos + 8)..(pos + 16)]).copy_from_slice(&query_id.to_ne_bytes());
    query.to_slice(&mut bytes);
    let skip_enc = 16;
    s_encrypt(&QUERY_KEY, &mut bytes, skip_enc, rng);
    bytes
}
