
use super::unique_u8_triplets::UNIQUE_U8_TRIPLETS;

// We have to use this really ugly way because Merlin only allows static strings :(
pub fn label_commit_real_elgamal(index: usize) -> &'static [u8] {
    return &UNIQUE_U8_TRIPLETS[index];
}

pub static LABEL_COMMIT_PRIME_ELGAMAL: &[u8] = b"C_prime_eg";

pub static LABEL_RESPONSE_Z_M: &[u8] = b"Z_m";
pub static LABEL_RESPONSE_R: &[u8] = b"ZR";

pub static LABEL_CHALLENGE_SCALAR: &[u8] = b"c";
