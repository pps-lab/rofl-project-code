#![allow(non_snake_case)]
use core::fmt::Debug;
use curve25519_dalek_ng::ristretto::RistrettoPoint;
use curve25519_dalek_ng::scalar::Scalar;
use merlin::Transcript;
use rayon::prelude::*;
use serde::de::Visitor;
use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

use self::dealer::*;
use self::party::*;
use self::util::read32;
use super::rand_proof::transcript::TranscriptProtocol;
use self::types::CompressedRandProofCommitments;

pub mod constants;
mod dealer;
mod party;
pub mod types;
mod util;
mod unique_u8_triplets;
pub use super::rand_proof::el_gamal::{ElGamalGens, ElGamalPair};
mod errors;
pub use self::errors::ProofError;
use crate::compressed_rand_proof::constants::{
    LABEL_CHALLENGE_SCALAR, LABEL_COMMIT_PRIME_ELGAMAL,
    label_commit_real_elgamal, LABEL_RESPONSE_R,
    LABEL_RESPONSE_Z_M,
};
use crate::conversion32::{exponentiate, f32_to_scalar_vec, precompute_exponentiate};

#[derive(PartialEq, Clone)]
pub struct CompressedRandProof {
    C_prime: ElGamalPair,
    Z_m: Scalar,
    Z_r: Scalar
}

/**
 * Implements a RandProof as well as a commitment to the square of m
 */
impl CompressedRandProof {
    pub fn prove(
        eg_gens: &ElGamalGens,
        transcript: &mut Transcript,
        m_vec: Vec<Scalar>,
        r_vec: Vec<Scalar>,
    ) -> Result<(CompressedRandProof, CompressedRandProofCommitments), ProofError> {
        let (party, c_vec, c_prime) = Party::new(&eg_gens, m_vec, r_vec)?;
        let dealer = Dealer::new(eg_gens, transcript, &c_vec);

        let (dealer, challenge) = dealer.receive_commitment(c_prime)?;
        let (z_m, z_r) = party.apply_challenge(challenge);

        let rand_proof = dealer.receive_challenge_response(z_m, z_r)?;
        Ok((rand_proof, c_vec))
    }

    pub fn prove_existing(
        eg_gens: &ElGamalGens,
        transcript: &mut Transcript,
        m: Vec<Scalar>,
        m_com: Vec<RistrettoPoint>,
        r: Vec<Scalar>,
    ) -> Result<(CompressedRandProof, CompressedRandProofCommitments), ProofError> {
        let (party, c_vec, c_prime) = PartyExisting::new(&eg_gens, m, m_com, r)?;
        let dealer = Dealer::new(eg_gens, transcript, &c_vec);

        let (dealer, challenge) = dealer.receive_commitment(c_prime)?;
        let (z_m, z_r) = party.apply_challenge(challenge);

        let rand_proof = dealer.receive_challenge_response(z_m, z_r)?;
        Ok((rand_proof, c_vec))
    }

    pub fn verify(
        &self,
        eg_gens: &ElGamalGens,
        transcript: &mut Transcript,
        c_vec: CompressedRandProofCommitments,
    ) -> Result<(), ProofError> {
        transcript.rand_proof_domain_sep();
        for (i, c) in c_vec.c_vec.iter().enumerate() {
            transcript.commit_eg_point(label_commit_real_elgamal(i), &c);
        }
        transcript.commit_eg_point(LABEL_COMMIT_PRIME_ELGAMAL, &self.C_prime);
        let challenge: Scalar = transcript.challenge_scalar(LABEL_CHALLENGE_SCALAR);
        transcript.commit_scalar(LABEL_RESPONSE_Z_M, &self.Z_m);
        transcript.commit_scalar(LABEL_RESPONSE_R, &self.Z_r);

        let precomputation_table: Vec<Scalar> = precompute_exponentiate(&challenge, c_vec.c_vec.len()+1);
        // let precomputation_table: Vec<Scalar> = challenge.precompute_exponentiate(c_vec.c_vec.len()+1);

        let dst_eg_pair: ElGamalPair = eg_gens.commit(self.Z_m, self.Z_r);
        // TODO: pow
        let src_eg_pair: ElGamalPair = self.C_prime + c_vec.c_vec.par_iter().enumerate().map(|(i, m)| m * &precomputation_table[i+1]).sum();
        if dst_eg_pair != src_eg_pair {
            return Err(ProofError::VerificationError);
        }

        Ok(())
    }

    pub fn serialized_size() -> usize {
        4 * 32
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(CompressedRandProof::serialized_size());
        buf.extend_from_slice(&self.C_prime.to_bytes());
        buf.extend_from_slice(self.Z_m.as_bytes());
        buf.extend_from_slice(self.Z_r.as_bytes());
        buf
    }

    pub fn from_bytes(slice: &[u8]) -> Result<CompressedRandProof, ProofError> {
        if slice.len() != CompressedRandProof::serialized_size() {
            return Err(ProofError::FormatError);
        }
        let C_prime_opt = ElGamalPair::from_bytes(&slice[0..2 * 32]);
        let Z_m_opt = Scalar::from_canonical_bytes(read32(&slice[2 * 32..3 * 32]));
        let Z_r_opt = Scalar::from_canonical_bytes(read32(&slice[3 * 32..4 * 32]));
        if C_prime_opt.is_err() || Z_m_opt.is_none() || Z_r_opt.is_none() {
            return Err(ProofError::FormatError);
        }

        Ok(CompressedRandProof {
            C_prime: C_prime_opt.unwrap(),
            Z_m: Z_m_opt.unwrap(),
            Z_r: Z_r_opt.unwrap(),
        })
    }

    pub fn helper_prove(m_vec: &Vec<f32>,
                    r_vec: Vec<Scalar>) -> Result<(CompressedRandProof, CompressedRandProofCommitments), ProofError> {
        let eg_gens = ElGamalGens::default();
        let mut prove_transcript = Transcript::new(b"CompressedRandProof");
        let m_vec_scalar = f32_to_scalar_vec(m_vec);
        return CompressedRandProof::prove(&eg_gens, &mut prove_transcript, m_vec_scalar, r_vec);
    }
    pub fn helper_prove_existing(m_vec: &Vec<f32>, m_com: Vec<RistrettoPoint>, r_vec: Vec<Scalar>)
        -> Result<(CompressedRandProof, CompressedRandProofCommitments), ProofError> {
        let eg_gens = ElGamalGens::default();
        let mut prove_transcript = Transcript::new(b"CompressedRandProof");
        let m_vec_scalar = f32_to_scalar_vec(m_vec);
        return CompressedRandProof::prove_existing(&eg_gens, &mut prove_transcript, m_vec_scalar, m_com, r_vec);
    }
    pub fn helper_verify(
        &self,
        c_vec: Vec<ElGamalPair>,
    ) -> Result<(), ProofError> {
        let eg_gens = ElGamalGens::default();
        let mut prove_transcript = Transcript::new(b"CompressedRandProof");
        let c_vec_commitments = CompressedRandProofCommitments {
            c_vec: c_vec,
        };
        return self.verify(&eg_gens, &mut prove_transcript, c_vec_commitments);
    }
}

impl Serialize for CompressedRandProof {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
    {
        serializer.serialize_bytes(&self.to_bytes()[..])
    }
}

impl<'de> Deserialize<'de> for CompressedRandProof {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
    {
        struct ComressedRandProofVisitor;

        impl<'de> Visitor<'de> for ComressedRandProofVisitor {
            type Value = CompressedRandProof;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                formatter.write_str("a valid CompressedRandProof")
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<CompressedRandProof, E>
                where
                    E: serde::de::Error,
            {
                CompressedRandProof::from_bytes(v).map_err(serde::de::Error::custom)
            }
        }

        deserializer.deserialize_bytes(ComressedRandProofVisitor)
    }
}

impl Eq for CompressedRandProof {}

impl Debug for CompressedRandProof {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        write!(f, "CompressedRandProof: {:?}", self.to_bytes())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use bincode;
    use curve25519_dalek_ng::scalar::Scalar;
    use crate::conversion32::{f32_to_scalar, scalar_to_f32, square};
    use crate::fp::{Fix, N_BITS, read_from_bytes};


    #[test]
    fn test_scalar_multiply_fp() {
        let num_bits = Fix::frac_nbits();
        let correction = f32::powf(2.0, -(num_bits as f32));

        let val: f32 = 1.0;
        let exp: f32 = 2.0;
        let val_scalar = f32_to_scalar(&val);
        let value = val_scalar * f32_to_scalar(&exp) * f32_to_scalar(&correction);
        let zero_float = 0.0;
        let zero_scalar = f32_to_scalar(&zero_float);
        println!(" {:?} ", N_BITS);
        println!("in Scalar: {:?}", scalar_to_f32(&value));
        println!("in Scalar val_scalar: {:?}", scalar_to_f32(&val_scalar));

        println!("in f32: {:?}", val.powf(exp));
    }

    #[test]
    fn test_serde_randproof_roundtrip() {
        let eg_gens = ElGamalGens::default();
        let mut transcript = Transcript::new(b"test_serde");
        let mut rng = rand::thread_rng();
        let n_proofs = 10000;
        let m = (0..n_proofs).map(|_| Scalar::random(&mut rng)).collect::<Vec<_>>();
        let r = (0..n_proofs).map(|_| Scalar::random(&mut rng)).collect::<Vec<_>>();
        let randproof = CompressedRandProof::prove(&eg_gens, &mut transcript, m, r).unwrap();
        let randproof_ser = bincode::serialize(&randproof).unwrap();
        let randproof_des = bincode::deserialize(&randproof_ser).unwrap();
        assert_eq!(randproof, randproof_des);
    }
    //
    #[test]
    fn test_randproof_roundtrip() {
        let eg_gens = ElGamalGens::default();
        let mut prove_transcript = Transcript::new(b"CompressedRandProof");
        let mut rng = rand::thread_rng();
        let m = vec![Scalar::random(&mut rng), Scalar::random(&mut rng), Scalar::random(&mut rng)];
        let r = vec![Scalar::random(&mut rng), Scalar::random(&mut rng), Scalar::random(&mut rng)];

        let (rand_proof, c) = CompressedRandProof::prove(&eg_gens, &mut prove_transcript, m, r).unwrap();
        let mut verify_transcript = Transcript::new(b"CompressedRandProof");
        let res = rand_proof.verify(&eg_gens, &mut verify_transcript, c);
        assert!(res.is_ok());
    }

    // #[test]
    // fn test_compressed_exponentiate() {
    //     let scalar = Scalar::one() + Scalar::one();
    //     let res = scalar.exponentiate(10);
    //     let conv  = read_from_bytes(&(res).to_bytes());
    //     println!("Conf {:?}", conv);
    // }

    #[test]
    fn test_fake_randproof() {
        let eg_gens = ElGamalGens::default();
        let mut prove_transcript = Transcript::new(b"test_serde");
        let mut rng = rand::thread_rng();
        let m = vec![Scalar::random(&mut rng), Scalar::random(&mut rng), Scalar::random(&mut rng)];
        let r = vec![Scalar::random(&mut rng), Scalar::random(&mut rng), Scalar::random(&mut rng)];
        let (rand_proof, _) = CompressedRandProof::prove(&eg_gens, &mut prove_transcript, m.clone(), r.clone()).unwrap();
        let r_fake = vec![Scalar::random(&mut rng), Scalar::random(&mut rng), Scalar::random(&mut rng)];
        if r_fake == r {
            println!("Problem detected");
            panic!();
        }
        let c_fake_egs: Vec<ElGamalPair> = m.iter().zip(r_fake).map(|(m, r_fake)| eg_gens.commit(*m, r_fake)).collect();
        let c_fake = CompressedRandProofCommitments { c_vec: c_fake_egs };
        let mut verify_transcript = Transcript::new(b"test_serde");
        let res = rand_proof.verify(&eg_gens, &mut verify_transcript, c_fake);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), ProofError::VerificationError);
    }
}
