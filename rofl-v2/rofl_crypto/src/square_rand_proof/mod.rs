#![allow(non_snake_case)]
use core::fmt::Debug;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;
use serde::de::Visitor;
use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

use self::dealer::*;
use self::party::*;
use self::util::read32;
use super::rand_proof::transcript::TranscriptProtocol;

pub mod constants;
mod dealer;
mod party;
pub mod pedersen;
mod util;
pub use super::rand_proof::el_gamal::{ElGamalGens, ElGamalPair};
mod errors;
pub use self::errors::ProofError;
use crate::square_rand_proof::constants::{
    LABEL_CHALLENGE_SCALAR, LABEL_COMMIT_PRIME_ELGAMAL, LABEL_COMMIT_PRIME_PEDERSEN,
    LABEL_COMMIT_REAL_ELGAMAL, LABEL_COMMIT_REAL_PEDERSEN, LABEL_RESPONSE_R_1, LABEL_RESPONSE_R_2,
    LABEL_RESPONSE_Z_M,
};
use crate::square_rand_proof::pedersen::{PedersenCommitment, SquareRandProofCommitments};

#[derive(PartialEq, Clone)]
pub struct SquareRandProof {
    C_prime: SquareRandProofCommitments,
    Z_m: Scalar,
    Z_r_1: Scalar,
    Z_r_2: Scalar,
}

/**
 * Implements a RandProof as well as a commitment to the square of m
 */
impl SquareRandProof {
    pub fn prove_existing(
        eg_gens: &ElGamalGens,
        transcript: &mut Transcript,
        m: Scalar,
        m_com: RistrettoPoint,
        r_1: Scalar,
        r_2: Scalar,
    ) -> Result<(SquareRandProof, SquareRandProofCommitments), ProofError> {
        let (party, c, c_prime) = PartyExisting::new(&eg_gens, m, m_com, r_1, r_2)?;
        let dealer = Dealer::new(eg_gens, transcript, c);

        let (dealer, challenge) = dealer.receive_commitment(c_prime)?;
        let (z_m, z_r_1, z_r_2) = party.apply_challenge(challenge);

        let rand_proof = dealer.receive_challenge_response(z_m, z_r_1, z_r_2)?;
        Ok((rand_proof, c))
    }

    pub fn prove(
        eg_gens: &ElGamalGens,
        transcript: &mut Transcript,
        m: Scalar,
        r_1: Scalar,
        r_2: Scalar,
    ) -> Result<(SquareRandProof, SquareRandProofCommitments), ProofError> {
        let (party, c, c_prime) = Party::new(&eg_gens, m, r_1, r_2)?;
        let dealer = Dealer::new(eg_gens, transcript, c);

        let (dealer, challenge) = dealer.receive_commitment(c_prime)?;
        let (z_m, z_r_1, z_r_2) = party.apply_challenge(challenge);

        let rand_proof = dealer.receive_challenge_response(z_m, z_r_1, z_r_2)?;
        Ok((rand_proof, c))
    }

    pub fn verify(
        &self,
        eg_gens: &ElGamalGens,
        transcript: &mut Transcript,
        c: SquareRandProofCommitments,
    ) -> Result<(), ProofError> {
        transcript.rand_proof_domain_sep();
        transcript.commit_eg_point(LABEL_COMMIT_REAL_ELGAMAL, &c.c);
        transcript.commit_ped_point(LABEL_COMMIT_REAL_PEDERSEN, &c.c_sq);
        transcript.commit_eg_point(LABEL_COMMIT_PRIME_ELGAMAL, &self.C_prime.c);
        transcript.commit_ped_point(LABEL_COMMIT_PRIME_PEDERSEN, &self.C_prime.c_sq);

        let challenge: Scalar = transcript.challenge_scalar(LABEL_CHALLENGE_SCALAR);
        transcript.commit_scalar(LABEL_RESPONSE_Z_M, &self.Z_m);
        transcript.commit_scalar(LABEL_RESPONSE_R_1, &self.Z_r_1);
        transcript.commit_scalar(LABEL_RESPONSE_R_2, &self.Z_r_2);

        // elgamal
        let dst_eg_pair: ElGamalPair = eg_gens.commit(self.Z_m, self.Z_r_1);
        let src_eg_pair: ElGamalPair = &self.C_prime.c + &(&challenge * &c.c);

        if dst_eg_pair != src_eg_pair {
            return Err(ProofError::VerificationError);
        }

        // Pedersen
        let eg_base: PedersenCommitment = c.c.L;
        let ped_blinding: RistrettoPoint = eg_gens.B_blinding;
        let lhs_ped_pair: PedersenCommitment = (eg_base * self.Z_m) + (ped_blinding * self.Z_r_2);
        let rhs_ped_pair: PedersenCommitment = self.C_prime.c_sq + (challenge * c.c_sq);

        if lhs_ped_pair != rhs_ped_pair {
            return Err(ProofError::VerificationError);
        }

        Ok(())
    }

    pub fn serialized_size() -> usize {
        SquareRandProofCommitments::serialized_size() + 3 * 32
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(6 * 32);
        buf.extend_from_slice(&self.C_prime.to_bytes());
        buf.extend_from_slice(self.Z_m.as_bytes());
        buf.extend_from_slice(self.Z_r_1.as_bytes());
        buf.extend_from_slice(self.Z_r_2.as_bytes());
        buf
    }

    pub fn from_bytes(slice: &[u8]) -> Result<SquareRandProof, ProofError> {
        if slice.len() != 6 * 32 {
            return Err(ProofError::FormatError);
        }
        let C_prime_opt = SquareRandProofCommitments::from_bytes(&slice[0 * 32..3 * 32]);
        let Z_m_opt = Scalar::from_canonical_bytes(read32(&slice[3 * 32..4 * 32]));
        let Z_r_1_opt = Scalar::from_canonical_bytes(read32(&slice[4 * 32..5 * 32]));
        let Z_r_2_opt = Scalar::from_canonical_bytes(read32(&slice[5 * 32..6 * 32]));

        if C_prime_opt.is_err() || Z_m_opt.is_none() || Z_r_1_opt.is_none() || Z_r_2_opt.is_none() {
            return Err(ProofError::FormatError);
        }

        Ok(SquareRandProof {
            C_prime: C_prime_opt.unwrap(),
            Z_m: Z_m_opt.unwrap(),
            Z_r_1: Z_r_1_opt.unwrap(),
            Z_r_2: Z_r_2_opt.unwrap(),
        })
    }
}

impl Serialize for SquareRandProof {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.to_bytes()[..])
    }
}

impl<'de> Deserialize<'de> for SquareRandProof {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RandProofVisitor;

        impl<'de> Visitor<'de> for RandProofVisitor {
            type Value = SquareRandProof;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                formatter.write_str("a valid RandProof")
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<SquareRandProof, E>
            where
                E: serde::de::Error,
            {
                SquareRandProof::from_bytes(v).map_err(serde::de::Error::custom)
            }
        }

        deserializer.deserialize_bytes(RandProofVisitor)
    }
}

impl Eq for SquareRandProof {}

impl Debug for SquareRandProof {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        write!(f, "RandProof: {:?}", self.to_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bincode;
    use bulletproofs::PedersenGens;

    #[test]
    fn test_serde_randproof_roundtrip() {
        let eg_gens = ElGamalGens::default();
        let mut transcript = Transcript::new(b"test_serde");
        let mut rng = rand::thread_rng();
        let m = Scalar::random(&mut rng);
        let r_1 = Scalar::random(&mut rng);
        let r_2 = Scalar::random(&mut rng);
        let randproof = SquareRandProof::prove(&eg_gens, &mut transcript, m, r_1, r_2).unwrap();
        let randproof_ser = bincode::serialize(&randproof).unwrap();
        let randproof_des = bincode::deserialize(&randproof_ser).unwrap();
        assert_eq!(randproof, randproof_des);
    }

    #[test]
    fn test_randproof_roundtrip() {
        let eg_gens = ElGamalGens::default();
        let mut prove_transcript = Transcript::new(b"test_serde");
        let mut rng = rand::thread_rng();
        let m = Scalar::random(&mut rng);
        let r_1 = Scalar::random(&mut rng);
        let r_2 = Scalar::random(&mut rng);
        let (rand_proof, c) =
            SquareRandProof::prove(&eg_gens, &mut prove_transcript, m, r_1, r_2).unwrap();
        let mut verify_transcript = Transcript::new(b"test_serde");
        let res = rand_proof.verify(&eg_gens, &mut verify_transcript, c);
        assert!(res.is_ok());
    }

    #[test]
    fn test_fake_randproof() {
        let eg_gens = ElGamalGens::default();
        let p_gens = PedersenGens {
            B: eg_gens.B,
            B_blinding: eg_gens.B_blinding,
        };
        let mut prove_transcript = Transcript::new(b"test_serde");
        let mut rng = rand::thread_rng();
        let m = Scalar::random(&mut rng);
        let m_sq = m * m;
        let r_1 = Scalar::random(&mut rng);
        let r_2 = Scalar::random(&mut rng);
        let (rand_proof, _) =
            SquareRandProof::prove(&eg_gens, &mut prove_transcript, m, r_1, r_2).unwrap();
        let r_fake = Scalar::random(&mut rng);
        if r_fake == r_2 {
            println!("YOU WON 1 BILLION DOLLARS!!!");
            panic!();
        }
        let c_fake = SquareRandProofCommitments {
            c: eg_gens.commit(m, r_1),
            c_sq: p_gens.commit(m_sq, r_fake),
        };
        let mut verify_transcript = Transcript::new(b"test_serde");
        let res = rand_proof.verify(&eg_gens, &mut verify_transcript, c_fake);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), ProofError::VerificationError);
    }
}
