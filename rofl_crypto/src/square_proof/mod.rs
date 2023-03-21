#![allow(non_snake_case)]
use core::fmt::Debug;
use bulletproofs::PedersenGens;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;
use serde::de::Visitor;
use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

use self::dealer::*;
use self::party::*;
use super::square_rand_proof::util::read32;
use super::rand_proof::transcript::TranscriptProtocol;

pub mod constants;
mod dealer;
mod party;
pub mod pedersen;
pub use super::rand_proof::el_gamal::{ElGamalGens, ElGamalPair};
mod errors;
pub use self::errors::ProofError;
use crate::square_rand_proof::constants::{
    LABEL_CHALLENGE_SCALAR, LABEL_COMMIT_PRIME_ELGAMAL, LABEL_COMMIT_PRIME_PEDERSEN,
    LABEL_COMMIT_REAL_ELGAMAL, LABEL_COMMIT_REAL_PEDERSEN, LABEL_RESPONSE_R_1, LABEL_RESPONSE_R_2,
    LABEL_RESPONSE_Z_M,
};
use crate::square_proof::pedersen::{SquareProofCommitments};
use crate::square_rand_proof::pedersen::{PedersenCommitment};

#[derive(PartialEq, Clone)]
pub struct SquareProof {
    C_prime: SquareProofCommitments,
    Z_m: Scalar,
    Z_r_1: Scalar,
    Z_r_2: Scalar,
}

/**
 * Implements a RandProof as well as a commitment to the square of m
 */
impl SquareProof {
    pub fn prove_existing(
        ped_gens: &PedersenGens,
        transcript: &mut Transcript,
        m: Scalar,
        m_com: RistrettoPoint,
        r_1: Scalar,
        r_2: Scalar,
    ) -> Result<(SquareProof, SquareProofCommitments), ProofError> {
        let (party, c, c_prime) = PartyExisting::new(&ped_gens, m, m_com, r_1, r_2)?;
        let dealer = Dealer::new(ped_gens, transcript, c);

        let (dealer, challenge) = dealer.receive_commitment(c_prime)?;
        let (z_m, z_r_1, z_r_2) = party.apply_challenge(challenge);

        let rand_proof = dealer.receive_challenge_response(z_m, z_r_1, z_r_2)?;
        Ok((rand_proof, c))
    }

    pub fn prove(
        ped_gens: &PedersenGens,
        transcript: &mut Transcript,
        m: Scalar,
        r_1: Scalar,
        r_2: Scalar,
    ) -> Result<(SquareProof, SquareProofCommitments), ProofError> {
        let (party, c, c_prime) = Party::new(&ped_gens, m, r_1, r_2)?;
        let dealer = Dealer::new(ped_gens, transcript, c);

        let (dealer, challenge) = dealer.receive_commitment(c_prime)?;
        let (z_m, z_r_1, z_r_2) = party.apply_challenge(challenge);

        let rand_proof = dealer.receive_challenge_response(z_m, z_r_1, z_r_2)?;
        Ok((rand_proof, c))
    }

    pub fn verify(
        &self,
        ped_gens: &PedersenGens,
        transcript: &mut Transcript,
        c: SquareProofCommitments,
    ) -> Result<(), ProofError> {
        transcript.rand_proof_domain_sep();
        transcript.commit_ped_point(LABEL_COMMIT_REAL_ELGAMAL, &c.c_l);
        transcript.commit_ped_point(LABEL_COMMIT_REAL_PEDERSEN, &c.c_sq);
        transcript.commit_ped_point(LABEL_COMMIT_PRIME_ELGAMAL, &self.C_prime.c_l);
        transcript.commit_ped_point(LABEL_COMMIT_PRIME_PEDERSEN, &self.C_prime.c_sq);

        let challenge: Scalar = transcript.challenge_scalar(LABEL_CHALLENGE_SCALAR);
        transcript.commit_scalar(LABEL_RESPONSE_Z_M, &self.Z_m);
        transcript.commit_scalar(LABEL_RESPONSE_R_1, &self.Z_r_1);
        transcript.commit_scalar(LABEL_RESPONSE_R_2, &self.Z_r_2);

        // elgamal
        let dst_eg_pair: RistrettoPoint = ped_gens.commit(self.Z_m, self.Z_r_1);
        let src_eg_pair: RistrettoPoint = &self.C_prime.c_l + &(&challenge * &c.c_l);
        if dst_eg_pair != src_eg_pair {
            return Err(ProofError::VerificationError);
        }

        // Pedersen
        let eg_base: PedersenCommitment = c.c_l;
        let ped_blinding: RistrettoPoint = ped_gens.B_blinding;
        let lhs_ped_pair: PedersenCommitment = (eg_base * self.Z_m) + (ped_blinding * self.Z_r_2);
        let rhs_ped_pair: PedersenCommitment = self.C_prime.c_sq + (challenge * c.c_sq);

        if lhs_ped_pair != rhs_ped_pair {
            return Err(ProofError::VerificationError);
        }

        Ok(())
    }

    pub fn serialized_size() -> usize {
        SquareProofCommitments::serialized_size() + 3 * 32
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(5 * 32);
        buf.extend_from_slice(&self.C_prime.to_bytes());
        buf.extend_from_slice(self.Z_m.as_bytes());
        buf.extend_from_slice(self.Z_r_1.as_bytes());
        buf.extend_from_slice(self.Z_r_2.as_bytes());
        buf
    }

    pub fn from_bytes(slice: &[u8]) -> Result<SquareProof, ProofError> {
        if slice.len() != 5 * 32 {
            return Err(ProofError::FormatError);
        }
        let C_prime_opt = SquareProofCommitments::from_bytes(&slice[0..2 * 32]);
        let Z_m_opt = Scalar::from_canonical_bytes(read32(&slice[2 * 32..3 * 32]));
        let Z_r_1_opt = Scalar::from_canonical_bytes(read32(&slice[3 * 32..4 * 32]));
        let Z_r_2_opt = Scalar::from_canonical_bytes(read32(&slice[4 * 32..5 * 32]));

        if C_prime_opt.is_err() || Z_m_opt.is_none() || Z_r_1_opt.is_none() || Z_r_2_opt.is_none() {
            return Err(ProofError::FormatError);
        }

        Ok(SquareProof {
            C_prime: C_prime_opt.unwrap(),
            Z_m: Z_m_opt.unwrap(),
            Z_r_1: Z_r_1_opt.unwrap(),
            Z_r_2: Z_r_2_opt.unwrap(),
        })
    }
}

impl Serialize for SquareProof {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.to_bytes()[..])
    }
}

impl<'de> Deserialize<'de> for SquareProof {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RandProofVisitor;

        impl<'de> Visitor<'de> for RandProofVisitor {
            type Value = SquareProof;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                formatter.write_str("a valid RandProof")
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<SquareProof, E>
            where
                E: serde::de::Error,
            {
                SquareProof::from_bytes(v).map_err(serde::de::Error::custom)
            }
        }

        deserializer.deserialize_bytes(RandProofVisitor)
    }
}

impl Eq for SquareProof {}

impl Debug for SquareProof {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        write!(f, "SquareProof: {:?}", self.to_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use bincode;
    use bulletproofs::PedersenGens;

    #[test]
    fn test_serde_randproof_roundtrip() {
        let ped_gens = PedersenGens::default();
        let mut transcript = Transcript::new(b"test_serde");
        let mut rng = rand::thread_rng();
        let m = Scalar::random(&mut rng);
        let r_1 = Scalar::random(&mut rng);
        let r_2 = Scalar::random(&mut rng);
        let randproof = SquareProof::prove(&ped_gens, &mut transcript, m, r_1, r_2).unwrap();
        let randproof_ser = bincode::serialize(&randproof).unwrap();
        let randproof_des = bincode::deserialize(&randproof_ser).unwrap();
        assert_eq!(randproof, randproof_des);
    }

    #[test]
    fn test_randproof_roundtrip() {
        let ped_gens = PedersenGens::default();
        let mut prove_transcript = Transcript::new(b"test_serde");
        let mut rng = rand::thread_rng();
        let m = Scalar::random(&mut rng);
        let r_1 = Scalar::random(&mut rng);
        let r_2 = Scalar::random(&mut rng);
        let (rand_proof, c) =
            SquareProof::prove(&ped_gens, &mut prove_transcript, m, r_1, r_2).unwrap();
        let mut verify_transcript = Transcript::new(b"test_serde");
        let res = rand_proof.verify(&ped_gens, &mut verify_transcript, c);
        assert!(res.is_ok());
    }

    #[test]
    fn test_fake_randproof() {
        let ped_gens = PedersenGens::default();
        let mut prove_transcript = Transcript::new(b"test_serde");
        let mut rng = rand::thread_rng();
        let m = Scalar::random(&mut rng);
        let m_sq = m * m;
        let r_1 = Scalar::random(&mut rng);
        let r_2 = Scalar::random(&mut rng);
        let (rand_proof, _) =
            SquareProof::prove(&ped_gens, &mut prove_transcript, m, r_1, r_2).unwrap();
        let r_fake = Scalar::random(&mut rng);
        if r_fake == r_2 {
            println!("YOU WON 1 BILLION DOLLARS!!!");
            panic!();
        }
        let c_fake = SquareProofCommitments {
            c_l: ped_gens.commit(m, r_1),
            c_sq: ped_gens.commit(m_sq, r_fake),
        };
        let mut verify_transcript = Transcript::new(b"test_serde");
        let res = rand_proof.verify(&ped_gens, &mut verify_transcript, c_fake);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), ProofError::VerificationError);
    }
}
