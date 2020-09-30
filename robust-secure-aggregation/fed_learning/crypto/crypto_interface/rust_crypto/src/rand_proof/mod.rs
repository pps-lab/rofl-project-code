#![allow(non_snake_case)]
use curve25519_dalek::scalar::Scalar;
use merlin::Transcript;
use serde::{self, Serialize, Deserialize, Serializer, Deserializer};
use serde::de::Visitor;
use core::fmt::Debug;

use self::party::*;
use self::dealer::*;
use self::transcript::TranscriptProtocol;
use self::util::read32;

pub mod transcript;
mod dealer;
mod party;
mod util;
pub mod el_gamal;
pub use self::el_gamal::{ElGamalGens, ElGamalPair};
mod errors;
pub use self::errors::ProofError;

#[derive(PartialEq)]
pub struct RandProof {
    C_prime: ElGamalPair,
    Z_m: Scalar,
    Z_r: Scalar,
}

impl RandProof {
    pub fn prove(
        eg_gens: &ElGamalGens,
        transcript: &mut Transcript,
        m: Scalar,
        r: Scalar) -> Result<(RandProof, ElGamalPair), ProofError> {
        
        let (party, c, c_prime) = Party::new(&eg_gens, m, r)?;
        let dealer = Dealer::new(eg_gens, transcript, c);
        
        let (dealer, challenge) = dealer.receive_commitment(c_prime)?;
        let (z_m, z_r) = party.apply_challenge(challenge);

        let rand_proof = dealer.receive_challenge_response(z_m, z_r)?;
        Ok((rand_proof, c))
    }

    pub fn verify(
        &self,
        eg_gens: &ElGamalGens,
        transcript: &mut Transcript,
        c: ElGamalPair) -> Result<(), ProofError> {
        
        transcript.rand_proof_domain_sep();
        transcript.commit_eg_point(b"C", &c);
        transcript.commit_eg_point(b"C_prime", &self.C_prime);
        let challenge: Scalar = transcript.challenge_scalar(b"c");
        transcript.commit_scalar(b"Z_m", &self.Z_m);
        transcript.commit_scalar(b"Z_r", &self.Z_r);

        let dst_eg_pair: ElGamalPair = eg_gens.commit(self.Z_m, self.Z_r);
        let src_eg_pair: ElGamalPair = &self.C_prime + &(&challenge*&c);
        
        if dst_eg_pair != src_eg_pair {
            return Err(ProofError::VerificationError);
        }

        Ok(())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(4*32);
        buf.extend_from_slice(&self.C_prime.to_bytes());
        buf.extend_from_slice(self.Z_m.as_bytes());
        buf.extend_from_slice(self.Z_r.as_bytes());
        buf
    }

    pub fn from_bytes(slice: &[u8]) -> Result<RandProof, ProofError> {
        if slice.len() != 4*32 {
            return Err(ProofError::FormatError);
        }
        let C_prime_opt = ElGamalPair::from_bytes(&slice[0*32..2*32]);
        let Z_m_opt = Scalar::from_canonical_bytes(read32(&slice[2*32..3*32]));
        let Z_r_opt = Scalar::from_canonical_bytes(read32(&slice[3*32..4*32]));
        if 
            C_prime_opt.is_err() ||
            Z_m_opt.is_none() ||
            Z_r_opt.is_none() {
                return Err(ProofError::FormatError)
        }

        Ok(RandProof {
            C_prime: C_prime_opt.unwrap(),
            Z_m: Z_m_opt.unwrap(),
            Z_r: Z_r_opt.unwrap()
        })
    }
}

impl Serialize for RandProof {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.to_bytes()[..])
    }
}

impl<'de> Deserialize<'de> for RandProof {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct RandProofVisitor;

        impl<'de> Visitor<'de> for RandProofVisitor {
            type Value = RandProof;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                formatter.write_str("a valid RandProof")
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<RandProof, E>
            where
                E: serde::de::Error,
            {
                RandProof::from_bytes(v).map_err(serde::de::Error::custom)
            }
        }

        deserializer.deserialize_bytes(RandProofVisitor)
    }
}

impl Eq for RandProof {}

impl Debug for RandProof {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        write!(f, "RandProof: {:?}", self.to_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode;

    #[test]
    fn test_serde_randproof_roundtrip() {
        let eg_gens = ElGamalGens::default();
        let mut transcript = Transcript::new(b"test_serde");
        let mut rng = rand::thread_rng();
        let m = Scalar::random(&mut rng);
        let r = Scalar::random(&mut rng);
        let randproof = RandProof::prove(&eg_gens, &mut transcript, m, r).unwrap();
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
        let r = Scalar::random(&mut rng);
        let (rand_proof, c) = RandProof::prove(&eg_gens, &mut prove_transcript, m, r).unwrap();
        let mut verify_transcript = Transcript::new(b"test_serde");
        let res = rand_proof.verify(&eg_gens, &mut verify_transcript, c);
        assert!(res.is_ok());
    }

    #[test]
    fn test_fake_randproof() {
        let eg_gens = ElGamalGens::default();
        let mut prove_transcript = Transcript::new(b"test_serde");
        let mut rng = rand::thread_rng();
        let m = Scalar::random(&mut rng);
        let r = Scalar::random(&mut rng);
        let (rand_proof, _) = RandProof::prove(&eg_gens, &mut prove_transcript, m, r).unwrap();
        let r_fake = Scalar::random(&mut rng);
        if r_fake == r {
            println!("YOU WON 1 BILLION DOLLARS!!!");
            panic!();
        }
        let c_fake: ElGamalPair = eg_gens.commit(m, r_fake);
        let mut verify_transcript = Transcript::new(b"test_serde");
        let res = rand_proof.verify(&eg_gens, &mut verify_transcript, c_fake);
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), ProofError::VerificationError);
    }
}