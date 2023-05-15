use super::super::rand_proof::ElGamalPair;
use super::errors::CompressedRandProofCommitmentsError;
// use crate::square_rand_proof::util::read32;
use curve25519_dalek_ng::ristretto::{CompressedRistretto, RistrettoPoint};

use serde::de::Visitor;
use serde::{self, Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::Debug;

#[derive(PartialEq, Eq, Clone)]
pub struct CompressedRandProofCommitments {
    pub c_vec: Vec<ElGamalPair>
}

impl CompressedRandProofCommitments {
    pub fn serialized_size(&self) -> usize {
        2 * 32 * self.c_vec.len()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.serialized_size());
        for c in self.c_vec.iter() {
            buf.extend_from_slice(c.to_bytes().as_slice());
        }
        buf
    }

    pub fn from_bytes(
        slice: &[u8],
    ) -> Result<CompressedRandProofCommitments, CompressedRandProofCommitmentsError> {
        let elgamal_size = ElGamalPair::serialized_size();
        if slice.len() % elgamal_size != 0 {
            return Err(CompressedRandProofCommitmentsError::FormatError);
        }
        let num_commitments = slice.len() / elgamal_size;
        let mut out = Vec::with_capacity(num_commitments);
        for id in 0..num_commitments {
            out.push(ElGamalPair::from_bytes(&slice[id * elgamal_size..((id + 1) * elgamal_size)]).unwrap());
        }
        Ok(CompressedRandProofCommitments { c_vec: out })
    }
}

// ---- Serde -------

impl Serialize for CompressedRandProofCommitments {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.to_bytes()[..])
    }
}

impl<'de> Deserialize<'de> for CompressedRandProofCommitments {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CompressedRandProofCommitmentsVisitor;

        impl<'de> Visitor<'de> for CompressedRandProofCommitmentsVisitor {
            type Value = CompressedRandProofCommitments;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                formatter.write_str("a valid CompressedRandProofCommitments")
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<CompressedRandProofCommitments, E>
            where
                E: serde::de::Error,
            {
                CompressedRandProofCommitments::from_bytes(v).map_err(serde::de::Error::custom)
            }
        }
        deserializer.deserialize_bytes(CompressedRandProofCommitmentsVisitor)
    }
}

impl Debug for CompressedRandProofCommitments {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        write!(f, "CompressedRandProofCommitments: {:?}", self.to_bytes())
    }
}
