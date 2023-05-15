use super::super::rand_proof::ElGamalPair;
use crate::square_rand_proof::errors::SquareRandProofCommitmentsError;
use crate::square_rand_proof::util::read32;
use curve25519_dalek_ng::ristretto::{CompressedRistretto, RistrettoPoint};

use serde::de::Visitor;
use serde::{self, Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::Debug;

pub type PedersenCommitment = RistrettoPoint;

#[derive(PartialEq, Eq, Copy, Clone)]
pub struct SquareRandProofCommitments {
    // g^mh^r
    pub c: ElGamalPair,
    // g^(m^2)h^r
    pub c_sq: RistrettoPoint,
}

impl SquareRandProofCommitments {
    pub fn serialized_size() -> usize {
        3 * 32
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(SquareRandProofCommitments::serialized_size());
        buf.extend_from_slice(self.c.to_bytes().as_slice());
        buf.extend_from_slice(self.c_sq.compress().as_bytes());
        buf
    }

    pub fn from_bytes(
        slice: &[u8],
    ) -> Result<SquareRandProofCommitments, SquareRandProofCommitmentsError> {
        if slice.len() != 3 * 32 {
            return Err(SquareRandProofCommitmentsError::FormatError);
        }
        let eg = ElGamalPair::from_bytes(&slice[0..2 * 32]).unwrap();
        let ped = CompressedRistretto(read32(&slice[2 * 32..]))
            .decompress()
            .ok_or(SquareRandProofCommitmentsError::FormatError)?;

        Ok(SquareRandProofCommitments { c: eg, c_sq: ped })
    }
}

// ---- Serde -------

impl Serialize for SquareRandProofCommitments {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.to_bytes()[..])
    }
}

impl<'de> Deserialize<'de> for SquareRandProofCommitments {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SquareRandProofCommitmentsVisitor;

        impl<'de> Visitor<'de> for SquareRandProofCommitmentsVisitor {
            type Value = SquareRandProofCommitments;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                formatter.write_str("a valid RangeProof")
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<SquareRandProofCommitments, E>
            where
                E: serde::de::Error,
            {
                SquareRandProofCommitments::from_bytes(v).map_err(serde::de::Error::custom)
            }
        }
        deserializer.deserialize_bytes(SquareRandProofCommitmentsVisitor)
    }
}

impl Debug for SquareRandProofCommitments {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        write!(f, "SquareRandProofCommitments: {:?}", self.to_bytes())
    }
}
