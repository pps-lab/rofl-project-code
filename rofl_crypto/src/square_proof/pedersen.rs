use super::super::rand_proof::ElGamalPair;
use crate::square_proof::errors::SquareProofCommitmentsError;
use crate::square_rand_proof::util::read32;
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};

use serde::de::Visitor;
use serde::{self, Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::Debug;

#[derive(PartialEq, Eq, Copy, Clone)]
pub struct SquareProofCommitments {
    // g^mh^r
    pub c_l: RistrettoPoint,
    // g^(m^2)h^r
    pub c_sq: RistrettoPoint,
}

impl SquareProofCommitments {
    pub fn serialized_size() -> usize {
        2 * 32
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(SquareProofCommitments::serialized_size());
        buf.extend_from_slice(self.c_l.compress().as_bytes());
        buf.extend_from_slice(self.c_sq.compress().as_bytes());
        buf
    }

    pub fn from_bytes(
        slice: &[u8],
    ) -> Result<SquareProofCommitments, SquareProofCommitmentsError> {
        if slice.len() != 2 * 32 {
            return Err(SquareProofCommitmentsError::FormatError);
        }
        let eg = CompressedRistretto(read32(&slice[0..1 * 32]))
            .decompress()
            .ok_or(SquareProofCommitmentsError::FormatError)?;
        let ped = CompressedRistretto(read32(&slice[1 * 32..]))
            .decompress()
            .ok_or(SquareProofCommitmentsError::FormatError)?;

        Ok(SquareProofCommitments { c_l: eg, c_sq: ped })
    }
}

// ---- Serde -------

impl Serialize for SquareProofCommitments {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.to_bytes()[..])
    }
}

impl<'de> Deserialize<'de> for SquareProofCommitments {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct SquareProofCommitmentsVisitor;

        impl<'de> Visitor<'de> for SquareProofCommitmentsVisitor {
            type Value = SquareProofCommitments;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                formatter.write_str("a valid SquaredProof")
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<SquareProofCommitments, E>
            where
                E: serde::de::Error,
            {
                SquareProofCommitments::from_bytes(v).map_err(serde::de::Error::custom)
            }
        }
        deserializer.deserialize_bytes(SquareProofCommitmentsVisitor)
    }
}

impl Debug for SquareProofCommitments {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        write!(f, "SquareProofCommitments: {:?}", self.to_bytes())
    }
}
