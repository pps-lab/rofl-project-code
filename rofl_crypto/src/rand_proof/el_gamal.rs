#![allow(non_snake_case)]

use curve25519_dalek::constants::RISTRETTO_BASEPOINT_COMPRESSED;
use curve25519_dalek::constants::RISTRETTO_BASEPOINT_POINT;
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::{Identity, MultiscalarMul};

use bulletproofs::PedersenGens;

use super::util::read32;
use core::fmt::Debug;
use core::ops::{Add, AddAssign, Mul};
use std::iter::Sum;
use serde::de::Visitor;
use serde::{self, Deserialize, Deserializer, Serialize, Serializer};
use sha3::Sha3_512;

use super::errors::*;

// ElGamalGens -------------------------------------------------

#[derive(Copy, Clone)]
pub struct ElGamalGens {
    /// Base for the committed value
    pub B: RistrettoPoint,
    /// Base for the blinding factor
    pub B_blinding: RistrettoPoint,
}

impl Default for ElGamalGens {
    fn default() -> Self {
        ElGamalGens {
            B: RISTRETTO_BASEPOINT_POINT,
            B_blinding: RistrettoPoint::hash_from_bytes::<Sha3_512>(
                RISTRETTO_BASEPOINT_COMPRESSED.as_bytes(),
            ),
        }
    }
}

impl ElGamalGens {
    pub fn from_pedersen_gens(pc_gens: &PedersenGens) -> ElGamalGens {
        ElGamalGens {
            B: pc_gens.B,
            B_blinding: pc_gens.B_blinding,
        }
    }

    pub fn from_pedersen(self, pedersen: &RistrettoPoint, randomness: &Scalar) -> ElGamalPair {
        ElGamalPair {
            L: pedersen.clone(),
            R: (self.B_blinding * randomness),
        }
    }

    pub fn commit(&self, value: Scalar, blinding: Scalar) -> ElGamalPair {
        ElGamalPair {
            L: RistrettoPoint::multiscalar_mul(&[value, blinding], &[self.B, self.B_blinding]),
            R: (&blinding * &self.B),
        }
    }

    pub fn complete_existing(&self, value: RistrettoPoint, blinding: Scalar) -> ElGamalPair {
        ElGamalPair {
            L: value,
            R: (&blinding * &self.B),
        }
    }
}

// ElGamalPair -------------------------------------------------

#[derive(PartialEq, Eq, Copy, Clone)]
pub struct ElGamalPair {
    /// g^m*h^r
    pub L: RistrettoPoint,
    /// g^r
    pub R: RistrettoPoint,
}

impl ElGamalPair {
    pub fn unity() -> Self {
        ElGamalPair {
            L: RISTRETTO_BASEPOINT_POINT,
            R: RISTRETTO_BASEPOINT_POINT,
        }
    }

    pub fn identity() -> Self {
        ElGamalPair {
            L: RistrettoPoint::identity(),
            R: RistrettoPoint::identity(),
        }
    }

    pub fn serialized_size() -> usize {
        2 * 32
    }

    pub fn right_elem_is_unity(&self) -> bool {
        self.R == RISTRETTO_BASEPOINT_POINT
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::serialized_size());
        buf.extend_from_slice(self.L.compress().as_bytes());
        buf.extend_from_slice(self.R.compress().as_bytes());
        buf
    }

    pub fn from_bytes(slice: &[u8]) -> Result<ElGamalPair, ElGamalPairError> {
        if slice.len() != 2 * 32 {
            return Err(ElGamalPairError::FormatError);
        }
        let L = CompressedRistretto(read32(&slice[0..]))
            .decompress()
            .ok_or(ElGamalPairError::FormatError)?;
        let R = CompressedRistretto(read32(&slice[32..]))
            .decompress()
            .ok_or(ElGamalPairError::FormatError)?;
        Ok(ElGamalPair { L: L, R: R })
    }
}

// ---- Basic arithmetic -------

impl<'a, 'b> Add<&'b ElGamalPair> for &'a ElGamalPair {
    type Output = ElGamalPair;

    fn add(self, other: &'b ElGamalPair) -> ElGamalPair {
        ElGamalPair {
            L: self.L + other.L,
            R: self.R + other.R,
        }
    }
}

impl<'b> AddAssign<&'b ElGamalPair> for ElGamalPair {
    fn add_assign(&mut self, _rhs: &ElGamalPair) {
        self.L += _rhs.L;
        self.R += _rhs.R;
    }
}

impl Add<ElGamalPair> for ElGamalPair {
    type Output = ElGamalPair;

    fn add(self, other: ElGamalPair) -> ElGamalPair {
        ElGamalPair {
            L: self.L + other.L,
            R: self.R + other.R,
        }
    }
}

impl<'a, 'b> Mul<&'b Scalar> for &'a ElGamalPair {
    type Output = ElGamalPair;
    fn mul(self, scalar: &'b Scalar) -> ElGamalPair {
        ElGamalPair {
            L: self.L * scalar,
            R: self.R * scalar,
        }
    }
}

impl<'a, 'b> Mul<&'b ElGamalPair> for &'a Scalar {
    type Output = ElGamalPair;
    fn mul(self, eg_pair: &'b ElGamalPair) -> ElGamalPair {
        ElGamalPair {
            L: self * eg_pair.L,
            R: self * eg_pair.R,
        }
    }
}

// Implement Sum trait for ElGamalPair
impl<'a, 'b> Sum<&'b ElGamalPair> for ElGamalPair {
    fn sum<I: Iterator<Item = &'b ElGamalPair>>(iter: I) -> ElGamalPair {
        iter.fold(ElGamalPair::identity(), |acc, x| acc + *x)
    }
}
impl Sum<ElGamalPair> for ElGamalPair {
    fn sum<I: Iterator<Item = ElGamalPair>>(iter: I) -> ElGamalPair {
        iter.fold(ElGamalPair::identity(), |acc, x| acc + x)
    }
}
// impl<T> Sum<T> for ElGamalPair
//     where
//         T: Borrow<ElGamalPair>
// {
//     fn sum<I>(iter: I) -> Self
//         where
//             I: Iterator<Item = T>
//     {
//         iter.fold(ElGamalPair::unity(), |acc, item| acc + item.borrow())
//     }
// }

impl Debug for ElGamalPair {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        write!(f, "ElGamalPair: {:?}", self.to_bytes())
    }
}

// ---- Serde -------

impl Serialize for ElGamalPair {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_bytes(&self.to_bytes()[..])
    }
}

impl<'de> Deserialize<'de> for ElGamalPair {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ElGamalPairVisitor;

        impl<'de> Visitor<'de> for ElGamalPairVisitor {
            type Value = ElGamalPair;

            fn expecting(&self, formatter: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                formatter.write_str("a valid RangeProof")
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<ElGamalPair, E>
            where
                E: serde::de::Error,
            {
                ElGamalPair::from_bytes(v).map_err(serde::de::Error::custom)
            }
        }
        deserializer.deserialize_bytes(ElGamalPairVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::{deserialize, serialize};

    #[test]
    fn test_elgamalpair_serde_rountrip() {
        let mut rng = rand::thread_rng();
        let x_val: Scalar = Scalar::random(&mut rng);
        let x_blinding: Scalar = Scalar::random(&mut rng);
        let eg_gens = ElGamalGens::default();

        let x_eg: ElGamalPair = eg_gens.commit(x_val, x_blinding);
        let x_eg_ser: Vec<u8> = serialize(&x_eg).unwrap();
        let x_eg_des: ElGamalPair = deserialize(&x_eg_ser).unwrap();
        assert_eq!(x_eg, x_eg_des);
    }

    #[test]
    fn test_eg_addition() {
        let mut rng = rand::thread_rng();
        let m1 = Scalar::random(&mut rng);
        let r1 = Scalar::random(&mut rng);
        let m2 = Scalar::random(&mut rng);
        let r2 = Scalar::random(&mut rng);

        let eg_gens = ElGamalGens::default();
        let c_a = eg_gens.commit(m1 + m2, r1 + r2);
        let c_b = eg_gens.commit(m1, r1) + eg_gens.commit(m2, r2);
        assert_eq!(c_a, c_b);
    }

    #[test]
    fn test_eg_scalar_multiplication() {
        let mut rng = rand::thread_rng();
        let m = Scalar::random(&mut rng);
        let r = Scalar::random(&mut rng);
        let c = Scalar::random(&mut rng);

        let eg_gens = ElGamalGens::default();
        let c_a = eg_gens.commit(c * m, c * r);
        let c_b = &c * &eg_gens.commit(m, r);
        let c_c = &eg_gens.commit(m, r) * &c;
        assert_eq!(c_a, c_b);
        assert_eq!(c_a, c_c);
    }
}
