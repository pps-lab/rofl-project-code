use serde_json;

use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::ristretto::{RistrettoPoint, CompressedRistretto};
use bulletproofs::RangeProof;
use crate::rand_proof::{RandProof, ElGamalPair};
// Scalars serialized into bincode have constant size
pub const SCALAR_BINCODE_SIZE: usize = 40;
pub const RP_BINCODE_SIZE: usize = 40;


pub fn serialize_scalar_vec(rp_vec: &Vec<Scalar>) -> Vec<u8> {
    bincode::serialize(rp_vec).unwrap()
}

pub fn deserialize_scalar_vec(bytes: &[u8]) -> Vec<Scalar> {
    bincode::deserialize(bytes).unwrap()
}

pub fn serialize_rp_vec(rp_vec: &Vec<RistrettoPoint>) -> Vec<u8> {
    bincode::serialize(rp_vec).unwrap()
}

pub fn deserialize_rp_vec(bytes: &[u8]) -> Vec<RistrettoPoint> {
    bincode::deserialize(bytes).unwrap()
}

pub fn serialize_crp_vec(comp_rp_vec: &Vec<CompressedRistretto>) -> Vec<u8> {
    bincode::serialize(comp_rp_vec).unwrap()
}

pub fn deserialize_crp_vec(bytes: &[u8]) -> Vec<CompressedRistretto> {
    bincode::deserialize(bytes).unwrap()
}

pub fn serialize_crp_vec_vec(comp_rp_vec_vec: &Vec<Vec<CompressedRistretto>>) -> Vec<u8> {
    bincode::serialize(comp_rp_vec_vec).unwrap()
}

pub fn deserialize_crp_vec_vec(bytes: &[u8]) -> Vec<Vec<CompressedRistretto>> {
    bincode::deserialize(&bytes).unwrap()
}

pub fn serialize_range_proof_vec(range_proof_vec: &Vec<RangeProof>) -> Vec<u8> {
    bincode::serialize(range_proof_vec).unwrap()
}

pub fn deserialize_range_proof_vec(bytes: &[u8]) -> Vec<RangeProof> {
    bincode::deserialize(bytes).unwrap()
}

pub fn serialize_range_proof(range_proof: &RangeProof) -> Vec<u8> {
    bincode::serialize(range_proof).unwrap()
}

pub fn deserialize_range_proof(range_proof_ser: &[u8]) -> RangeProof {
    bincode::deserialize(range_proof_ser).unwrap()
}

pub fn serialize_eg_pair_vec(eg_pair_vec: &Vec<ElGamalPair>) -> Vec<u8> {
    bincode::serialize(eg_pair_vec).unwrap()
}

pub fn deserialize_eg_pair_vec(eg_pair_vec_ser: &[u8]) -> Vec<ElGamalPair> {
    bincode::deserialize(eg_pair_vec_ser).unwrap()
}

pub fn serialize_rand_proof_vec(rand_proof_vec: &Vec<RandProof>) -> Vec<u8> {
    bincode::serialize(rand_proof_vec).unwrap()
}

pub fn deserialize_rand_proof_vec(rand_proof_ser: &[u8]) -> Vec<RandProof> {
    bincode::deserialize(rand_proof_ser).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bulletproofs::PedersenGens;
    use crate::pedersen_ops::rnd_scalar_vec;
    use crate::range_proof_vec::create_rangeproof;
    use crate::fp::N_BITS;
    use crate::rand_proof::{ElGamalPair, ElGamalGens};
    use crate::rand_proof_vec::create_randproof_vec;

    #[test]
    fn test_bincode_rp_vec_roundtrip() {
        let pc_gens = PedersenGens::default();
        let a: Vec<Scalar> = rnd_scalar_vec(5);
        let a_rp: Vec<RistrettoPoint> = a.iter().map(|x| pc_gens.commit(*x, Scalar::zero())).collect();
        let a_ser: Vec<u8> = serialize_rp_vec(&a_rp);
        let a_de: Vec<RistrettoPoint> = deserialize_rp_vec(&a_ser);

        assert_eq!(a.len(), a_de.len());
        for (x, y) in a_rp.iter().zip(a_de) {
            assert_eq!(*x, y);
        }
    }

    #[test]
    fn test_bincode_serde_scalar() {
        let a: Vec<Scalar> = rnd_scalar_vec(5);
        let b: Vec<Vec<u8>> = a.iter().map(|x| bincode::serialize(x).unwrap()).collect();
        let c: Vec<Scalar> = b.iter().map(|x| bincode::deserialize(x).unwrap()).collect();
        for (x, y) in a.iter().zip(&c) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn test_bincode_range_proof_roundtrip() {
        let values: Vec<f32> = vec![-1.25, 0.5];
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (range_proof_vec, _) = create_rangeproof(&values, &blindings, N_BITS, 4).unwrap();
        let range_proof_ser: Vec<u8> = serialize_range_proof_vec(&range_proof_vec);
        let range_proof_vec_deser: Vec<RangeProof> = deserialize_range_proof_vec(&range_proof_ser);
        for (x, y) in range_proof_vec.iter().zip(&range_proof_vec_deser) {
            assert_eq!(x.to_bytes(), y.to_bytes());
        }
    }

    #[test]
    fn test_bincode_eg_pair_roundtrip() {
        let len: usize = 5;
        let values: Vec<Scalar> = rnd_scalar_vec(len);
        let blindings: Vec<Scalar> = rnd_scalar_vec(len);
        let eg_gens: ElGamalGens = ElGamalGens::default();
        let eg_pair_vec: Vec<ElGamalPair> =
        values.iter().zip(&blindings).map(|(&v, &b)| eg_gens.commit(v, b)).collect();
        let eg_pair_vec_ser: Vec<u8> = serialize_eg_pair_vec(&eg_pair_vec);
        let eg_pair_vec_des: Vec<ElGamalPair> = deserialize_eg_pair_vec(&eg_pair_vec_ser);
        assert_eq!(eg_pair_vec, eg_pair_vec_des);
    }

    #[test]
    fn test_bincode_rand_proof_roundtrip() {
        let values: Vec<f32> = vec![-1.25, 0.5];
        let blindings: Vec<Scalar> = rnd_scalar_vec(values.len());
        let (rand_proof_vec, _) = create_randproof_vec(&values, &blindings).unwrap();
        let rand_proof_vec_ser: Vec<u8> = serialize_rand_proof_vec(&rand_proof_vec);
        let rand_proof_vec_des: Vec<RandProof> = deserialize_rand_proof_vec(&rand_proof_vec_ser);
        assert_eq!(rand_proof_vec, rand_proof_vec_des);
    }
}