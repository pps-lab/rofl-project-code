#![allow(dead_code)]
#![allow(non_snake_case)]

use clear_on_drop::clear::Clear;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use crate::conversion32::exponentiate;
use super::types::CompressedRandProofCommitments;

use crate::rand_proof::el_gamal::{ElGamalGens, ElGamalPair};
use super::errors::*;

pub struct PartyExisting {}

impl PartyExisting {
    pub fn new<'a>(
        eg_gens: &'a ElGamalGens,
        m: Vec<Scalar>,
        m_com: Vec<RistrettoPoint>,
        r: Vec<Scalar>,
    ) -> Result<(PartyAwaitingChallenge, CompressedRandProofCommitments, ElGamalPair), ProofError> {
        let c: Vec<ElGamalPair> = m_com.iter().zip(&r)
            .map(|(m_i, r_i)| eg_gens.complete_existing(m_i.clone(), r_i.clone())).collect();

        let mut rng = rand::thread_rng();

        let m_prime: Scalar = Scalar::random(&mut rng);
        let r_prime: Scalar = Scalar::random(&mut rng);

        let c_prime: ElGamalPair = eg_gens.commit(m_prime, r_prime);

        Ok((
            PartyAwaitingChallenge {
                eg_gens: eg_gens,
                m: m,
                r: r,
                m_prime: m_prime,
                r_prime: r_prime,
            },
            CompressedRandProofCommitments { c_vec: c },
            c_prime,
        ))
    }
}

pub struct Party {}

impl Party {
    pub fn new<'a>(
        eg_gens: &'a ElGamalGens,
        m: Vec<Scalar>,
        r: Vec<Scalar>,
    ) -> Result<(PartyAwaitingChallenge, CompressedRandProofCommitments, ElGamalPair), ProofError> {
        let c: Vec<ElGamalPair> = m.iter().zip(&r)
            .map(|(m_i, r_i)| eg_gens.commit(m_i.clone(), r_i.clone())).collect();

        // let c: Vec<ElGamalPair> = vec![eg_gens.commit(m.get(0).unwrap().clone(), r.get(0).unwrap().clone())];

        let mut rng = rand::thread_rng();

        let m_prime: Scalar = Scalar::random(&mut rng);
        let r_prime: Scalar = Scalar::random(&mut rng);

        let c_prime: ElGamalPair = eg_gens.commit(m_prime, r_prime);

        Ok((
            PartyAwaitingChallenge {
                eg_gens: eg_gens,
                m: m,
                r: r,
                m_prime: m_prime,
                r_prime: r_prime,
            },
            CompressedRandProofCommitments { c_vec: c },
            c_prime,
        ))
    }
}

pub struct PartyAwaitingChallenge<'a> {
    eg_gens: &'a ElGamalGens,
    m: Vec<Scalar>,
    r: Vec<Scalar>,
    m_prime: Scalar,
    r_prime: Scalar,
}

impl<'a> PartyAwaitingChallenge<'a> {
    pub fn apply_challenge(self, c: Scalar) -> (Scalar, Scalar) {
        // apply c^i iteratively to elements of m where i is the index
        // TODO: POW
        let z_m: Scalar = self.m_prime + self.m.iter().enumerate().map(|(i, m)| m.clone() * exponentiate(&c, i+1)).sum::<Scalar>();
        let z_r: Scalar = self.r_prime + self.r.iter().enumerate().map(|(i, m)| m.clone() * exponentiate(&c, i+1)).sum::<Scalar>();

        (z_m, z_r)
    }
}

impl<'a> Drop for PartyAwaitingChallenge<'a> {
    fn drop(&mut self) {
        self.m.clear();
        self.r.clear();
        self.m_prime.clear();
        self.r_prime.clear();
    }
}
