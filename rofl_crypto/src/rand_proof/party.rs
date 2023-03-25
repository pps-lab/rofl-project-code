#![allow(dead_code)]
#![allow(non_snake_case)]

use clear_on_drop::clear::Clear;
use curve25519_dalek_ng::ristretto::RistrettoPoint;
use curve25519_dalek_ng::scalar::Scalar;

use super::el_gamal::{ElGamalGens, ElGamalPair};
use super::errors::*;

pub struct PartyExisting {}

impl PartyExisting {
    pub fn new<'a>(
        eg_gens: &'a ElGamalGens,
        m: Scalar,
        m_com: RistrettoPoint,
        r: Scalar,
    ) -> Result<(PartyAwaitingChallenge, ElGamalPair, ElGamalPair), ProofError> {
        let c: ElGamalPair = eg_gens.complete_existing(m_com, r);
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
            c,
            c_prime,
        ))
    }
}

pub struct Party {}

impl Party {
    pub fn new<'a>(
        eg_gens: &'a ElGamalGens,
        m: Scalar,
        r: Scalar,
    ) -> Result<(PartyAwaitingChallenge, ElGamalPair, ElGamalPair), ProofError> {
        let c: ElGamalPair = eg_gens.commit(m, r);
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
            c,
            c_prime,
        ))
    }
}

pub struct PartyAwaitingChallenge<'a> {
    eg_gens: &'a ElGamalGens,
    m: Scalar,
    r: Scalar,
    m_prime: Scalar,
    r_prime: Scalar,
}

impl<'a> PartyAwaitingChallenge<'a> {
    pub fn apply_challenge(self, c: Scalar) -> (Scalar, Scalar) {
        let z_m: Scalar = self.m_prime + self.m * c;
        let z_r: Scalar = self.r_prime + self.r * c;

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
