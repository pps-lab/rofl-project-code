#![allow(dead_code)]
#![allow(non_snake_case)]

use clear_on_drop::clear::Clear;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;

use super::super::rand_proof::el_gamal::{ElGamalGens, ElGamalPair};
use super::errors::*;
use super::pedersen::SquareRandProofCommitments;
use crate::commit;
use crate::conversion32::square;
use bulletproofs::range_proof_mpc::messages::BitCommitment;
use bulletproofs::PedersenGens;

pub struct PartyExisting {}

impl PartyExisting {
    pub fn new<'a>(
        eg_gens: &'a ElGamalGens,
        m: Scalar,
        m_com: RistrettoPoint,
        r_1: Scalar,
        r_2: Scalar,
    ) -> Result<
        (
            PartyAwaitingChallenge,
            SquareRandProofCommitments,
            SquareRandProofCommitments,
        ),
        ProofError,
    > {
        // Construct corresponding pedersen generators
        let c_eg: ElGamalPair = eg_gens.complete_existing(m_com, r_1);
        let p_gens = PedersenGens {
            B: eg_gens.B,
            B_blinding: eg_gens.B_blinding,
        };

        let m_sq = m * m; // TODO lhidde: Can this overflow?
        let c_p: RistrettoPoint = p_gens.commit(m_sq, r_2);

        let mut rng = rand::thread_rng();

        let m_prime: Scalar = Scalar::random(&mut rng);
        let r_1_prime: Scalar = Scalar::random(&mut rng);
        let r_2_prime: Scalar = Scalar::random(&mut rng);

        let c_prime: ElGamalPair = eg_gens.commit(m_prime, r_1_prime);

        let p_gens_prime = PedersenGens {
            B: c_eg.L,
            B_blinding: eg_gens.B_blinding,
        };
        let c_sq_prime: RistrettoPoint = p_gens_prime.commit(m_prime, r_2_prime);

        let commitments = SquareRandProofCommitments { c: c_eg, c_sq: c_p };
        let commitments_prime = SquareRandProofCommitments {
            c: c_prime,
            c_sq: c_sq_prime,
        };

        Ok((
            PartyAwaitingChallenge {
                eg_gens: eg_gens,
                m: m,
                r_1: r_1,
                r_2: r_2,

                m_prime: m_prime,
                r_1_prime: r_1_prime,
                r_2_prime: r_2_prime,
            },
            commitments,
            commitments_prime,
        ))
    }
}

pub struct Party {}

impl Party {
    pub fn new<'a>(
        eg_gens: &'a ElGamalGens,
        m: Scalar,
        r_1: Scalar,
        r_2: Scalar,
    ) -> Result<
        (
            PartyAwaitingChallenge,
            SquareRandProofCommitments,
            SquareRandProofCommitments,
        ),
        ProofError,
    > {
        // Construct corresponding pedersen generators
        let c_eg: ElGamalPair = eg_gens.commit(m, r_1);
        let p_gens = PedersenGens {
            B: eg_gens.B,
            B_blinding: eg_gens.B_blinding,
        };

        let m_sq = m * m; // TODO lhidde: Can this overflow?
        let c_p: RistrettoPoint = p_gens.commit(m_sq, r_2);

        let mut rng = rand::thread_rng();

        let m_prime: Scalar = Scalar::random(&mut rng);
        let r_1_prime: Scalar = Scalar::random(&mut rng);
        let r_2_prime: Scalar = Scalar::random(&mut rng);

        let c_prime: ElGamalPair = eg_gens.commit(m_prime, r_1_prime);

        let p_gens_prime = PedersenGens {
            B: c_eg.L,
            B_blinding: eg_gens.B_blinding,
        };
        let c_sq_prime: RistrettoPoint = p_gens_prime.commit(m_prime, r_2_prime);

        let commitments = SquareRandProofCommitments { c: c_eg, c_sq: c_p };
        let commitments_prime = SquareRandProofCommitments {
            c: c_prime,
            c_sq: c_sq_prime,
        };

        Ok((
            PartyAwaitingChallenge {
                eg_gens: eg_gens,
                m: m,
                r_1: r_1,
                r_2: r_2,

                m_prime: m_prime,
                r_1_prime: r_1_prime,
                r_2_prime: r_2_prime,
            },
            commitments,
            commitments_prime,
        ))
    }
}

pub struct PartyAwaitingChallenge<'a> {
    eg_gens: &'a ElGamalGens,
    m: Scalar,
    r_1: Scalar,
    r_2: Scalar,

    m_prime: Scalar,
    r_1_prime: Scalar,
    r_2_prime: Scalar,
}

impl<'a> PartyAwaitingChallenge<'a> {
    pub fn apply_challenge(self, c: Scalar) -> (Scalar, Scalar, Scalar) {
        let z_m: Scalar = self.m_prime + (self.m * c);
        let z_r_1: Scalar = self.r_1_prime + (self.r_1 * c);
        let z_r_2: Scalar = self.r_2_prime + (self.r_2 - (self.m * self.r_1)) * c;

        (z_m, z_r_1, z_r_2)
    }
}

impl<'a> Drop for PartyAwaitingChallenge<'a> {
    fn drop(&mut self) {
        self.m.clear();
        self.r_1.clear();
        self.r_2.clear();
        self.m_prime.clear();
        self.r_1_prime.clear();
        self.r_2_prime.clear();
    }
}
