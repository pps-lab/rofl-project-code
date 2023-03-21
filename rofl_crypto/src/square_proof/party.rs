#![allow(dead_code)]
#![allow(non_snake_case)]

use clear_on_drop::clear::Clear;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;

use super::super::rand_proof::el_gamal::{ElGamalGens, ElGamalPair};
use super::errors::*;
use super::pedersen::SquareProofCommitments;

use bulletproofs::PedersenGens;
use crate::square_rand_proof::pedersen::PedersenCommitment;

pub struct PartyExisting {}

impl PartyExisting {
    pub fn new<'a>(
        ped_gens: &'a PedersenGens,
        m: Scalar,
        m_com: RistrettoPoint,
        r_1: Scalar,
        r_2: Scalar,
    ) -> Result<
        (
            PartyAwaitingChallenge,
            SquareProofCommitments,
            SquareProofCommitments,
        ),
        ProofError,
    > {
        // Construct corresponding pedersen generators
        let p_base: PedersenCommitment = m_com;

        let m_sq = m * m; // TODO lhidde: Can this overflow?
        let c_p: RistrettoPoint = ped_gens.commit(m_sq, r_2);

        let mut rng = rand::thread_rng();

        let m_prime: Scalar = Scalar::random(&mut rng);
        let r_1_prime: Scalar = Scalar::random(&mut rng);
        let r_2_prime: Scalar = Scalar::random(&mut rng);

        let c_prime: PedersenCommitment = ped_gens.commit(m_prime, r_1_prime);

        let p_gens_prime = PedersenGens {
            B: p_base,
            B_blinding: ped_gens.B_blinding,
        };
        let c_sq_prime: RistrettoPoint = p_gens_prime.commit(m_prime, r_2_prime);

        let commitments = SquareProofCommitments { c_l: p_base, c_sq: c_p };
        let commitments_prime = SquareProofCommitments {
            c_l: c_prime,
            c_sq: c_sq_prime,
        };

        Ok((
            PartyAwaitingChallenge {
                ped_gens: ped_gens,
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
        ped_gens: &'a PedersenGens,
        m: Scalar,
        r_1: Scalar,
        r_2: Scalar,
    ) -> Result<
        (
            PartyAwaitingChallenge,
            SquareProofCommitments,
            SquareProofCommitments,
        ),
        ProofError,
    > {
        // Construct corresponding pedersen generators
        let c_base: RistrettoPoint = ped_gens.commit(m, r_1);

        let m_sq = m * m; // TODO lhidde: Can this overflow?
        let c_p: RistrettoPoint = ped_gens.commit(m_sq, r_2);

        let mut rng = rand::thread_rng();

        let m_prime: Scalar = Scalar::random(&mut rng);
        let r_1_prime: Scalar = Scalar::random(&mut rng);
        let r_2_prime: Scalar = Scalar::random(&mut rng);

        let c_prime: RistrettoPoint = ped_gens.commit(m_prime, r_1_prime);

        let p_gens_prime = PedersenGens {
            B: c_base,
            B_blinding: ped_gens.B_blinding,
        };
        let c_sq_prime: RistrettoPoint = p_gens_prime.commit(m_prime, r_2_prime);

        let commitments = SquareProofCommitments { c_l: c_base, c_sq: c_p };
        let commitments_prime = SquareProofCommitments {
            c_l: c_prime,
            c_sq: c_sq_prime,
        };

        Ok((
            PartyAwaitingChallenge {
                ped_gens: ped_gens,
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
    ped_gens: &'a PedersenGens,
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
