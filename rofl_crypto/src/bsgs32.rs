#![allow(non_snake_case)]
use curve25519_dalek_ng::constants::RISTRETTO_BASEPOINT_POINT;
use curve25519_dalek_ng::ristretto::RistrettoPoint;
use curve25519_dalek_ng::scalar::Scalar;

use hashbrown::HashMap;

use crate::fp::{BSGS_URawFix, BSGS_N_BITS, PRECOMP_BIAS};
// Fix for 8 bits for L2 evaluation
// pub const N_BITS: usize = 16;
// pub type URawFix = u16;
// pub const PRECOMP_BIAS: usize = 7;

pub struct BSGSTable {
    pub table: HashMap<[u8; 32], BSGS_URawFix>,
    pub mG: RistrettoPoint,
}

impl BSGSTable {
    pub fn new(m: usize) -> BSGSTable {
        let mut tab = BSGSTable {
            table: HashMap::with_capacity(m as usize),
            mG: RISTRETTO_BASEPOINT_POINT * Scalar::from(m as BSGS_URawFix),
        };

        let mut cur = RISTRETTO_BASEPOINT_POINT;
        let id = RISTRETTO_BASEPOINT_POINT * Scalar::from(0u32);
        tab.table.insert(id.compress().to_bytes(), 0);
        for x in 1..(m + 1) {
            tab.table.insert(cur.compress().to_bytes(), x as BSGS_URawFix);
            cur += RISTRETTO_BASEPOINT_POINT;
        }
        tab
    }

    pub fn default() -> BSGSTable {
        BSGSTable::new((1 as usize) << (BSGS_N_BITS / 2 + PRECOMP_BIAS))
    }

    pub fn get_value(&self, point: RistrettoPoint) -> Option<&BSGS_URawFix> {
        self.table.get(point.compress().as_bytes())
    }

    pub fn get_size(&self) -> u64 {
        (self.table.len() as u64) - 1
    }

    pub fn solve_discrete_log(&self, M: RistrettoPoint, max_it: u64) -> Option<BSGS_URawFix> {
        let mut cur_point = M.clone();
        for i in 0..max_it {
            let cur_pow = self.get_value(cur_point);
            match cur_pow {
                Some(&pow) => return Some((i * self.get_size() + (pow as u64)) as BSGS_URawFix),
                None => cur_point -= self.mG,
            }
        }
        None
    }

    pub fn solve_discrete_log_default(&self, M: RistrettoPoint) -> Option<BSGS_URawFix> {
        return self.solve_discrete_log(M, (1u64 << BSGS_N_BITS) / self.get_size());
    }

    pub fn solve_discrete_log_with_neg(&self, M: RistrettoPoint) -> Scalar {
        let res: Option<BSGS_URawFix> = self.solve_discrete_log_default(M);
        match res {
            Some(val) => val.into(),
            None => {
                let inv_res = self.solve_discrete_log_default(-M);
                -Scalar::from(inv_res.unwrap())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use super::*;
    use crate::conversion32::f32_to_scalar_vec;
    use crate::fp::Fix;
    use crate::pedersen_ops::commit_no_blinding_vec;
    use rand::Rng;

    #[test]
    fn test_solve_discrete_log_positive() {
        let bsgs: BSGSTable = BSGSTable::default();
        let n_values: usize = 30;
        let mut rng = rand::thread_rng();

        let x_vec: Vec<f32> = (0..n_values)
            .map(|_| rng.gen_range::<f32, Range<f32>>(0.0..(Fix::max_value().to_float::<f32>())))
            .collect();
        let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
        let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
        let y_vec_scalar: Vec<Scalar> = x_vec_enc
            .iter()
            .map(|x| bsgs.solve_discrete_log_default(*x).unwrap().into())
            .collect();
        for (x, y) in x_vec_scalar.iter().zip(&y_vec_scalar) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn test_solve_discrete_log_negative() {
        let bsgs: BSGSTable = BSGSTable::default();
        let n_values: usize = 30;
        let mut rng = rand::thread_rng();

        let x_vec: Vec<f32> = (0..n_values)
            .map(|_| rng.gen_range::<f32, Range<f32>>(-Fix::max_value().to_float::<f32>()..0.0))
            .collect();
        //let x_vec: Vec<f32> = vec![Fix::max_value().to_float::<f32>()];
        let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
        let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
        let y_vec_scalar: Vec<Scalar> = x_vec_enc
            .iter()
            .map(|x| bsgs.solve_discrete_log_with_neg(*x))
            .collect();
        for (x, y) in x_vec_scalar.iter().zip(&y_vec_scalar) {
            assert_eq!(x, y);
        }
    }
}
