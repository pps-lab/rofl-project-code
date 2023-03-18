use bulletproofs::PedersenGens;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use rayon::prelude::*;
use std::ops::Add;

use crate::bsgs32::*;

pub fn commit_no_blinding_vec(scalar_vec: &Vec<Scalar>) -> Vec<RistrettoPoint> {
    let pcs = PedersenGens::default();
    let zero: Scalar = Scalar::zero();
    scalar_vec
        .par_iter()
        .map(|x| pcs.commit(*x, zero))
        .collect()
}

pub fn commit_vec(scalar_vec: &Vec<Scalar>, blinding_vec: &Vec<Scalar>) -> Vec<RistrettoPoint> {
    let pcs = PedersenGens::default();
    scalar_vec
        .par_iter()
        .zip(blinding_vec)
        .map(|(x, y)| pcs.commit(*x, *y))
        .collect()
}

pub fn default_discrete_log_vec(rp_vec: &Vec<RistrettoPoint>) -> Vec<Scalar> {
    let bsgs: BSGSTable = BSGSTable::default();
    // TODO mlei: error handling
    let scalar_vec: Vec<Scalar> = rp_vec
        .par_iter()
        .map(|x| bsgs.solve_discrete_log_with_neg(*x))
        .collect();
    scalar_vec
}

pub fn discrete_log_vec(rp_vec: &Vec<RistrettoPoint>, table_size: usize) -> Vec<Scalar> {
    let bsgs: BSGSTable = BSGSTable::new(table_size);
    // TODO mlei: error handling
    let scalar_vec: Vec<Scalar> = rp_vec
        .par_iter()
        .map(|x| bsgs.solve_discrete_log_with_neg(*x))
        .collect();
    scalar_vec
}

pub fn discrete_log_vec_table(rp_vec: &Vec<RistrettoPoint>, bsgs: &BSGSTable) -> Vec<Scalar> {
    let scalar_vec: Vec<Scalar> = rp_vec
        .par_iter()
        .map(|x| bsgs.solve_discrete_log_with_neg(*x))
        .collect();
    scalar_vec
}

/// in place addition
pub(crate) fn add_rp_vec(a_vec: &mut Vec<RistrettoPoint>, b_vec: &Vec<RistrettoPoint>) {
    assert_eq!(a_vec.len(), b_vec.len());
    a_vec.par_iter_mut().zip(b_vec).for_each(|(a, b)| *a += b)
}

pub fn add_rp_vec_vec(rp_vec_vec: &Vec<Vec<RistrettoPoint>>) -> Vec<RistrettoPoint> {
    let length: usize = rp_vec_vec[0].len();
    let mut res_vec: Vec<RistrettoPoint> = zero_rp_vec(length);

    for rp_vec in rp_vec_vec.iter() {
        add_rp_vec(&mut res_vec, rp_vec);
    }
    res_vec
}

pub fn zero_rp_vec(length: usize) -> Vec<RistrettoPoint> {
    let scalar_vec: Vec<Scalar> = vec![Scalar::zero(); length];
    commit_no_blinding_vec(&scalar_vec)
}

/// in place addition
fn add_scalar_vec(a_vec: &mut Vec<Scalar>, b_vec: &Vec<Scalar>) {
    assert_eq!(a_vec.len(), b_vec.len());
    a_vec.par_iter_mut().zip(b_vec).for_each(|(a, b)| *a += b)
}

pub fn add_scalar_vec_vec(scalar_vec_vec: &Vec<Vec<Scalar>>) -> Vec<Scalar> {
    let length: usize = scalar_vec_vec[0].len();
    let mut res_vec: Vec<Scalar> = zero_scalar_vec(length);

    for rp_vec in scalar_vec_vec.iter() {
        add_scalar_vec(&mut res_vec, rp_vec);
    }
    res_vec
}

pub fn zero_scalar_vec(length: usize) -> Vec<Scalar> {
    vec![Scalar::zero(); length]
}

pub fn compute_shifted_values_vec<T: Add<Output = T> + Copy>(
    value_vec: &Vec<T>,
    offset: &T,
) -> Vec<T> {
    value_vec.iter().map(|x| *x + *offset).collect()
}

pub fn compute_shifted_values_rp(
    rp_vec: &[RistrettoPoint],
    offset: &RistrettoPoint,
) -> Vec<RistrettoPoint> {
    rp_vec.par_iter().map(|x| x + offset).collect()
}

pub fn generate_cancelling_scalar_vec(n_vec: usize, n_dim: usize) -> Vec<Vec<Scalar>> {
    let mut scalar_vec_vec: Vec<Vec<Scalar>> = (0..n_vec).map(|_| rnd_scalar_vec(n_dim)).collect();
    let mut sum_scalar_vec: Vec<Scalar> = zero_scalar_vec(n_dim);

    for i in 0..n_vec - 1 {
        for j in 0..n_dim {
            sum_scalar_vec[j] += scalar_vec_vec[i][j];
        }
    }

    scalar_vec_vec[n_vec - 1] = sum_scalar_vec.iter().map(|x| -x).collect();
    scalar_vec_vec
}

pub fn rnd_scalar_vec(len: usize) -> Vec<Scalar> {
    let mut rng = rand::thread_rng();
    (0..len).map(|_| Scalar::random(&mut rng)).collect()
}

#[cfg(test)]
mod tests {
    use std::ops::Range;
    use super::*;
    use crate::conversion32::*;
    use crate::fp::{Fix, N_BITS};
    use rand::Rng;

    #[test]
    fn test_add_rp_vec_vec() {
        let x_vec_vec: Vec<Vec<f32>> = vec![
            vec![0.25, 1.25, -1.5],
            vec![-0.75, 1.25, -2.0],
            vec![0.5, 1.25, -3.0],
        ];

        let y_vec: Vec<f32> = vec![0.0, 3.75, -6.5];

        let x_vec_vec_scalar: Vec<Vec<Scalar>> =
            x_vec_vec.iter().map(|x| f32_to_scalar_vec(x)).collect();

        let x_vec_vec_enc: Vec<Vec<RistrettoPoint>> = x_vec_vec_scalar
            .iter()
            .map(|x| commit_no_blinding_vec(x))
            .collect();

        let sum_vec_enc: Vec<RistrettoPoint> = add_rp_vec_vec(&x_vec_vec_enc);
        let sum_vec_scalar: Vec<Scalar> = default_discrete_log_vec(&sum_vec_enc);
        let sum_vec: Vec<f32> = scalar_to_f32_vec(&sum_vec_scalar);

        for (s, y) in sum_vec.iter().zip(&y_vec) {
            assert_eq!(s, y);
        }
    }

    #[test]
    fn test_add_rp_vec() {
        let x_vec: Vec<f32> = vec![1.0, 1.25, -2.25];
        let y_vec: Vec<f32> = vec![-1.0, 1.25, -3.25];
        let z_vec: Vec<f32> = vec![0.0, 2.5, -5.5];

        let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
        let y_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&y_vec);
        let z_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&z_vec);

        let mut x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
        let x_vec_enc_old = x_vec_enc.clone();
        let y_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&y_vec_scalar);
        let z_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&z_vec_scalar);

        add_rp_vec(&mut x_vec_enc, &y_vec_enc);

        for (x, xc) in x_vec_enc.iter().zip(&x_vec_enc_old) {
            assert_ne!(x.compress(), xc.compress());
        }

        for (x, z) in x_vec_enc.iter().zip(&z_vec_enc) {
            assert_eq!(x.compress(), z.compress());
        }
    }

    #[test]
    fn test_default_discrete_loc_vec() {
        let x_vec: Vec<f32> = vec![-0.5, 1.25, 0.0];
        let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
        let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
        let y_vec_scalar: Vec<Scalar> = default_discrete_log_vec(&x_vec_enc);
        let y_vec: Vec<f32> = scalar_to_f32_vec(&y_vec_scalar);
        for (x, y) in x_vec.iter().zip(&y_vec) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn test_addition_rp_vec() {
        let x_vec: Vec<f32> = vec![1.0, 1.25, -2.25];
        let y_vec: Vec<f32> = vec![-1.0, 1.25, -3.25];
        let z_vec: Vec<f32> = vec![0.0, 2.5, -5.5];

        let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
        let y_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&y_vec);
        let z_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&z_vec);

        let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
        let y_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&y_vec_scalar);
        let z_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&z_vec_scalar);

        let sum_vec_enc: Vec<RistrettoPoint> = x_vec_enc
            .iter()
            .zip(&y_vec_enc)
            .map(|(x, y)| (x + y))
            .collect();

        for (s, z) in sum_vec_enc.iter().zip(&z_vec_enc) {
            assert_eq!(s.compress(), z.compress());
        }
    }

    #[test]
    fn test_generate_cancelling_scalar_vec() {
        let n_vec: usize = 10;
        let n_dim: usize = 12;

        let scalar_vec_vec: Vec<Vec<Scalar>> = generate_cancelling_scalar_vec(n_vec, n_dim);
        assert_eq!(scalar_vec_vec.len(), n_vec);
        assert_eq!(scalar_vec_vec[0].len(), n_dim);

        let mut sum_vec: Vec<Scalar> = zero_scalar_vec(n_dim);
        for i in 0..n_vec {
            for j in 0..n_dim {
                sum_vec[j] += scalar_vec_vec[i][j];
            }
        }

        let zero: Scalar = Scalar::zero();
        for i in 0..n_dim {
            assert_eq!(sum_vec[i], zero);
        }
    }

    #[test]
    fn test_generate_cancelling_scalar_vec_commited() {
        let n_vec: usize = 10;
        let n_dim: usize = 12;

        let blinding_scalar_vec_vec: Vec<Vec<Scalar>> =
            generate_cancelling_scalar_vec(n_vec, n_dim);
        assert_eq!(blinding_scalar_vec_vec.len(), n_vec);
        assert_eq!(blinding_scalar_vec_vec[0].len(), n_dim);

        let value_scalar_vec_vec: Vec<Vec<Scalar>> = (0..n_vec)
            .map(|_| rnd_scalar_vec_in_fp_range(n_vec, n_dim))
            .collect();

        let rp_vec_vec: Vec<Vec<RistrettoPoint>> = value_scalar_vec_vec
            .iter()
            .zip(&blinding_scalar_vec_vec)
            .map(|(x, y)| commit_vec(x, y))
            .collect();

        let sum_rp_vec: Vec<RistrettoPoint> = add_rp_vec_vec(&rp_vec_vec);
        let sum_scalar_vec: Vec<Scalar> = default_discrete_log_vec(&sum_rp_vec);

        let sum_scalar_vec_target: Vec<Scalar> = add_scalar_vec_vec(&value_scalar_vec_vec);

        for (x, y) in sum_scalar_vec.iter().zip(&sum_scalar_vec_target) {
            assert_eq!(x, y);
        }
    }

    #[test]
    fn test_discrete_log_full_precomp_for_8bit() {
        if N_BITS != 8 {
            return;
        }
        let n_values: usize = 64;
        let mut rng = rand::thread_rng();
        let (fp_min, fp_max) = get_clip_bounds(N_BITS);
        let x_vec: Vec<f32> = (0..n_values)
            .map(|_| rng.gen_range::<f32, Range<f32>>(fp_min..fp_max))
            .collect();
        let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
        let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
        let y_vec_scalar: Vec<Scalar> = discrete_log_vec(&x_vec_enc, N_BITS);
        for (x, y) in x_vec_scalar.iter().zip(&y_vec_scalar) {
            assert_eq!(x, y);
        }
    }

    // Note mlei: purpose of scale is to scale down the interval to avoid overflow when adding multple fixed precision values
    pub fn rnd_scalar_vec_in_fp_range(len: usize, scale: usize) -> Vec<Scalar> {
        let mut rng = rand::thread_rng();
        let max: f32 = Fix::max_value().to_float::<f32>() / (scale as f32);
        let min = -max;
        let rnd_f32_vec: Vec<f32> = (0..len).map(|_| rng.gen_range(min..max)).collect();
        let rnd_scalar_vec: Vec<Scalar> = f32_to_scalar_vec(&rnd_f32_vec);
        rnd_scalar_vec
    }
}
