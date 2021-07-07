#![allow(unused_imports)]
use byteorder::{ByteOrder, LittleEndian};
use rand::Rng;

use curve25519_dalek::scalar::Scalar;

use crate::fp::{read_from_bytes, Fix, IRawFix, URawFix, N_BITS};

// TODO mlei: quantify loss of conversion?
// TODO lhidde: Implement prob. quantization and see how it compares
pub fn f32_to_scalar(v_f32: &f32) -> Scalar {
    let v_scalar: Scalar = (Fix::saturating_from_float(v_f32.abs()).to_bits() as URawFix).into();
    if *v_f32 < 0.0 {
        -v_scalar
    } else {
        v_scalar
    }
}

pub fn f32_to_scalar_vec(values_f32: &Vec<f32>) -> Vec<Scalar> {
    values_f32.iter().map(|x| f32_to_scalar(x)).collect()
}

pub fn scalar_to_f32(v_scalar: &Scalar) -> f32 {
    let is_neg: bool = *v_scalar.as_bytes().last().unwrap() != 0u8;
    let value_uint: URawFix;
    if is_neg {
        value_uint = read_from_bytes(&(-v_scalar).to_bytes());
        -uint_to_f32(value_uint)
    } else {
        value_uint = read_from_bytes(&v_scalar.to_bytes());
        uint_to_f32(value_uint)
    }
}

pub fn scalar_to_f32_vec(values_scalar: &Vec<Scalar>) -> Vec<f32> {
    values_scalar.iter().map(|x| scalar_to_f32(x)).collect()
}

fn uint_to_f32(value_uint: URawFix) -> f32 {
    let value_fp: Fix = Fix::from_bits(value_uint as URawFix);
    value_fp.to_float()
}

pub fn uint_to_f32_vec(uint_vec: &Vec<URawFix>) -> Vec<f32> {
    uint_vec.iter().map(|x| uint_to_f32(*x)).collect()
}

pub fn f32_to_fp_vec(f32_vec: &Vec<f32>) -> Vec<URawFix> {
    f32_vec
        .iter()
        .map(|x| Fix::saturating_from_float(*x).to_bits())
        .collect()
}

pub fn get_clip_bounds(range: usize) -> (f32, f32) {
    let max: f32 = Fix::from_bits(((1u128 << range - 1) - 1) as URawFix).to_float();
    let min: f32 = -max;
    (min, max)
}

pub fn get_l2_clip_bounds(range: usize) -> f32 {
    return Fix::from_bits(((1u128 << range) - 1) as URawFix).to_float();
}

pub fn square(s1: &Scalar) -> Scalar {
    // let num_bits = Fix::frac_nbits();
    // let correction = i32::pow(2, num_bits) as f32;

    let is_neg: bool = *s1.as_bytes().last().unwrap() != 0u8;
    let value_fp: Fix;
    if is_neg {
        let value_uint = read_from_bytes(&(-s1).to_bytes());
        value_fp = Fix::from_bits(value_uint as URawFix);
    } else {
        let value_uint = read_from_bytes(&s1.to_bytes());
        value_fp = Fix::from_bits(value_uint as URawFix);
    }

    let mul: Option<Fix> = value_fp.checked_mul(value_fp);
    if mul.is_none() {
        panic!(
            "Value {:?} could not be multiplied because the result overflows {:?} bits",
            value_fp, N_BITS
        );
    }

    (mul.unwrap().to_bits() as URawFix).into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pedersen_ops::{commit_no_blinding_vec, default_discrete_log_vec};
    use curve25519_dalek::ristretto::RistrettoPoint;
    use curve25519_dalek::scalar::Scalar;
    use rand::Rng;

    #[test]
    fn test_print_fp_bit_conf() {
        // not actually a test, just print out bitconf of fp
        // only shown when run with
        // cargo test --nocapture
        println!("FP int_bits: {}", Fix::int_nbits());
        println!("FP frac_bits: {}", Fix::frac_nbits());
        println!("FP max: {:.32}", Fix::max_value());
        println!("FP min: {:.32}", Fix::min_value());
    }

    #[test]
    fn test_square() {
        let num_bits = Fix::frac_nbits();
        let correction = i32::pow(2, num_bits) as f32;

        let v_2: f32 = 12.5;
        let s_2 = f32_to_scalar(&v_2);

        assert_eq!(scalar_to_f32(&(s_2 * s_2)) / correction, v_2 * v_2);
    }

    #[test]
    fn test_square_half() {
        let s_2 = f32_to_scalar(&0.5);
        println!("Res {:?} {:?}", s_2 * s_2, scalar_to_f32(&(s_2 * s_2)));
    }

    #[test]
    fn test_square_fn() {
        let v: Vec<f32> = vec![2.0, 4.0, 2.25, 2.5, 12.5, 112.5];
        for v_2 in v {
            let s_2 = f32_to_scalar(&v_2);
            assert_eq!(scalar_to_f32(&square(&s_2)), v_2 * v_2);
        }
    }

    #[test]
    fn test_square_fn_neg() {
        let v: Vec<f32> = vec![-2.0];
        for v_2 in v {
            let s_2 = f32_to_scalar(&v_2);
            assert_eq!(scalar_to_f32(&square(&s_2)), v_2 * v_2);
        }
    }

    #[test]
    fn test_conversion_lossless() {
        let a_f32: f32 = 0.5;
        let b_f32: f32 = -1.25;
        let c_f32: f32 = Fix::max_value().to_float();

        let a_scalar: Scalar = f32_to_scalar(&a_f32);
        let b_scalar: Scalar = f32_to_scalar(&b_f32);
        let c_scalar: Scalar = f32_to_scalar(&c_f32);

        assert_eq!(a_f32, scalar_to_f32(&a_scalar));
        assert_eq!(b_f32, scalar_to_f32(&b_scalar));
        assert_eq!(c_f32, scalar_to_f32(&c_scalar));
    }

    #[test]
    fn test_conversion_f32_scalar_lossy_rounded() {
        // bigger values may lead to overflow depending
        let a_f32: f32 = (Fix::max_value().to_float::<f32>()) - 0.1;
        let b_f32: f32 = (Fix::min_value().to_float::<f32>()) + (1.0 / 3.0);

        let a_scalar = f32_to_scalar(&a_f32);
        let b_scalar = f32_to_scalar(&b_f32);

        // NOTE mlei: loss is bounded by fractional bits of fixed precision
        assert!(
            (a_f32 - scalar_to_f32(&a_scalar)).abs()
                <= 2f32.powf(-(Fix::frac_nbits() as f32) - 1.0)
        );
        assert!(
            (b_f32 - scalar_to_f32(&b_scalar)).abs()
                <= 2f32.powf(-(Fix::frac_nbits() as f32) - 1.0)
        );
    }

    #[test]
    fn test_conversion_f32_scalar_lossy_saturated() {
        let max: f32 = Fix::max_value().to_float::<f32>();
        let min: f32 = -Fix::max_value().to_float::<f32>();

        let a_f32: f32 = max + 5.0;
        let b_f32: f32 = min - 100.0;

        let a_scalar = f32_to_scalar(&a_f32);
        let b_scalar = f32_to_scalar(&b_f32);

        // NOTE mlei: loss due to f32 not fitting into (32bit) fixed point representation
        assert_eq!(max, scalar_to_f32(&a_scalar));
        assert_eq!(min, scalar_to_f32(&b_scalar));
    }

    #[test]
    fn test_commit_no_blinding_extract_value_saturated() {
        let max: f32 = Fix::max_value().to_float::<f32>();
        let min: f32 = -Fix::max_value().to_float::<f32>();
        let a_f32: f32 = max + 5.0;
        let b_f32: f32 = min - 100.0;

        let x_vec: Vec<f32> = vec![a_f32, b_f32];
        let x_vec_scalar: Vec<Scalar> = f32_to_scalar_vec(&x_vec);
        let x_vec_enc: Vec<RistrettoPoint> = commit_no_blinding_vec(&x_vec_scalar);
        let y_vec_scalar: Vec<Scalar> = default_discrete_log_vec(&x_vec_enc);
        let y_vec: Vec<f32> = scalar_to_f32_vec(&y_vec_scalar);

        assert_eq!(y_vec[0], max);
        assert_eq!(y_vec[1], min);
    }
}
