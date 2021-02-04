#![allow(unused_imports)]
use byteorder::{ByteOrder, LittleEndian};
use fixed::frac::{U0, U1, U10, U11, U12, U2, U3, U4, U5, U6, U7, U8, U9};
use fixed::{FixedI16, FixedI32, FixedI64, FixedI8, FixedU16, FixedU32, FixedU64, FixedU8};

use crate::conversion32::{f32_to_scalar_vec, scalar_to_f32_vec};

#[cfg(feature = "frac0")]
pub type Frac = U0;
#[cfg(feature = "frac1")]
pub type Frac = U1;
#[cfg(feature = "frac2")]
pub type Frac = U2;
#[cfg(feature = "frac3")]
pub type Frac = U3;
#[cfg(feature = "frac4")]
pub type Frac = U4;
#[cfg(feature = "frac5")]
pub type Frac = U5;
#[cfg(feature = "frac6")]
pub type Frac = U6;
#[cfg(feature = "frac7")]
pub type Frac = U7;
#[cfg(feature = "frac8")]
pub type Frac = U8;
#[cfg(feature = "frac9")]
pub type Frac = U9;
#[cfg(feature = "frac10")]
pub type Frac = U10;
#[cfg(feature = "frac11")]
pub type Frac = U11;
#[cfg(feature = "frac12")]
pub type Frac = U12;

#[cfg(feature = "fp8")]
mod fp_config {
    use super::*;
    pub const N_BITS: usize = 8;
    pub type Fix = FixedU8<Frac>;
    pub type IRawFix = i8;
    pub type URawFix = u8;
    pub fn read_from_bytes(x: &[u8]) -> u8 {
        *x.first().unwrap()
    }
    pub const PRECOMP_BIAS: usize = 3;
}

#[cfg(feature = "fp16")]
mod fp_config {
    use super::*;
    pub const N_BITS: usize = 16;
    pub type Fix = FixedU16<Frac>;
    pub type IRawFix = i16;
    pub type URawFix = u16;
    pub fn read_from_bytes(x: &[u8]) -> u16 {
        LittleEndian::read_u16(x)
    }
    pub const PRECOMP_BIAS: usize = 7;
}

#[cfg(feature = "fp32")]
mod fp_config {
    use super::*;
    pub const N_BITS: usize = 32;
    pub type Fix = FixedU32<Frac>;
    pub type IRawFix = i32;
    pub type URawFix = u32;
    pub fn read_from_bytes(x: &[u8]) -> u32 {
        LittleEndian::read_u32(x)
    }
    pub const PRECOMP_BIAS: usize = 6;
}

#[cfg(feature = "fp64")]
mod fp_config {
    use super::*;
    pub const N_BITS: usize = 64;
    pub type Fix = FixedU64<Frac>;
    pub type IRawFix = i64;
    pub type URawFix = u64;
    pub fn read_from_bytes(x: &[u8]) -> u64 {
        LittleEndian::read_u64(x)
    }
    pub const PRECOMP_BIAS: usize = 0;
}

#[cfg(not(feature = "frac0"))]
#[cfg(not(feature = "frac1"))]
#[cfg(not(feature = "frac2"))]
#[cfg(not(feature = "frac3"))]
#[cfg(not(feature = "frac4"))]
#[cfg(not(feature = "frac5"))]
#[cfg(not(feature = "frac6"))]
#[cfg(not(feature = "frac7"))]
#[cfg(not(feature = "frac8"))]
#[cfg(not(feature = "frac9"))]
#[cfg(not(feature = "frac10"))]
#[cfg(not(feature = "frac11"))]
#[cfg(not(feature = "frac12"))]
pub type Frac = U5; //default fractional

#[cfg(not(feature = "fp8"))]
#[cfg(not(feature = "fp16"))]
#[cfg(not(feature = "fp32"))]
#[cfg(not(feature = "fp64"))]
mod fp_config {
    //default default fixed precision
    use super::*;
    pub const N_BITS: usize = 32;
    pub type Fix = FixedU32<Frac>;
    pub type IRawFix = i32;
    pub type URawFix = u32;
    pub fn read_from_bytes(x: &[u8]) -> u32 {
        LittleEndian::read_u32(x)
    }
    pub const PRECOMP_BIAS: usize = 6;
}

pub use self::fp_config::{read_from_bytes, Fix, IRawFix, URawFix, N_BITS, PRECOMP_BIAS};
