use std::slice;
use std::mem;
use std::ffi::CString;
use std::ptr;
use std::fmt;

use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::ristretto::RistrettoPoint;
use bulletproofs::RangeProof;
use reduce::Reduce;

use crate::conversion32::*;
use crate::pedersen_ops::*;
use crate::serde_vec::*;
use crate::range_proof_vec;
use crate::rand_proof_vec;
use crate::square_rand_proof_vec;
use crate::l2_range_proof_vec;
use crate::rand_proof::{RandProof, ElGamalPair};

use libc::{c_void, c_char};
use crate::square_rand_proof::pedersen::SquareRandProofCommitments;
use crate::square_rand_proof::SquareRandProof;
use bincode::serialize;
use crate::square_rand_proof_vec::L2RangeProofError;
use itertools::Itertools;
use rayon::prelude::*;


#[repr(C)]
pub struct PyVec {
    pub data: *const c_void,
    pub len: usize
}

#[repr(C)]
pub struct PyRes {
    pub ret: usize,
    pub msg: *const c_char,
    pub res: *const c_void
}

#[no_mangle]
pub extern fn say_hello() -> PyVec {
    println!("Hello world");
    let x_vec: Vec<Scalar> = vec![create_x()];
    let x_vec_ser: Vec<u8> = serialize_scalar_vec(&x_vec);

    println!("RData: {:02x?}", x_vec_ser);
    let res: PyVec = PyVec{ data: x_vec_ser[..].as_ptr()  as *const c_void, len: x_vec_ser.len()};
    println!("R PyVec: {:p}", &res);
    println!("R Data: {:p}",  &x_vec_ser[..]);
    println!("R Data.as_ptr: {:p}", x_vec_ser[..].as_ptr());
    mem::forget(x_vec_ser);
    res
}

// NOTE mlei: rust raw pointers (*) should only be used in order to be compatible with
//            FFI calls.

#[no_mangle]
pub extern fn add_commitments(ptr_arr_ptr: *const *const u8, len_arr_ptr: *const usize, len: usize) -> PyVec {
    assert!(!ptr_arr_ptr.is_null() && !len_arr_ptr.is_null());
    
    let ptr_arr: &[*const u8] = unsafe { slice::from_raw_parts(ptr_arr_ptr, len as usize) };
    let len_arr: &[usize] = unsafe { slice::from_raw_parts(len_arr_ptr, len as usize) };

    let bytes_arr: Vec<&[u8]> = ptr_arr.iter()
                                       .zip(len_arr)
                                       .map(|(x, y)| unsafe { slice::from_raw_parts(*x, *y as usize) } )
                                       .collect();

    let commit_vec_vec: Vec<Vec<RistrettoPoint>> = bytes_arr.iter()
                                                            .map(|x| deserialize_rp_vec(x))
                                                            .collect();
    let sum_vec: Vec<RistrettoPoint> = add_rp_vec_vec(&commit_vec_vec);
    let sum_vec_ser: Vec<u8> = serialize_rp_vec(&sum_vec);

    create_pyvec::<u8>(sum_vec_ser)
}

// We should actually flip the definitions
#[no_mangle]
pub extern fn add_commitments_transposed(ptr_arr_ptr: *const *const u8, len_arr_ptr: *const usize, len: usize) -> PyVec {
    assert!(!ptr_arr_ptr.is_null() && !len_arr_ptr.is_null());

    let ptr_arr: &[*const u8] = unsafe { slice::from_raw_parts(ptr_arr_ptr, len as usize) };
    let len_arr: &[usize] = unsafe { slice::from_raw_parts(len_arr_ptr, len as usize) };

    let bytes_arr: Vec<&[u8]> = ptr_arr.iter()
        .zip(len_arr)
        .map(|(x, y)| unsafe { slice::from_raw_parts(*x, *y as usize) } )
        .collect();

    let sum_vec: Vec<PyVec> = bytes_arr.iter()
        .map(|x| deserialize_rp_vec(x))
        .map(|x| x.into_iter().reduce(|a, b| a + b).unwrap())
        .map(|x| create_pyvec(serialize(&x).unwrap()))
        .collect();

    create_pyvec(sum_vec)
}

// Testing purposes only
#[no_mangle]
pub extern fn commit_no_blinding(value_ptr: *const f32, len: usize) -> PyVec {
    assert!(!value_ptr.is_null());
    // convert to f32 vector
    let values_f32: &[f32] = unsafe { slice::from_raw_parts(value_ptr, len as usize) };
    // convert to Scalar vector
    let scalar_vec: Vec<Scalar> = f32_to_scalar_vec(&values_f32.to_vec());
    let rp_vec :Vec<RistrettoPoint> = commit_no_blinding_vec(&scalar_vec);
    let ser: Vec<u8> = serialize_rp_vec(&rp_vec);
    create_pyvec::<u8>(ser)
}

#[no_mangle]
pub extern fn commit(value_ptr: *const f32, value_len: usize, blinding_ptr: *const u8, blinding_len: usize) -> PyVec {
    assert!(!value_ptr.is_null());
    assert!(!blinding_ptr.is_null());

    let values_f32: &[f32] = unsafe { slice::from_raw_parts(value_ptr, value_len as usize) };
    let values_scalar_vec: Vec<Scalar> = f32_to_scalar_vec(&values_f32.to_vec());

    let bytes_arr: &[u8] = unsafe { slice::from_raw_parts(blinding_ptr, blinding_len as usize) };
    let blinding_scalar_vec = deserialize_scalar_vec(bytes_arr);

    let rp_vec: Vec<RistrettoPoint> = commit_vec(&values_scalar_vec, &blinding_scalar_vec);
    let rp_vec_ser: Vec<u8> = serialize_rp_vec(&rp_vec);

    create_pyvec(rp_vec_ser)
}

/// generates cancelling blinding vectors for n_vec > 1
/// if n_vec = 1 it just generates a random blinding vector
#[no_mangle]
pub extern fn generate_cancelling_blindings(n_vec: usize, n_dim: usize) -> PyVec{
    let blinding_scalar_vec: Vec<Vec<Scalar>> = generate_cancelling_scalar_vec(n_vec, n_dim);
    let blinding_ser_vec: Vec<Vec<u8>> = blinding_scalar_vec.iter().map(|x| serialize_scalar_vec(&x)).collect();
    let blinding_ser_pyvec_vec: Vec<PyVec> = blinding_ser_vec.iter().map(|x| create_pyvec(x.to_vec())).collect();
    let blinding_ser_pyvec_pyvec: PyVec = create_pyvec(blinding_ser_pyvec_vec);
    blinding_ser_pyvec_pyvec
}

#[no_mangle]
pub extern fn select_blindings(blinding_ptr: *const u8, blinding_len: usize, indices_ptr: *const usize, indices_len: usize) -> PyVec {
    // Selects blindings for these indices

    let value_indices: &[usize] = unsafe { slice::from_raw_parts(indices_ptr, indices_len as usize) };
    let bytes_arr: &[u8] = unsafe { slice::from_raw_parts(blinding_ptr, blinding_len as usize) };
    let blinding_scalar_vec = deserialize_scalar_vec(bytes_arr);

    let selected = value_indices.iter().map(|&index| blinding_scalar_vec[index]).collect_vec();
    let blinding_ser_vec: Vec<u8> = serialize_scalar_vec(&selected);
    let blinding_ser_pyvec_pyvec: PyVec = create_pyvec(blinding_ser_vec);
    blinding_ser_pyvec_pyvec
}
#[no_mangle]
pub extern fn select_commitments(commit_ptr: *const u8, commit_len: usize, indices_ptr: *const usize, indices_len: usize) -> PyVec {
    // Selects blindings for these indices

    let value_indices: &[usize] = unsafe { slice::from_raw_parts(indices_ptr, indices_len as usize) };
    let commit_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_ptr, commit_len as usize) };
    let rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&commit_bytes);

    let selected = value_indices.iter().map(|&index| rp_vec[index]).collect_vec();
    let rp_vec_ser: Vec<u8> = serialize_rp_vec(&selected);
    create_pyvec(rp_vec_ser)
}


#[no_mangle]
pub extern fn extract_values(bytes_ptr: *const u8, len: usize) -> PyVec {
    assert!(!bytes_ptr.is_null() );

    let bytes: &[u8] = unsafe { slice::from_raw_parts(bytes_ptr, len as usize) };
    let rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(bytes);
    let scalar_vec: Vec<Scalar> = default_discrete_log_vec(&rp_vec);
    let f32_vec: Vec<f32> = scalar_to_f32_vec(&scalar_vec);
    create_pyvec::<f32>(f32_vec)
}


/// returns a PyRes
/// upon success it contains a pointer to a two element vector
/// first element points to serialized rangeproof
/// second element points to serialized commitments
#[no_mangle]
pub extern fn create_rangeproof(value_ptr: *const f32, value_len: usize, blinding_ptr: *const u8, blinding_len: usize, range_exp: usize, n_partition: usize) -> PyRes {
    assert!(!value_ptr.is_null());
    assert!(!blinding_ptr.is_null());

    let value_f32_vec_clipped: &[f32] = unsafe { slice::from_raw_parts(value_ptr, value_len as usize) };
    let bytes_arr: &[u8] = unsafe { slice::from_raw_parts(blinding_ptr, blinding_len as usize) };
    let blinding_scalar_vec = deserialize_scalar_vec(bytes_arr);

    match range_proof_vec::create_rangeproof(&value_f32_vec_clipped.to_vec(), &blinding_scalar_vec.to_vec(), range_exp, n_partition) {
        Ok((range_proof_vec, commit_vec)) => {
            let range_proof_ser: Vec<u8> = serialize_range_proof_vec(&range_proof_vec);
            let commit_vec_ser: Vec<u8> = serialize_rp_vec(&commit_vec);
            let range_proof_pyvec = create_pyvec(range_proof_ser);
            let commit_pyvec = create_pyvec(commit_vec_ser);
            let res_vec = vec![range_proof_pyvec, commit_pyvec];
            let res_pyvec = Box::into_raw(Box::new(create_pyvec(res_vec)));
            create_pyres_from_success(res_pyvec)
        }
        Err(e) => {
            create_pyres_from_err(&e)
        }
    }
}

/// if proof is valid, pyres.res points to a zero value
#[no_mangle]
pub extern fn verify_rangeproof(commit_ptr: *const u8, commit_len: usize, proof_ptr: *const u8, proof_len: usize, range_exp: usize) -> PyRes {
    assert!(!commit_ptr.is_null());
    assert!(!proof_ptr.is_null());

    let commit_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_ptr, commit_len as usize) };
    let rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&commit_bytes);
    let range_proof_bytes: &[u8] = unsafe { slice::from_raw_parts(proof_ptr, proof_len as usize) };
    let range_proof_vec: Vec<RangeProof> = deserialize_range_proof_vec(&range_proof_bytes);

    match range_proof_vec::verify_rangeproof(&range_proof_vec, &rp_vec, range_exp) {
        Ok(v) => {
            let v_ptr = Box::into_raw(Box::new(v));
            create_pyres_from_success(v_ptr)
        }
        Err(e) => {
            create_pyres_from_err(e)
        }
    }
}


// TODO mlei: currently create_randproof expects prove range parameter due to
// clipping, as otherwise the commitments would not match from the rangeproof.
// In future, clipping should be exposed as FFI function. But right now it would
// require a change on the Python side as well for the aggregation protocols:
// rangeproofaggregator and randrangeproofaggregator
#[no_mangle]
pub extern fn create_randproof(
    value_ptr: *const f32,
    value_len: usize,
    blinding_ptr: *const u8,
    blinding_len: usize,)
    -> PyRes {
    assert!(!value_ptr.is_null());
    assert!(!blinding_ptr.is_null());

    let value_f32_vec: &[f32] = unsafe { slice::from_raw_parts(value_ptr, value_len as usize) };
    let bytes_arr: &[u8] = unsafe { slice::from_raw_parts(blinding_ptr, blinding_len as usize) };
    let blinding_scalar_vec: Vec<Scalar> = deserialize_scalar_vec(bytes_arr);

    let res = rand_proof_vec::create_randproof_vec(&value_f32_vec.to_vec(), &blinding_scalar_vec);
    match res {
        Ok((rand_proof_vec, eg_pair_vec)) => {
            let rand_proof_vec_ser: Vec<u8> = serialize_rand_proof_vec(&rand_proof_vec);
            let eg_pair_vec_ser: Vec<u8> = serialize_eg_pair_vec(&eg_pair_vec);
            let rand_proof_pyvec = create_pyvec(rand_proof_vec_ser);
            let eg_pair_vec_ser = create_pyvec(eg_pair_vec_ser);
            let res_vec = vec![rand_proof_pyvec, eg_pair_vec_ser];
            let res_pyvec = Box::into_raw(Box::new(create_pyvec(res_vec)));
            create_pyres_from_success(res_pyvec)
        }
        Err(e) => create_pyres_from_err(&e)
    }
}   

#[no_mangle]
pub extern fn verify_randproof(
    ped_commit_ptr: *const u8,
    ped_commit_len: usize,
    rand_commit_ptr: *const u8,
    rand_commit_len: usize,
    randproof_ptr: *const u8,
    proof_len: usize)
    -> PyRes {
    assert!(!ped_commit_ptr.is_null());
    assert!(!rand_commit_ptr.is_null());
    assert!(!randproof_ptr.is_null());

    println!("Is this slow?");

    let ped_commit_bytes: &[u8] = unsafe { slice::from_raw_parts(ped_commit_ptr, ped_commit_len as usize) };
    let ped_commit_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&ped_commit_bytes);
    println!("Twelve");

    let rand_commit_bytes: &[u8] = unsafe { slice::from_raw_parts(rand_commit_ptr, rand_commit_len as usize) };
    let rand_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&rand_commit_bytes);
    println!("Thirteen");

    let eg_pair_vec: Vec<ElGamalPair> = ped_commit_rp_vec.par_iter().zip(&rand_rp_vec).map(|(x, y)| ElGamalPair{L: *x, R: *y}).collect();
    println!("Eighteen");

    let rand_proof_bytes: &[u8] = unsafe { slice::from_raw_parts(randproof_ptr, proof_len as usize) };
    println!("Fourteen");

    let rand_proof_vec: Vec<RandProof> = deserialize_rand_proof_vec(&rand_proof_bytes);

    println!("Eight");

    match rand_proof_vec::verify_randproof_vec(&rand_proof_vec, &eg_pair_vec) {
        Ok(v) => {
            let v_ptr = Box::into_raw(Box::new(v));
            create_pyres_from_success(v_ptr)
        }
        Err(e) => {
            create_pyres_from_err(e)
        }
    }
}

#[no_mangle]
pub extern fn create_squarerandproof(
    value_ptr: *const f32,
    value_len: usize,
    blinding_1_ptr: *const u8,
    blinding_1_len: usize,
    blinding_2_ptr: *const u8,
    blinding_2_len: usize
    )
    -> PyRes {
    assert!(!value_ptr.is_null());
    assert!(!blinding_1_ptr.is_null());
    assert!(!blinding_2_ptr.is_null());

    let value_f32_vec: &[f32] = unsafe { slice::from_raw_parts(value_ptr, value_len as usize) };
    let bytes_1_arr: &[u8] = unsafe { slice::from_raw_parts(blinding_1_ptr, blinding_1_len as usize) };
    let blinding_1_scalar_vec: Vec<Scalar> = deserialize_scalar_vec(bytes_1_arr);
    let bytes_2_arr: &[u8] = unsafe { slice::from_raw_parts(blinding_2_ptr, blinding_2_len as usize) };
    let blinding_2_scalar_vec: Vec<Scalar> = deserialize_scalar_vec(bytes_2_arr);

    let res: Result<(Vec<SquareRandProof>, Vec<SquareRandProofCommitments>), L2RangeProofError> = square_rand_proof_vec::create_l2rangeproof_vec(&value_f32_vec.to_vec(),
                                                             &blinding_1_scalar_vec,
                                                             &blinding_2_scalar_vec);

    match res {
        Ok((rand_proof_vec, eg_pair_vec)) => {
            let rand_proof_vec_ser: Vec<u8> = bincode::serialize(&rand_proof_vec).unwrap();
            let eg_pair_vec_ser: Vec<u8> = bincode::serialize(&eg_pair_vec).unwrap();
            let rand_proof_pyvec = create_pyvec(rand_proof_vec_ser);
            let eg_pair_vec_ser = create_pyvec(eg_pair_vec_ser);
            let res_vec = vec![rand_proof_pyvec, eg_pair_vec_ser];
            let res_pyvec = Box::into_raw(Box::new(create_pyvec(res_vec)));
            create_pyres_from_success(res_pyvec)
        }
        Err(e) => create_pyres_from_err(&e)
    }
}

#[no_mangle]
pub extern fn verify_squarerandproof(
    commit_ptr: *const u8,
    commit_len: usize,
    randproof_ptr: *const u8,
    proof_len: usize)
    -> PyRes {
    assert!(!commit_ptr.is_null());
    assert!(!randproof_ptr.is_null());

    let commit_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_ptr, commit_len as usize) };
    let eg_pair_vec: Vec<SquareRandProofCommitments> = bincode::deserialize(&commit_bytes).unwrap();
    let rand_proof_bytes: &[u8] = unsafe { slice::from_raw_parts(randproof_ptr, proof_len as usize) };
    let rand_proof_vec: Vec<SquareRandProof> = bincode::deserialize(&rand_proof_bytes).unwrap();

    match square_rand_proof_vec::verify_l2rangeproof_vec(&rand_proof_vec, &eg_pair_vec) {
        Ok(v) => {
            let v_ptr = Box::into_raw(Box::new(v));
            create_pyres_from_success(v_ptr)
        }
        Err(e) => {
            create_pyres_from_err(e)
        }
    }
}

// Helper method to speed everything up ...!
#[no_mangle]
pub extern fn create_l2proof(
    value_ptr: *const f32,
    value_len: usize,
    blinding_1_ptr: *const u8,
    blinding_1_len: usize,
    blinding_2_ptr: *const u8,
    blinding_2_len: usize,
    range_exp: usize,
    n_partition: usize
)
    -> PyRes {
    assert!(!value_ptr.is_null());
    assert!(!blinding_1_ptr.is_null());
    assert!(!blinding_2_ptr.is_null());

    let value_f32_vec: &[f32] = unsafe { slice::from_raw_parts(value_ptr, value_len as usize) };
    let bytes_1_arr: &[u8] = unsafe { slice::from_raw_parts(blinding_1_ptr, blinding_1_len as usize) };
    let blinding_1_scalar_vec: Vec<Scalar> = deserialize_scalar_vec(bytes_1_arr);
    let bytes_2_arr: &[u8] = unsafe { slice::from_raw_parts(blinding_2_ptr, blinding_2_len as usize) };
    let blinding_2_scalar_vec: Vec<Scalar> = deserialize_scalar_vec(bytes_2_arr);

    // Should we clip?
    // TODO: clip
    let res_range = l2_range_proof_vec::create_rangeproof_l2(&value_f32_vec.to_vec(),
                                                             &blinding_2_scalar_vec.to_vec(),
                                                             range_exp,
                                                             n_partition);

    let res_rand: Result<(Vec<SquareRandProof>, Vec<SquareRandProofCommitments>), L2RangeProofError> = square_rand_proof_vec::create_l2rangeproof_vec(&value_f32_vec.to_vec(),
                                                                                                                                                 &blinding_1_scalar_vec,
                                                                                                                                                 &blinding_2_scalar_vec);



    match (res_rand, res_range) {
        (Ok((rand_proof_vec, eg_pair_vec)), Ok((range_proof, square_commit))) => {
            let rand_proof_vec_ser: Vec<u8> = bincode::serialize(&rand_proof_vec).unwrap();
            let eg_pair_vec_ser: Vec<u8> = bincode::serialize(&eg_pair_vec).unwrap();
            let range_proof_ser: Vec<u8> = bincode::serialize(&range_proof).unwrap();
            let square_commit_ser: Vec<u8> = bincode::serialize(&square_commit).unwrap();
            let rand_proof_pyvec = create_pyvec(rand_proof_vec_ser);
            let eg_pair_vec_pyvec = create_pyvec(eg_pair_vec_ser);
            let range_proof_pyvec = create_pyvec(range_proof_ser);
            let square_commit_pyvec = create_pyvec(square_commit_ser);
            let res_vec = vec![rand_proof_pyvec, eg_pair_vec_pyvec, range_proof_pyvec, square_commit_pyvec];
            let res_pyvec = Box::into_raw(Box::new(create_pyvec(res_vec)));
            create_pyres_from_success(res_pyvec)
        }
        (Err(e), _) => {
            create_pyres_from_err(&e)
        }
        (_, Err(e)) => create_pyres_from_err(&e)
    }
}

// Helper method to speed everything up ...!
#[no_mangle]
pub extern fn verify_l2proof(
    commit_ptr: *const u8,
    commit_len: usize,
    randproof_ptr: *const u8,
    proof_len: usize,
    rangeproof_ptr: *const u8,
    square_ptr: *const u8,
    prove_range: usize)
    -> PyRes {
    assert!(!commit_ptr.is_null());
    assert!(!randproof_ptr.is_null());

    let commit_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_ptr, commit_len as usize) };
    let eg_pair_vec: Vec<SquareRandProofCommitments> = bincode::deserialize(&commit_bytes).unwrap();
    let rand_proof_bytes: &[u8] = unsafe { slice::from_raw_parts(randproof_ptr, proof_len as usize) };
    let rand_proof_vec: Vec<SquareRandProof> = bincode::deserialize(&rand_proof_bytes).unwrap();
    let range_proof_bytes: &[u8] = unsafe { slice::from_raw_parts(rangeproof_ptr, 616 as usize) };
    let range_proof: RangeProof = bincode::deserialize(&range_proof_bytes).unwrap();
    let square_bytes: &[u8] = unsafe { slice::from_raw_parts(square_ptr, 40) };
    let square_commit: RistrettoPoint = bincode::deserialize(&square_bytes).unwrap();

    // Verify that the commitments to the squares add up to the square_commit !
    let sum = eg_pair_vec.iter().map(|x| x.c_sq).reduce(|a, b| a + b).unwrap();
    if sum != square_commit {
        let err = l2_range_proof_vec::errors::L2RangeProofError::SumError;
        return create_pyres_from_err(err)
    }

    let v_rand = square_rand_proof_vec::verify_l2rangeproof_vec(&rand_proof_vec, &eg_pair_vec);
    let v_range = l2_range_proof_vec::verify_rangeproof_l2(&range_proof, &square_commit, prove_range);

    match (v_rand, v_range) {
        (Ok(v1), Ok(v2)) => {
            let v_ptr = Box::into_raw(Box::new(v1 && v2));
            create_pyres_from_success(v_ptr)
        }
        (Err(e), _) => {
            create_pyres_from_err(e)
        }
        (_, Err(e)) => {
            create_pyres_from_err(e)
        }
    }
}

#[no_mangle]
pub extern fn split_elgamal_pair_vector(commit_ptr: *const u8, commit_len: usize) -> PyVec {
    assert!(!commit_ptr.is_null());
    let commit_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_ptr, commit_len as usize) };
    let eg_pair_vec: Vec<ElGamalPair> = deserialize_eg_pair_vec(commit_bytes);
    
    let left_rp_vec: Vec<RistrettoPoint> = eg_pair_vec.iter().map(|x| x.L).collect();
    let right_rp_vec: Vec<RistrettoPoint> = eg_pair_vec.iter().map(|x| x.R).collect();

    let left_rp_vec_ser: Vec<u8> = serialize_rp_vec(&left_rp_vec);
    let right_rp_vec_ser: Vec<u8> = serialize_rp_vec(&right_rp_vec);

    let left_pyvec: PyVec = create_pyvec(left_rp_vec_ser);
    let right_pyvec: PyVec = create_pyvec(right_rp_vec_ser);
    let res_vec = vec![left_pyvec, right_pyvec];
    create_pyvec(res_vec)
}

#[no_mangle]
pub extern fn join_to_elgamal_pair_vector(
    ped_commit_ptr: *const u8,
    ped_commit_len: usize,
    rand_commit_ptr: *const u8,
    rand_commit_len: usize)
    -> PyVec {
    assert!(!ped_commit_ptr.is_null());
    assert!(!rand_commit_ptr.is_null());
        
    let ped_commit_bytes: &[u8] = unsafe { slice::from_raw_parts(ped_commit_ptr, ped_commit_len as usize) };
    let ped_commit_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&ped_commit_bytes);
    let rand_commit_bytes: &[u8] = unsafe { slice::from_raw_parts(rand_commit_ptr, rand_commit_len as usize) };
    let rand_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&rand_commit_bytes);
    let eg_pair_vec: Vec<ElGamalPair> = ped_commit_rp_vec.par_iter().zip(&rand_rp_vec).map(|(x, y)| ElGamalPair{L: *x, R: *y}).collect();
    let eg_pair_vec_ser: Vec<u8> = serialize_eg_pair_vec(&eg_pair_vec);
    create_pyvec(eg_pair_vec_ser)
}

#[no_mangle]
pub extern fn split_squaretriple_pair_vector(commit_ptr: *const u8, commit_len: usize) -> PyVec {
    assert!(!commit_ptr.is_null());
    let commit_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_ptr, commit_len as usize) };
    let eg_pair_vec: Vec<SquareRandProofCommitments> = bincode::deserialize(commit_bytes).unwrap();

    let left_rp_vec: Vec<RistrettoPoint> = eg_pair_vec.iter().map(|x| x.c.L).collect();
    let right_rp_vec: Vec<RistrettoPoint> = eg_pair_vec.iter().map(|x| x.c.R).collect();
    let square_ped_vec: Vec<RistrettoPoint> = eg_pair_vec.iter().map(|x| x.c_sq).collect();

    let left_pyvec: PyVec = create_pyvec_ristretto(left_rp_vec);
    let right_pyvec: PyVec = create_pyvec_ristretto(right_rp_vec);
    let square_pyvec: PyVec = create_pyvec_ristretto(square_ped_vec);
    let res_vec = vec![left_pyvec, right_pyvec, square_pyvec];
    create_pyvec(res_vec)
}

#[no_mangle]
pub extern fn join_to_squaretriple_pair_vector(
    ped_commit_ptr: *const u8,
    ped_commit_len: usize,
    rand_commit_ptr: *const u8,
    rand_commit_len: usize,
    square_commit_ptr: *const u8,
    square_commit_len: usize)
    -> PyVec {
    assert!(!ped_commit_ptr.is_null());
    assert!(!rand_commit_ptr.is_null());

    let ped_commit_bytes: &[u8] = unsafe { slice::from_raw_parts(ped_commit_ptr, ped_commit_len as usize) };
    let ped_commit_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&ped_commit_bytes);
    let rand_commit_bytes: &[u8] = unsafe { slice::from_raw_parts(rand_commit_ptr, rand_commit_len as usize) };
    let rand_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&rand_commit_bytes);
    let square_commit_bytes: &[u8] = unsafe { slice::from_raw_parts(square_commit_ptr, square_commit_len as usize) };
    let square_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&square_commit_bytes);
    let commit_vec: Vec<SquareRandProofCommitments> = ped_commit_rp_vec.iter()
        .zip(rand_rp_vec).zip(square_rp_vec)
        .map(|((x, y), ped)| SquareRandProofCommitments {c: ElGamalPair{L: *x, R: y}, c_sq: ped})
        .collect();
    let eg_pair_vec_ser: Vec<u8> = bincode::serialize(&commit_vec).unwrap();
    create_pyvec(eg_pair_vec_ser)
}

#[no_mangle]
pub extern fn clip_to_range(value_ptr: *const f32, value_len: usize, range: usize) -> PyVec {
    assert!(!value_ptr.is_null());
    let value_f32_vec: &[f32] = unsafe { slice::from_raw_parts(value_ptr, value_len as usize) };
    let value_f32_vec_clipped: Vec<f32> = range_proof_vec::clip_f32_to_range_vec(&value_f32_vec.to_vec(), range);
    create_pyvec(value_f32_vec_clipped)
}

#[no_mangle]
pub extern fn quantize_probabilistic(value_ptr: *const f32, value_len: usize, range: usize) -> PyVec {
    assert!(!value_ptr.is_null());
    let value_f32_vec: &[f32] = unsafe { slice::from_raw_parts(value_ptr, value_len as usize) };
    let value_f32_vec_clipped: Vec<f32> = range_proof_vec::clip_f32_to_range_vec(&value_f32_vec.to_vec(), range);



    create_pyvec(value_f32_vec_clipped)
}

#[no_mangle]
pub extern fn commits_equal(
    commit_a_ptr: *const u8,
    commit_b_ptr: *const u8,
    commit_len: usize
    ) -> PyRes {
    assert!(!commit_a_ptr.is_null());
    assert!(!commit_b_ptr.is_null());
    
    let commit_a_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_a_ptr, commit_len as usize) };
    let commit_a_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&commit_a_bytes);
    let commit_b_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_b_ptr, commit_len as usize) };
    let commit_b_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&commit_b_bytes);
    let res: bool = commit_a_rp_vec == commit_b_rp_vec;
    let res_ptr = Box::into_raw(Box::new(res));
    create_pyres_from_success(res_ptr)
}

#[no_mangle]
pub extern fn equals_neutral_group_element_vec(commit_ptr: *const u8, commit_len: usize) -> PyRes {
    assert!(!commit_ptr.is_null());
    let commit_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_ptr, commit_len as usize) };
    let commit_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&commit_bytes);
    let res: bool = commit_rp_vec == zero_rp_vec(commit_rp_vec.len());
    let res_ptr = Box::into_raw(Box::new(res));
    create_pyres_from_success(res_ptr)
}

#[no_mangle]
pub extern fn create_zero_scalar_vector(len: usize) -> PyVec {
    let zero_scalar_vec: Vec<Scalar> = zero_scalar_vec(len);
    let zero_scalar_vec_ser: Vec<u8> = serialize_scalar_vec(&zero_scalar_vec);
    create_pyvec(zero_scalar_vec_ser)
}

#[no_mangle]
pub extern fn create_zero_group_element_vector(len: usize) -> PyVec {
    let zero_rp_vec: Vec<RistrettoPoint> = zero_rp_vec(len);
    let zero_rp_vec_ser: Vec<u8> = serialize_rp_vec(&zero_rp_vec);
    create_pyvec(zero_rp_vec_ser)
}

#[no_mangle]
pub extern fn create_random_blinding_vector(len: usize) -> PyVec {
    let rnd_vec: Vec<Scalar> = rnd_scalar_vec(len);
    let rnd_vec_ser: Vec<u8> = serialize_scalar_vec(&rnd_vec);
    create_pyvec(rnd_vec_ser)
}

#[no_mangle]
pub extern fn add_scalars(commit_ptr: *const u8, commit_len: usize) -> PyVec {
    assert!(!commit_ptr.is_null());
    let commit_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_ptr, commit_len as usize) };
    let commit_scalar_vec: Vec<Scalar> = deserialize_scalar_vec(&commit_bytes);
    let sum: Scalar = commit_scalar_vec.into_iter().reduce(|a, b| a + b).unwrap();
    let encoded: Vec<u8> = bincode::serialize(&sum).unwrap();
    create_pyvec(encoded)
}

#[no_mangle]
pub extern fn filter_unequal_commits(
    commit_a_ptr: *const u8,
    commit_b_ptr: *const u8,
    commit_len: usize)
    -> PyVec {

    assert!(!commit_a_ptr.is_null());
    assert!(!commit_b_ptr.is_null());
    let commit_a_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_a_ptr, commit_len as usize) };
    let commit_a_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&commit_a_bytes);
    let commit_b_bytes: &[u8] = unsafe { slice::from_raw_parts(commit_b_ptr, commit_len as usize) };
    let commit_b_rp_vec: Vec<RistrettoPoint> = deserialize_rp_vec(&commit_b_bytes);
    let commit_tuples: Vec<(&RistrettoPoint, &RistrettoPoint)> = commit_a_rp_vec.iter().zip(&commit_b_rp_vec).collect();
    let unequal_commit_tuples: Vec<(&RistrettoPoint, &RistrettoPoint)> = commit_tuples.into_iter().filter(|(x, y)| x != y).collect();
    let mut left_commit_vec: Vec<RistrettoPoint> = Vec::with_capacity(unequal_commit_tuples.len());
    let mut right_commit_vec: Vec<RistrettoPoint> = Vec::with_capacity(unequal_commit_tuples.len());
    for (l, r) in unequal_commit_tuples {
        left_commit_vec.push(*l);
        right_commit_vec.push(*r);
    }
    let left_commit_vec_ser = serialize_rp_vec(&left_commit_vec);
    let right_commit_vec_ser = serialize_rp_vec(&right_commit_vec);
    let left_pyvec = create_pyvec(left_commit_vec_ser);
    let right_pyvec = create_pyvec(right_commit_vec_ser);
    let res_vec = vec![left_pyvec, right_pyvec];
    create_pyvec(res_vec)
}

// Returns the sum of squares in vec. Discretizes floats
// #[no_mangle]
// pub extern fn sum_of_squares(value_ptr: *const f32,
//                      value_len: usize,
//                     prove_range: usize) -> PyRes {
//     assert!(!value_ptr.is_null());
//     let value_f32_vec: &[f32] = unsafe { slice::from_raw_parts(value_ptr, value_len as usize) };
//     let value_f32_vec_clipped: Vec<f32> = range_proof_vec::clip_f32_to_range_vec(&value_f32_vec.to_vec(), range);
//
//     let offset_value_scalar: Scalar = Scalar::from((1 as URawFix) << (prove_range-1));
//
//     let value_squared: Vec<Scalar> =
//         value_f32_vec_clipped.iter()
//             .map(|x| f32_to_scalar(x) + &offset_value_scalar)
//             // Now we add
//             .map(|x| &x * x)
//             .collect();
//
//     let added: Scalar = Scalar.sum(value_squared.iter());
//     let serialized = bincode::serialize(&added).unwrap();
//     let boxed = Box::into_raw(Box::new(serialized));
//
//     create_pyres_from_success(boxed)
// }

/// Creates a PyVec struct for ffi
/// CAUTION: it will make no assumption of the size of the data (byte, u32, f32...)
/// On the python side, the client is responsible for unpacking the data accordingly
/// Will cause intentional memory leak in rust (mem management handled by python)
fn create_pyvec<T>(vec: Vec<T>) -> PyVec {
    let res = PyVec { data: vec.as_ptr() as *const c_void, len: vec.len()};
    mem::forget(vec);
    res
}

fn create_pyres_from_success<T>(content_ptr: *mut T) -> PyRes {
    let pyres = PyRes{ret: 0usize, msg: ptr::null(), res: content_ptr as *const c_void};
    mem::forget(content_ptr);
    pyres
}

fn create_pyres_from_err<T: fmt::Display>(e: T) -> PyRes {
    let err_msg= CString::new(format!("{}", e)).unwrap();
    let pyres = PyRes{ret: 1usize, msg: err_msg.as_ptr(), res: ptr::null()};
    mem::forget(err_msg);
    pyres
}

fn create_x() -> Scalar {
    let bytes: [u8; 32] = [
            0x4e, 0x5a, 0xb4, 0x34, 0x5d, 0x47, 0x08, 0x84,
            0x59, 0x13, 0xb4, 0x64, 0x1b, 0xc2, 0x7d, 0x52,
            0x52, 0xa5, 0x85, 0x10, 0x1b, 0xcc, 0x42, 0x44,
            0xd4, 0x49, 0xf4, 0xa8, 0x79, 0xd9, 0xf2, 0x04,
    ];
    
    Scalar::from_bytes_mod_order(bytes)
}


fn create_pyvec_ristretto(point: Vec<RistrettoPoint>) -> PyVec {
    create_pyvec(serialize_rp_vec(&point))
}