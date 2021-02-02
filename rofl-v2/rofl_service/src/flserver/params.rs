use prost::Message;
use rofl_crypto::rand_proof::{ElGamalPair, RandProof};
use rofl_crypto::rand_proof_vec;
use rofl_crypto::range_proof_vec;
use rofl_crypto::pedersen_ops::default_discrete_log_vec;
use rofl_crypto::conversion32::scalar_to_f32_vec;
use bulletproofs::RangeProof;
use curve25519_dalek::ristretto::{RistrettoPoint, CompressedRistretto};
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::MultiscalarMul;
use super::flservice::{EncRangeData, CryptoConfig};
use std::io::Cursor;



pub const PLAIN_TYPE : u8 = 1;
pub const ENC_RANGE_TYPE : u8 = 2;

#[derive(Clone, Debug)]
pub enum EncModelParamType {
    Plain,
    EncRange
}

impl EncModelParamType {
    pub fn get_type_int(&self) -> u8 {
        match *self {
            EncModelParamType::Plain => {
                PLAIN_TYPE
            }
            EncModelParamType::EncRange => {
                ENC_RANGE_TYPE
            }
        }
    }

    pub fn get_type_from_int(int_type : &i32) -> Option<Self> {
        let int_type_u8 = *int_type as u8;
        match int_type_u8 {
            PLAIN_TYPE => Some(EncModelParamType::Plain),
            ENC_RANGE_TYPE => Some(EncModelParamType::EncRange),
            _ => None
        }
    }
}

fn extract_pedersen_vec(gamal_vec : &Vec<ElGamalPair>) -> Vec<RistrettoPoint> {
    return gamal_vec.iter().map(|x| x.L).collect();
}

#[derive(Clone)]
pub enum EncModelParamsAccumulator {
    Plain(PlainParams),
    Enc(Vec<ElGamalPair>)
}

impl EncModelParamsAccumulator {

    fn gamal_accumulate(&mut self, other : &Vec<ElGamalPair>) -> bool {
        if let EncModelParamsAccumulator::Enc(gamal_vector) = self {
            gamal_vector.iter_mut().zip(other).for_each(|(a, b)| *a += b);
            return true;
        }
        false
    }

    pub fn accumulate_other(&mut self, other: &EncModelParams) -> bool {
        match other {
            EncModelParams::Plain(params) => {
                if let EncModelParamsAccumulator::Plain(accumulator) = self {
                    return accumulator.accumulate_other(params);
                }
                false
            }
            EncModelParams::EncRange(params) => {
                return self.gamal_accumulate(&params.enc_values);
            }
        }
    }

    pub fn extract(&self) -> Option<Vec<f32>> {
        match self {
            EncModelParamsAccumulator::Plain(params) => {
                return Some(params.content.clone());
            }
            EncModelParamsAccumulator::Enc(enc_params) => {
                let rp_vec = extract_pedersen_vec(enc_params);
                let scalar_vec: Vec<Scalar> = default_discrete_log_vec(&rp_vec);
                let f32_vec: Vec<f32> = scalar_to_f32_vec(&scalar_vec);
                return Some(f32_vec);
            }
        }
    }
}


#[derive(Clone)]
pub enum EncModelParams {
    Plain(PlainParams),
    EncRange(EncParamsRange)
}


// Lubu: I started implementing the diffrent version of EncParams with enums instead of traits 
// due to object safty problems 
// https://www.mattkennedy.io/blog/rust_polymorphism/
// https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md
// https://doc.rust-lang.org/book/ch17-02-trait-objects.html
impl EncModelParams {
    pub fn unity(param_type : &EncModelParamType, size : usize) -> EncModelParamsAccumulator {
        match param_type {
            EncModelParamType::Plain => {
                return EncModelParamsAccumulator::Plain(PlainParams {
                    content : vec![0.0; size],
                });
            }
            EncModelParamType::EncRange => {
                return EncModelParamsAccumulator::Enc(vec![ElGamalPair::unity(); size]);
            }
        }
    }

    /*pub fn accumulate_other(&mut self, other: &Self) -> bool {
        match self {
            EncModelParams::Plain(params) => {
                if let EncModelParams::Plain(other_plain) = other {
                    return params.accumulate_other(&other_plain);
                }
                false
            }
            EncModelParamType::EncRange => {
                return EncModelParamsAccumulator::Enc(vec![ElGamalPair::unity(); size]);
            }
        }
    }*/

    pub fn verify(&self) -> bool {
        match self {
            EncModelParams::Plain(params) => {
                return true;
            }
            EncModelParams::EncRange(params) => {
                //Check rand proof 
                let res = rand_proof_vec::verify_randproof_vec(&params.rand_proofs, &params.enc_values);
                if let Ok(ok) = res {
                     //Check range proof 
                    let vec_tmp = extract_pedersen_vec(&params.enc_values);
                    let range_res = range_proof_vec::verify_rangeproof(&params.range_proofs, &vec_tmp, params.prove_range);
                    if let Ok(ok_range) = range_res {
                        return ok && ok_range;
                    }
                }
                return false;
            }
        }
    }

    pub fn verifiable(&self) -> bool {
        match self {
            EncModelParams::Plain(params) => {
                return false;
            }
            EncModelParams::EncRange(params) => {
                return true;
            }
        }
    }

    pub fn length(&self) -> usize {
        match self {
            EncModelParams::Plain(params) => {
                return params.content.len()
            }
            EncModelParams::EncRange(params) => {
                return params.enc_values.len()
            }
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        match self {
            EncModelParams::Plain(params) => {
                return bincode::serialize(&params.content).unwrap();
            }
            EncModelParams::EncRange(params) => {
                return params.serialize();
            }
        }
    }

    pub fn deserialize(param_type : &EncModelParamType, data : &Vec<u8>) -> Self {
        match param_type {
            EncModelParamType::Plain => {
                return EncModelParams::Plain(PlainParams {
                    content : bincode::deserialize(data).unwrap(),
                });
            }
            EncModelParamType::EncRange => {
                return EncModelParams::EncRange(EncParamsRange::deserialize(data));
            }
        }
    }

    pub fn encrypt(param_type : &EncModelParamType, plain_params : &PlainParams, config : &CryptoConfig, blindings : &Vec<Scalar>) -> Option<Self> {
        match param_type {
            EncModelParamType::Plain => {
                return Some(EncModelParams::Plain(PlainParams {
                    content : plain_params.content.clone(),
                }));
            }
            EncModelParamType::EncRange => {
                return Some(EncModelParams::EncRange(EncParamsRange::encrypt(
                    &plain_params.content, 
                    blindings, 
                    config.value_range as usize, 
                    config.n_partition as usize)));

            }
        }
    }
}
#[derive(Clone)]
pub struct EncParamsRange {
    pub enc_values : Vec<ElGamalPair>,
    pub rand_proofs : Vec<RandProof>,
    pub range_proofs : Vec<RangeProof>,
    pub prove_range : usize
}

fn encode_el_gamal_vec(enc_values : &Vec<ElGamalPair>) -> Vec<u8> {
    let mut out = Vec::with_capacity(ElGamalPair::serialized_size() * enc_values.len());
    enc_values.iter().for_each(|x| { out.extend(x.to_bytes().iter()) });
    out
}

fn decode_el_gamal_vec(encoded : &Vec<u8>) -> Vec<ElGamalPair> {
    //TODO error handling
    let num_elems = encoded.len() / ElGamalPair::serialized_size();
    let mut out = Vec::with_capacity(num_elems);
    let size = ElGamalPair::serialized_size();
    for id in 0..num_elems {
        out.push(ElGamalPair::from_bytes(
            &encoded[id*size..((id + 1) * size)]
        ).unwrap());
    }
    out
}

fn encode_rand_proof_vec(values : &Vec<RandProof>) -> Vec<u8> {
    let mut out = Vec::with_capacity(RandProof::serialized_size() * values.len());
    values.iter().for_each(|x| { out.extend(x.to_bytes().iter()) });
    out
}

fn decode_rand_proof_vec(encoded : &Vec<u8>) -> Vec<RandProof> {
    //TODO error handling
    let num_elems = encoded.len() / RandProof::serialized_size();
    let mut out = Vec::with_capacity(num_elems);
    let size = RandProof::serialized_size();
    for id in 0..num_elems {
        out.push(RandProof::from_bytes(
            &encoded[id*size..((id + 1) * size)]
        ).unwrap());
    }
    out
}

fn encode_range_proof_vec(values : &Vec<RangeProof>) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(values.len());
    values.iter().for_each(|x| { out.push(x.to_bytes()) });
    out
}

fn decode_range_proof_vec(encoded : &Vec<Vec<u8>>) -> Vec<RangeProof> {
    //TODO error handling
    let num_elems = encoded.len();
    let mut out = Vec::with_capacity(num_elems);
    encoded.iter().for_each(|x| {out.push(RangeProof::from_bytes(&x[..]).unwrap())});
    out
}


impl EncParamsRange {
    pub fn encrypt(plaintext_vec : &Vec<f32>, blinding_vec : &Vec<Scalar>, prove_range : usize, n_partition : usize) -> Self {
        let range_clipped = range_proof_vec::clip_f32_to_range_vec(plaintext_vec, prove_range);
        let (range_proofs, enc_com ) =  range_proof_vec::create_rangeproof(&range_clipped, blinding_vec, prove_range, n_partition).unwrap();
        let (rand_proofs, enc_update) = rand_proof_vec::create_randproof_vec_existing(plaintext_vec, enc_com, &blinding_vec).unwrap();
        EncParamsRange {
            enc_values : enc_update,
            rand_proofs : rand_proofs,
            range_proofs : range_proofs,
            prove_range : prove_range
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let enc_values = encode_el_gamal_vec(&self.enc_values);
        let rand_proofs = encode_rand_proof_vec(&self.rand_proofs);
        let range_proofs = encode_range_proof_vec(&self.range_proofs);
        let enc_data = EncRangeData {
            enc_values : enc_values,
            rand_proof : rand_proofs,
            range_proof : range_proofs,
            range_bits : self.prove_range as i32
        };
        let mut buffer = Vec::with_capacity(enc_data.encoded_len() + 1);
        enc_data.encode_length_delimited(&mut buffer);
        buffer
    }

    pub fn deserialize(data : &Vec<u8>) -> EncParamsRange {
        let msg = EncRangeData::decode_length_delimited(&mut Cursor::new(data)).unwrap();
        let enc_values = decode_el_gamal_vec(&msg.enc_values);
        let rand_proofs = decode_rand_proof_vec(&msg.rand_proof);
        let range_proofs = decode_range_proof_vec(&msg.range_proof);
        EncParamsRange {
            enc_values : enc_values,
            rand_proofs : rand_proofs,
            range_proofs : range_proofs,
            prove_range : msg.range_bits as usize
        }
    }
}



#[derive(Clone)]
pub struct PlainParams {
    pub content : Vec<f32>,
}

impl PlainParams  {
    pub fn unity(size : usize) -> Self {
        return PlainParams {
            content : vec![0.0; size],
        };
    }
    
    pub fn accumulate_other(&mut self, other : &PlainParams) -> bool {
        if other.content.len() != self.content.len() {
            return false;
        }
        for i in 0..self.content.len() {
            self.content[i] += other.content[i]
        }
        return true;
    }

    pub fn serialize(&self) -> Vec<u8> {
        return bincode::serialize(&self.content).unwrap();
    }

    pub fn deserialize(data : &Vec<u8>) -> Self {
        return PlainParams {
            content : bincode::deserialize(data).unwrap(),
        };       
    }
}





#[cfg(test)]
mod tests {
    use super::*;

    fn test_helper(p1 : &mut EncModelParams, p2 : &EncModelParams, enc_type : &EncModelParamType, acumulator : &mut EncModelParamsAccumulator) {
        acumulator.accumulate_other(&p1);
        acumulator.accumulate_other(&p2);
        assert_eq!(&acumulator.extract().unwrap()[..], &vec![2.0;10][..]);
        assert!(p1.verifiable() == false);
        let ser = p1.serialize();
        println!("{}" , ser.len());
        //assert_eq!(&acumulator.extract().unwrap()[..], &EncModelParams::deserialize(&enc_type, &ser).extract().unwrap()[..]);
    }

    #[test]
    fn test_plain_params() {
        let mut p1 =  EncModelParams::Plain(PlainParams {
            content : vec![1.0; 10]
        });
        let p2 =  EncModelParams::Plain (PlainParams {
            content : vec![1.0; 10]
        });
        //test_helper(&mut p1, &p2, &EncModelParamType::Plain, EncModelParams::unity(&EncModelParamType::Plain, 10));
    }

    #[test]
    fn test_plain_encoding() {
        //let mut p1 =  vec![ElGamalPair::unity(), ElGamalPair::unity()];
        let encoded = ElGamalPair::unity().to_bytes();
        println!("len {}", encoded.len());
    }
}