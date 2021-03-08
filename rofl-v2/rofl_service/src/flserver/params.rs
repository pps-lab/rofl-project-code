use rofl_crypto::bsgs32::BSGSTable;
use rofl_crypto::pedersen_ops::discrete_log_vec_table;
use super::flservice::{CryptoConfig, EncNormData, EncRangeData, FloatBlock};
use bulletproofs::RangeProof;
use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::scalar::Scalar;
use prost::Message;
use rofl_crypto::conversion32::scalar_to_f32_vec;
use rofl_crypto::rand_proof_vec;
use rofl_crypto::range_proof_vec;
use rofl_crypto::{
    l2_range_proof_vec,
    pedersen_ops::rnd_scalar_vec,
    rand_proof::{ElGamalPair, RandProof},
    square_rand_proof::{pedersen::SquareRandProofCommitments, SquareRandProof},
    square_rand_proof_vec,
};
use std::io::Cursor;
use log::info;
use rand::distributions::{Normal, Distribution};

pub const PLAIN_TYPE: u8 = 1;
pub const ENC_RANGE_TYPE: u8 = 2;
pub const ENC_L2_TYPE: u8 = 3;

#[derive(Clone, Debug)]
pub enum EncModelParamType {
    Plain,
    EncRange,
    EncL2,
}

impl EncModelParamType {
    pub fn get_type_int(&self) -> u8 {
        match *self {
            EncModelParamType::Plain => PLAIN_TYPE,
            EncModelParamType::EncRange => ENC_RANGE_TYPE,
            EncModelParamType::EncL2 => ENC_L2_TYPE,
        }
    }

    pub fn get_type_from_int(int_type: &i32) -> Option<Self> {
        let int_type_u8 = *int_type as u8;
        match int_type_u8 {
            PLAIN_TYPE => Some(EncModelParamType::Plain),
            ENC_RANGE_TYPE => Some(EncModelParamType::EncRange),
            ENC_L2_TYPE => Some(EncModelParamType::EncL2),
            _ => None,
        }
    }
}

fn extract_pedersen_vec(gamal_vec: &Vec<ElGamalPair>, num_elems: usize) -> Vec<RistrettoPoint> {
    return gamal_vec[..num_elems].iter().map(|x| x.L).collect();
}

fn extract_pedersen_vec_l2(gamal_vec: &[SquareRandProofCommitments]) -> Vec<RistrettoPoint> {
    return gamal_vec.iter().map(|x| x.c.L).collect();
}

#[derive(Clone)]
pub enum EncModelParamsAccumulator {
    Plain(PlainParams),
    Enc(Vec<ElGamalPair>),
}

impl EncModelParamsAccumulator {
    fn gamal_accumulate(&mut self, other: &Vec<ElGamalPair>) -> bool {
        if let EncModelParamsAccumulator::Enc(gamal_vector) = self {
            gamal_vector
                .iter_mut()
                .zip(other)
                .for_each(|(a, b)| *a += b);
            return true;
        }
        false
    }

    fn l2_vec_accumulate(&mut self, other: &Vec<SquareRandProofCommitments>) -> bool {
        if let EncModelParamsAccumulator::Enc(gamal_vector) = self {
            gamal_vector
                .iter_mut()
                .zip(other)
                .for_each(|(a, b)| *a += &b.c);
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
            EncModelParams::EncL2(params) => {
                return self.l2_vec_accumulate(&params.enc_values);
            }
        }
    }

    pub fn extract(&self, table: &BSGSTable) -> Option<Vec<f32>> {
        match self {
            EncModelParamsAccumulator::Plain(params) => {
                return Some(params.content.clone());
            }
            EncModelParamsAccumulator::Enc(enc_params) => {
                // Check if aggregation is correct
                for pair in enc_params {
                    if !pair.right_elem_is_unity() {
                        //Todo: error with result would be nicer instead of option
                        return None;
                    }
                }

                let rp_vec = extract_pedersen_vec(enc_params, enc_params.len());
                let scalar_vec: Vec<Scalar> = discrete_log_vec_table(&rp_vec, table);
                let f32_vec: Vec<f32> = scalar_to_f32_vec(&scalar_vec);
                // info!("Client result {}", f32_vec[0]);
                return Some(f32_vec);
            }
        }
    }
}

#[derive(Clone)]
pub enum EncModelParams {
    Plain(PlainParams),
    EncRange(EncParamsRange),
    EncL2(EncParamsL2),
}

// Lubu: I started implementing the diffrent version of EncParams with enums instead of traits
// due to object safty problems
// https://www.mattkennedy.io/blog/rust_polymorphism/
// https://github.com/rust-lang/rfcs/blob/master/text/0255-object-safety.md
// https://doc.rust-lang.org/book/ch17-02-trait-objects.html
impl EncModelParams {
    pub fn unity(param_type: &EncModelParamType, size: usize) -> EncModelParamsAccumulator {
        match param_type {
            EncModelParamType::Plain => {
                return EncModelParamsAccumulator::Plain(PlainParams {
                    content: vec![0.0; size],
                });
            }
            EncModelParamType::EncRange => {
                return EncModelParamsAccumulator::Enc(vec![ElGamalPair::unity(); size]);
            }
            EncModelParamType::EncL2 => {
                return EncModelParamsAccumulator::Enc(vec![ElGamalPair::unity(); size]);
            }
        }
    }

    pub fn verify(&self) -> bool {
        match self {
            EncModelParams::Plain(_) => {
                return true;
            }
            EncModelParams::EncRange(params) => {
                //Check rand proof
                let res =
                    rand_proof_vec::verify_randproof_vec(&params.rand_proofs, &params.enc_values);
                if let Ok(ok) = res {
                    //Check range proof
                    let num_elems = params.enc_values.len() * params.check_percentage / 100;
                    let vec_tmp = extract_pedersen_vec(&params.enc_values, num_elems);
                    let range_res = range_proof_vec::verify_rangeproof(
                        &params.range_proofs,
                        &vec_tmp,
                        params.prove_range,
                    );
                    if let Ok(ok_range) = range_res {
                        return ok && ok_range;
                    }
                }
                return false;
            }
            EncModelParams::EncL2(params) => {
                //Check rand proof
                let res = square_rand_proof_vec::verify_l2rangeproof_vec(
                    &params.square_proofs,
                    &params.enc_values,
                );
                if let Ok(ok) = res {
                    //Check range proof
                    let vec_tmp = extract_pedersen_vec_l2(&params.enc_values);
                    let range_res = range_proof_vec::verify_rangeproof(
                        &params.range_proofs,
                        &vec_tmp,
                        params.prove_range,
                    );
                    if let Ok(ok_range) = range_res {
                        //Check square
                        let sum = params.enc_values.iter().map(|x| x.c_sq).sum();
                        let res_sum = l2_range_proof_vec::verify_rangeproof_l2(
                            &params.square_range_proof,
                            &sum,
                            params.l2_prove_range,
                        );
                        if let Ok(ok_sum) = res_sum {
                            return ok && ok_range && ok_sum;
                        }
                    }
                }
                return false;
            }
        }
    }

    pub fn verifiable(&self) -> bool {
        match self {
            EncModelParams::Plain(_) => {
                return false;
            }
            _ => {
                return true;
            }
        }
    }

    pub fn length(&self) -> usize {
        match self {
            EncModelParams::Plain(params) => return params.content.len(),
            EncModelParams::EncRange(params) => return params.enc_values.len(),
            EncModelParams::EncL2(params) => return params.enc_values.len(),
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
            EncModelParams::EncL2(params) => {
                return params.serialize();
            }
        }
    }

    pub fn deserialize(param_type: &EncModelParamType, data: &Vec<u8>) -> Self {
        match param_type {
            EncModelParamType::Plain => {
                return EncModelParams::Plain(PlainParams {
                    content: bincode::deserialize(data).unwrap(),
                });
            }
            EncModelParamType::EncRange => {
                return EncModelParams::EncRange(EncParamsRange::deserialize(data));
            }
            EncModelParamType::EncL2 => {
                return EncModelParams::EncL2(EncParamsL2::deserialize(data));
            }
        }
    }

    pub fn encrypt(
        param_type: &EncModelParamType,
        plain_params: &PlainParams,
        config: &CryptoConfig,
        blindings: &Vec<Scalar>,
    ) -> Option<Self> {
        match param_type {
            EncModelParamType::Plain => {
                return Some(EncModelParams::Plain(PlainParams {
                    content: plain_params.content.clone(),
                }));
            }
            EncModelParamType::EncRange => {
                return Some(EncModelParams::EncRange(EncParamsRange::encrypt(
                    &plain_params.content,
                    blindings,
                    config.value_range as usize,
                    config.n_partition as usize,
                    config.check_percentage as usize,
                )));
            }
            EncModelParamType::EncL2 => {
                return Some(EncModelParams::EncL2(EncParamsL2::encrypt(
                    &plain_params.content,
                    blindings,
                    config.value_range as usize,
                    config.n_partition as usize,
                    config.l2_value_range as usize,
                )));
            }
        }
    }
}
#[derive(Clone)]
pub struct EncParamsRange {
    pub enc_values: Vec<ElGamalPair>,
    pub rand_proofs: Vec<RandProof>,
    pub range_proofs: Vec<RangeProof>,
    pub prove_range: usize,
    pub check_percentage: usize,
}

fn encode_el_gamal_vec(enc_values: &Vec<ElGamalPair>) -> Vec<u8> {
    let mut out = Vec::with_capacity(ElGamalPair::serialized_size() * enc_values.len());
    enc_values
        .iter()
        .for_each(|x| out.extend(x.to_bytes().iter()));
    out
}

fn decode_el_gamal_vec(encoded: &[u8]) -> Vec<ElGamalPair> {
    //TODO error handling
    let num_elems = encoded.len() / ElGamalPair::serialized_size();
    let mut out = Vec::with_capacity(num_elems);
    let size = ElGamalPair::serialized_size();
    for id in 0..num_elems {
        out.push(ElGamalPair::from_bytes(&encoded[id * size..((id + 1) * size)]).unwrap());
    }
    out
}

fn encode_rand_proof_vec(values: &Vec<RandProof>) -> Vec<u8> {
    let mut out = Vec::with_capacity(RandProof::serialized_size() * values.len());
    values.iter().for_each(|x| out.extend(x.to_bytes().iter()));
    out
}

fn decode_rand_proof_vec(encoded: &[u8]) -> Vec<RandProof> {
    //TODO error handling
    let num_elems = encoded.len() / RandProof::serialized_size();
    let mut out = Vec::with_capacity(num_elems);
    let size = RandProof::serialized_size();
    for id in 0..num_elems {
        out.push(RandProof::from_bytes(&encoded[id * size..((id + 1) * size)]).unwrap());
    }
    out
}

fn encode_range_proof_vec(values: &Vec<RangeProof>) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(values.len());
    values.iter().for_each(|x| out.push(x.to_bytes()));
    out
}

fn decode_range_proof_vec(encoded: &Vec<Vec<u8>>) -> Vec<RangeProof> {
    //TODO error handling
    let num_elems = encoded.len();
    let mut out = Vec::with_capacity(num_elems);
    encoded
        .iter()
        .for_each(|x| out.push(RangeProof::from_bytes(&x[..]).unwrap()));
    out
}

impl EncParamsRange {
    pub fn encrypt(
        plaintext_vec: &Vec<f32>,
        blinding_vec: &Vec<Scalar>,
        prove_range: usize,
        n_partition: usize,
        check_percentage: usize,
    ) -> Self {
        let range_clipped = range_proof_vec::clip_f32_to_range_vec(plaintext_vec, prove_range);
        // info!("First param {}, {}", plaintext_vec[0], range_clipped[0]);
        // Dummy probabilistic checking
        let (range_proofs, enc_com) = if check_percentage == 100 {
            range_proof_vec::create_rangeproof(
                &range_clipped,
                blinding_vec,
                prove_range,
                n_partition,
            )
            .unwrap()
        } else {
            let num_elems = range_clipped.len() * check_percentage / 100;
            let filtered_elems = range_clipped[..num_elems].to_vec();
            let filtered_blidings = blinding_vec[..num_elems].to_vec();
            range_proof_vec::create_rangeproof(
                &filtered_elems,
                &filtered_blidings,
                prove_range,
                n_partition,
            )
            .unwrap()
        };
        let (rand_proofs, enc_update) = if check_percentage == 100 {
            rand_proof_vec::create_randproof_vec_existing(plaintext_vec, enc_com, &blinding_vec)
                .unwrap()
        } else {
            rand_proof_vec::create_randproof_vec(plaintext_vec, &blinding_vec).unwrap()
        };
        EncParamsRange {
            enc_values: enc_update,
            rand_proofs: rand_proofs,
            range_proofs: range_proofs,
            prove_range: prove_range,
            check_percentage: check_percentage,
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let enc_values = encode_el_gamal_vec(&self.enc_values);
        let rand_proofs = encode_rand_proof_vec(&self.rand_proofs);
        let range_proofs = encode_range_proof_vec(&self.range_proofs);
        let enc_data = EncRangeData {
            enc_values: enc_values,
            rand_proof: rand_proofs,
            range_proof: range_proofs,
            range_bits: self.prove_range as i32,
            check_percentage: self.check_percentage as i32,
        };
        let mut buffer = Vec::with_capacity(enc_data.encoded_len() + 1);
        let _res = enc_data.encode_length_delimited(&mut buffer);
        buffer
    }

    pub fn deserialize(data: &[u8]) -> EncParamsRange {
        let msg = EncRangeData::decode_length_delimited(&mut Cursor::new(data)).unwrap();
        let enc_values = decode_el_gamal_vec(&msg.enc_values);
        let rand_proofs = decode_rand_proof_vec(&msg.rand_proof);
        let range_proofs = decode_range_proof_vec(&msg.range_proof);
        EncParamsRange {
            enc_values: enc_values,
            rand_proofs: rand_proofs,
            range_proofs: range_proofs,
            prove_range: msg.range_bits as usize,
            check_percentage: msg.check_percentage as usize,
        }
    }
}

#[derive(Clone)]
pub struct EncParamsL2 {
    pub enc_values: Vec<SquareRandProofCommitments>,
    pub square_proofs: Vec<SquareRandProof>,
    pub range_proofs: Vec<RangeProof>,
    pub square_range_proof: RangeProof,
    pub prove_range: usize,
    pub l2_prove_range: usize,
}

fn encode_l2enc_vec(values: &Vec<SquareRandProofCommitments>) -> Vec<u8> {
    let mut out = Vec::with_capacity(SquareRandProofCommitments::serialized_size() * values.len());
    values.iter().for_each(|x| out.extend(x.to_bytes().iter()));
    out
}

fn decode_l2enc_vec(encoded: &[u8]) -> Vec<SquareRandProofCommitments> {
    //TODO error handling
    let num_elems = encoded.len() / SquareRandProofCommitments::serialized_size();
    let mut out = Vec::with_capacity(num_elems);
    let size = SquareRandProofCommitments::serialized_size();
    for id in 0..num_elems {
        out.push(
            SquareRandProofCommitments::from_bytes(&encoded[id * size..((id + 1) * size)]).unwrap(),
        );
    }
    out
}

fn encode_square_proof_vec(values: &Vec<SquareRandProof>) -> Vec<u8> {
    let mut out = Vec::with_capacity(SquareRandProof::serialized_size() * values.len());
    values.iter().for_each(|x| out.extend(x.to_bytes().iter()));
    out
}

fn decode_square_proof_vec(encoded: &[u8]) -> Vec<SquareRandProof> {
    //TODO error handling
    let num_elems = encoded.len() / SquareRandProof::serialized_size();
    let mut out = Vec::with_capacity(num_elems);
    let size = SquareRandProof::serialized_size();
    for id in 0..num_elems {
        out.push(SquareRandProof::from_bytes(&encoded[id * size..((id + 1) * size)]).unwrap());
    }
    out
}

impl EncParamsL2 {
    pub fn encrypt(
        plaintext_vec: &Vec<f32>,
        blinding_vec: &Vec<Scalar>,
        prove_range: usize,
        n_partition: usize,
        l2_range: usize,
    ) -> Self {
        let rand_scalars = rnd_scalar_vec(plaintext_vec.len());
        let range_clipped = range_proof_vec::clip_f32_to_range_vec(plaintext_vec, prove_range);
        let (range_proofs, enc_com) = range_proof_vec::create_rangeproof(
            &range_clipped,
            blinding_vec,
            prove_range,
            n_partition,
        )
        .unwrap();
        let (sum_range_proofs, _sum_cm) = l2_range_proof_vec::create_rangeproof_l2(
            plaintext_vec,
            &rand_scalars,
            l2_range,
            n_partition,
        )
        .unwrap();
        let (rand_proofs, enc_update) = square_rand_proof_vec::create_l2rangeproof_vec_existing(
            plaintext_vec,
            enc_com,
            blinding_vec,
            &rand_scalars,
        )
        .unwrap();
        EncParamsL2 {
            enc_values: enc_update,
            square_proofs: rand_proofs,
            range_proofs: range_proofs,
            prove_range: prove_range,
            square_range_proof: sum_range_proofs,
            l2_prove_range: l2_range,
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let enc_values = encode_l2enc_vec(&self.enc_values);
        let square_proofs = encode_square_proof_vec(&self.square_proofs);
        let range_proofs = encode_range_proof_vec(&self.range_proofs);
        let enc_data = EncNormData {
            enc_values: enc_values,
            square_proof: square_proofs,
            range_proof: range_proofs,
            square_range_proof: self.square_range_proof.to_bytes(),
            range_bits: self.prove_range as i32,
            l2_range_bits: self.l2_prove_range as i32,
        };
        let mut buffer = Vec::with_capacity(enc_data.encoded_len() + 1);
        let _res = enc_data.encode_length_delimited(&mut buffer);
        buffer
    }

    pub fn deserialize(data: &[u8]) -> Self {
        let msg = EncNormData::decode_length_delimited(&mut Cursor::new(data)).unwrap();
        let enc_values = decode_l2enc_vec(&msg.enc_values);
        let square_proofs = decode_square_proof_vec(&msg.square_proof);
        let range_proofs = decode_range_proof_vec(&msg.range_proof);
        EncParamsL2 {
            enc_values: enc_values,
            square_proofs: square_proofs,
            range_proofs: range_proofs,
            prove_range: msg.range_bits as usize,
            square_range_proof: RangeProof::from_bytes(&msg.square_range_proof).unwrap(),
            l2_prove_range: msg.l2_range_bits as usize,
        }
    }

    pub fn get_sum_proof(&self) -> &RangeProof {
        self.range_proofs.last().unwrap()
    }

    pub fn get_range_proof_slice(&self) -> &[RangeProof] {
        &self.range_proofs[0..(self.range_proofs.len() - 1)]
    }
}

#[derive(Clone)]
pub struct PlainParams {
    pub content: Vec<f32>,
}

impl PlainParams {
    pub fn unity(size: usize) -> Self {
        // Some basic initialization, as all 0 makes initial training VERY slow.
        let normal = Normal::new(0.0, 0.05);

        return PlainParams {
            content: (0..size).map(|_| normal.sample(&mut rand::thread_rng()) as f32).collect(),
        };
    }

    pub fn into_vec(self) -> Vec<f32> {
        self.content
    }

    pub fn accumulate_other(&mut self, other: &PlainParams) -> bool {
        if other.content.len() != self.content.len() {
            return false;
        }
        self.content
            .iter_mut()
            .zip(other.content.iter())
            .for_each(|(local_v, other_v)| *local_v += other_v);
        return true;
    }

    pub fn ml_update_in_place(&mut self, other: &PlainParams, learning_rate: f32) -> bool {
        if other.content.len() != self.content.len() {
            return false;
        }
        self.content
            .iter_mut()
            .zip(other.content.iter())
            .for_each(|(local_v, other_v)| *local_v += other_v * learning_rate);
        return true;
    }

    pub fn multiply_inplace(&mut self, value: f32) {
        self.content.iter_mut().for_each(|val| *val *= value);
    }

    pub fn serialize(&self) -> Vec<u8> {
        //TODO: cloning is not necessary, but proto
        let block = FloatBlock {
            block_number: 0,
            floats: self.content.clone(),
        };
        let mut buffer = Vec::with_capacity(block.encoded_len() + 1);
        let _res = block.encode(&mut buffer);
        buffer
    }

    pub fn deserialize(data: &[u8]) -> Self {
        let msg = FloatBlock::decode(&mut Cursor::new(data)).unwrap();
        return PlainParams {
            content: msg.floats,
        };
    }
}

#[derive(Clone)]
pub struct GlobalModel {
    pub params: PlainParams,
    pub learning_rate: f32,
}

impl GlobalModel {
    pub fn new(size: usize, learning_rate: f32) -> Self {
        GlobalModel {
            params: PlainParams::unity(size),
            learning_rate: learning_rate,
        }
    }

    pub fn update(&mut self, aggregated_update: &PlainParams) -> bool {
        self.params
            .ml_update_in_place(aggregated_update, self.learning_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_helper(
        p1: &mut EncModelParams,
        p2: &EncModelParams,
        enc_type: &EncModelParamType,
        acumulator: &mut EncModelParamsAccumulator,
    ) {
        acumulator.accumulate_other(&p1);
        acumulator.accumulate_other(&p2);
        assert_eq!(&acumulator.extract(&BSGSTable::default()).unwrap()[..], &vec![2.0; 10][..]);
        assert!(p1.verifiable() == false);
        let ser = p1.serialize();
        println!("{}", ser.len());
        //assert_eq!(&acumulator.extract().unwrap()[..], &EncModelParams::deserialize(&enc_type, &ser).extract().unwrap()[..]);
    }

    #[test]
    fn test_plain_params() {
        let mut p1 = EncModelParams::Plain(PlainParams {
            content: vec![1.0; 10],
        });
        let p2 = EncModelParams::Plain(PlainParams {
            content: vec![1.0; 10],
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
