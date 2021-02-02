use super::flservice::{DataBlock, model_parameters::ModelParametersMeta};
pub struct DataBlockStorage {
    round_id : u32,
    expected_blocks : u32,
    block_counter : u32,
    data : Vec<u8>
}

impl DataBlockStorage {

    pub fn new()-> Self {
        DataBlockStorage {
            round_id : 0,
            expected_blocks : 0, 
            block_counter: 0, 
            data : Vec::new()}
    }

    pub fn done(&self) -> bool {
        self.expected_blocks == self.block_counter
    }

    pub fn apply(&mut self, data_block_msg : &DataBlock) -> bool {
        if data_block_msg.block_number != self.block_counter {
            return false;
        }
        self.data.extend_from_slice(&data_block_msg.data[..]); 
        self.block_counter += 1;      
        true
    }

    pub fn init(&mut self, meta : ModelParametersMeta) {
        self.expected_blocks = meta.num_blocks as u32;
        self.round_id = meta.round_id as u32;
        self.block_counter = 0;
        self.reset_mem();
    }

    pub fn reset_mem(&mut self) {
        self.data = Vec::new();
    }

    pub fn data_ref(&self) -> &Vec<u8> {
        &self.data
    }     

    pub fn verify_round(&self, round_id : u32) -> bool {
        self.round_id == round_id
    }

    pub fn get_round_id(&self) -> u32 {
        self.round_id
    }
}
