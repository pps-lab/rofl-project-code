use crate::flserver::params::EncModelParamsAccumulator;
use crate::flserver::util::DataBlockStorage;
use tokio::sync::mpsc;
use tonic::{Request, Response, Status, Streaming};
use super::{flservice::flservice_server::Flservice, params::{EncModelParamType, EncModelParams, PlainParams}};
use super::flservice::{DataBlock, Config, ModelConfig, CryptoConfig, ServerModelData, WorkerRegisterMessage, ModelRegisterResponse, StatusMessage, ModelParameters, TrainRequest, TrainResponse, ModelSelection};
use super::flservice::{train_request, train_response, server_model_data, model_parameters};
use std::sync::{Mutex, Arc, RwLock};
use std::sync::atomic::{AtomicI16, AtomicBool, Ordering};
use std::collections::HashMap;
use std::iter::FromIterator;
use tokio::sync::mpsc::Sender;
use tokio::sync::Notify;

const CHAN_BUFFER_SIZE: usize = 100;
const NUM_PARAM_BYTES_PER_PACKET: usize = 1 << 20;

#[derive(Clone)]
pub struct TrainingState {
    model_id : i32,
    model_config : ModelConfig,
    num_params : i32,
    in_memory_rounds : i32,
    in_training :  Arc<AtomicBool>,
    crypto_config : CryptoConfig,
    aggregation_type : EncModelParamType,
    channels :  Arc<RwLock<HashMap<i32, Sender<Result<TrainResponse, Status>>>>>,
    rounds :  Arc<RwLock<Vec<TrainingRoundState>>>,
}

impl TrainingState {

    pub fn new(model_id : i32, model_config : ModelConfig, crypto_config : CryptoConfig, num_parmas : i32, in_memory_rounds : i32) -> Self {
        TrainingState {
            model_id : model_id,
            model_config : model_config,
            num_params : num_parmas,
            in_memory_rounds : in_memory_rounds,
            in_training :  Arc::new(AtomicBool::new(false)),
            crypto_config : crypto_config.clone(),
            aggregation_type : EncModelParamType::get_type_from_int(&crypto_config.enc_type).unwrap(),
            channels :  Arc::new(RwLock::new(HashMap::new())),
            rounds :  Arc::new(RwLock::new(Vec::new())),
        }
    }

    fn register_channel(&self, client_id : i32, sender : Sender<Result<TrainResponse, Status>>) -> usize {
        let tmp = Arc::clone(&self.channels);
        let mut tmp = tmp.write().unwrap();
        tmp.insert(client_id, sender);
        tmp.len()
    }

    fn get_channels_to_broadcast(&self) -> Vec<Sender<Result<TrainResponse, Status>>> {
        let tmp = Arc::clone(&self.channels);
        let tmp = tmp.read().unwrap();
        let mut out : Vec<Sender<Result<TrainResponse, Status>>> = Vec::with_capacity(tmp.len());
        for (_key , sender) in  &*tmp {
            out.push(sender.clone());
        }
        out
    }

    fn get_current_round_state(&self) -> Option<TrainingRoundState> {
        let tmp = Arc::clone(&self.rounds);
        let tmp = tmp.read().unwrap();
        if tmp.is_empty() {
            None
        } else {
            Some(tmp.last().unwrap().clone())
        }
    }

    fn get_previous_round_state(&self) -> Option<TrainingRoundState> {
        let tmp = Arc::clone(&self.rounds);
        let tmp = tmp.read().unwrap();
        if tmp.len() < 2 {
            None
        } else {
            Some(tmp[tmp.len() - 2].clone())
        }
    }

    fn get_num_clients(&self) -> i32 {
        self.model_config.num_of_clients
    }

    fn get_round(&self) -> i32 {
        let state = self.get_current_round_state();
        if state.is_none() {
            return 0;
        }
        state.unwrap().round_id
    }

    fn set_traning_running(&self) {
        let tmp = Arc::clone(&self.in_training);
        tmp.store(true, Ordering::SeqCst);
    }

    fn init_params(&self) -> PlainParams {
        return PlainParams::unity(self.num_params as usize);
    }

    fn init_round(&self) {
        self.start_new_round(0);
    }

    fn start_new_round(&self, round_id : i32) {
        let tmp = Arc::clone(&self.rounds);
        let mut tmp = tmp.write().unwrap();

        tmp.push(TrainingRoundState::new(
            round_id,
            self.model_config.num_of_clients,
            EncModelParams::unity(&self.aggregation_type, self.num_params as usize),
        ));

        let to_keep =  self.in_memory_rounds as usize;
        if tmp.len() > to_keep {
            for it in 0..(tmp.len() - to_keep) {
                if tmp.first().unwrap().is_done() {
                    tmp.remove(0);
                } else {
                    break;
                }
            }
        }
    }

    async fn broadcast_models(&self, params : &PlainParams) {
        let channels = self.get_channels_to_broadcast();
        let config_message = TrainResponse {
            param_message : Some(train_response::ParamMessage::Params(ServerModelData {
                model_message : Some(server_model_data::ModelMessage::Config(Config { 
                    model_config: Some(self.model_config.clone()), 
                    crypto_config : Some(self.crypto_config.clone())
                }))
            }))
        };

        // Broadcast config message
        for chan in &channels {
            let mut out = chan.clone();
            let _res = out.send(Ok(config_message.clone())).await;
        }

        //Encode Params
        let buffer = params.serialize();
        let len_buffer = buffer.len();
        let mut num_packets = len_buffer / NUM_PARAM_BYTES_PER_PACKET;
        if len_buffer % NUM_PARAM_BYTES_PER_PACKET != 0 {
            num_packets += 1;
        }

        let meta_packet = TrainResponse {
            param_message : Some(train_response::ParamMessage::Params(ServerModelData {
                model_message : Some(server_model_data::ModelMessage::ModelBlock(ModelParameters { 
                    param_message: Some(model_parameters::ParamMessage::ParamMeta(model_parameters::ModelParametersMeta {
                        model_id : self.model_id, 
                        round_id : self.get_round(),
                        num_blocks : num_packets as i32,
                    })), 
                }))
            }))
        };

        // Broadcast Params message
        // TODO: Too much clones happen here :(
        // Could that be improved?
        for packet_num in 0..num_packets {
            let begin = packet_num * NUM_PARAM_BYTES_PER_PACKET;
            let end = (packet_num + 1) * NUM_PARAM_BYTES_PER_PACKET;
            let end = if end > len_buffer { len_buffer } else { end };
            
            let data_packet = TrainResponse {
                param_message : Some(train_response::ParamMessage::Params(ServerModelData {
                    model_message : Some(server_model_data::ModelMessage::ModelBlock(ModelParameters { 
                        param_message: Some(model_parameters::ParamMessage::ParamBlock(DataBlock {
                            block_number : packet_num as u32, 
                            data : Vec::from_iter(buffer[begin..end].iter().cloned())
                        })), 
                    }))
                }))
            };
            for chan in &channels {
                let mut out = chan.clone();
                if packet_num == 0 {
                    let _res = out.send(Ok(meta_packet.clone())).await;
                }
                
                let _res = out.send(Ok(data_packet.clone())).await;
            }
            
        }
    }
}

#[derive(Clone, PartialEq)]
enum RoundState {
    InProgress,
    Error(String),
    Done,
}
#[derive(Clone)]
pub struct TrainingRoundState {
    round_id : i32,
    expected_clients : i32,
    param_aggr : Arc<RwLock<EncModelParamsAccumulator>>,
    verifed_counter : Arc<AtomicI16>, 
    done_counter : Arc<AtomicI16>, 
    state : Arc<RwLock<RoundState>>,
    notify : Arc<Notify>
}

impl TrainingRoundState {
    pub fn new( round_id : i32, expected_clients : i32, param_aggr : EncModelParamsAccumulator) -> Self {
        return TrainingRoundState {
            round_id : round_id,
            expected_clients : expected_clients,
            param_aggr : Arc::new(RwLock::new(param_aggr)),
            verifed_counter : Arc::new(AtomicI16::new(0)),
            done_counter : Arc::new(AtomicI16::new(0)),
            state : Arc::new(RwLock::new(RoundState::InProgress)),
            notify : Arc::new(Notify::new())
        };
    }

    fn extract_model_data(&self) -> Option<PlainParams> {
        let tmp = Arc::clone(&self.param_aggr);
        let tmp = tmp.read().unwrap();
        match tmp.extract() {
            None => None,
            Some(content) => Some(PlainParams { 
                content : content
            }),
        }
    }

    fn increment_done_counter(&self) -> i16 {
        let compl_counter = self.done_counter.clone();
        compl_counter.fetch_add(1, Ordering::SeqCst)
    }

    fn verifaction_done(&self) -> () {
        let compl_counter = self.verifed_counter.clone();
        let done = compl_counter.fetch_add(1, Ordering::SeqCst);
        if done + 1 == self.expected_clients as i16 {
            let tmp_state = Arc::clone(&self.state);
            let mut mut_ref_state = tmp_state.write().unwrap();
            if let RoundState::InProgress = *mut_ref_state {
                *mut_ref_state = RoundState::Done;
            }
            let tmp_notify = Arc::clone(&self.notify);
            tmp_notify.notify();
        }
    }

    fn verifaction_error(&self, error_msg : String) -> () {
        let compl_counter = self.verifed_counter.clone();
        let tmp_state = Arc::clone(&self.state);
        let mut mut_ref_state = tmp_state.write().unwrap();
        let done = compl_counter.fetch_add(1, Ordering::SeqCst);
        *mut_ref_state = RoundState::Error(error_msg);
        
        if done + 1 == self.expected_clients as i16 {
            let tmp_notify = Arc::clone(&self.notify);
            tmp_notify.notify();
        }
    }

    fn is_done(&self) -> bool {
        let tmp_state = Arc::clone(&self.state);
        let ref_state = tmp_state.read().unwrap();
        *ref_state == RoundState::Done
    }

    async fn wait_for_verif_completion(&self) {
        let tmp_notify = Arc::clone(&self.notify);
        tmp_notify.notified().await;
    }

    pub fn accumulate(&self, param_aggr : &EncModelParams) -> bool {
        let tmp = Arc::clone(&self.param_aggr);
        let mut tmp = tmp.write().unwrap();
        tmp.accumulate_other(param_aggr)
    }
}

pub struct DefaultFlService {
    training_states : Arc<RwLock<Vec<TrainingState>>>,
}

impl DefaultFlService {
    pub fn new() -> Self {
        DefaultFlService {
            training_states : Arc::new(RwLock::new(Vec::new()))
        }
    }

    pub fn register_new_trainig_state(&self, state : TrainingState) {
        let tmp = Arc::clone(&self.training_states);
        let mut list = tmp.write().unwrap(); 
        list.push(state);
    }

    pub fn get_training_state_for_model(&self, model_id : i32) -> Option<TrainingState> {
        let tmp = Arc::clone(&self.training_states);
        let tmp = tmp.read().unwrap();
        for state in &*tmp {
            if state.model_id == model_id {
                return Some(state.clone());
            }
        }
        None
    }
}

#[tonic::async_trait]
impl Flservice for DefaultFlService {        
        type TrainModelStream = mpsc::Receiver<Result<TrainResponse, Status>>;
        type ObserverModelTrainingStream = mpsc::Receiver<Result<ServerModelData, Status>>;

        async fn train_model(
            &self,
            request: Request<Streaming<TrainRequest>>,
        ) -> Result<Response<Self::TrainModelStream>, Status> {
            let mut streamer = request.into_inner();
            let (mut tx, rx) = mpsc::channel(CHAN_BUFFER_SIZE);
            //println!("", client_id, init.model_id);

            // Handle Register Message verifed_counter
            let req = streamer.message().await.unwrap();
            let req = req.unwrap();
            // Check that the message is indeed a Register Message 
            let init = match req.param_message.unwrap() {
                train_request::ParamMessage::StatusMessage(_x) => None,
                train_request::ParamMessage::Params(_x) => None,
                train_request::ParamMessage::StartMessage(init_msg) => Some(init_msg),
            };
            // Error if not
            if init.is_none() {
                return Err(Status::invalid_argument("Should be an init message"));
            }
            let init = init.unwrap();

            let client_id = init.client_id;

            println!("Register Messages received from from client {} for model {}", client_id, init.model_id);

            //Add The client to the Training State
            let training_state = self.get_training_state_for_model(init.model_id).unwrap();
            let is_in_training = training_state.in_training.clone();

            // If the training is running, register is not possible
            if is_in_training.load(Ordering::SeqCst) {
                /*let _res = tx.send(Ok(TrainResponse {
                    param_message : Some(train_response::ParamMessage::StatusMessage(StatusMessage {
                        status : 0,
                        error_msg : String::from("Session is in progress"),
                    })),
                })).await;*/
                todo!();
            } 
            
            // Register the client
            let num_channels = training_state.register_channel(init.client_id, tx.clone());

            // Are all clients registered?
            if (num_channels == training_state.get_num_clients() as usize) {
               println!("Initialize model and broadcast it to clients");
               training_state.set_traning_running();
               training_state.init_round();
               training_state.broadcast_models(&training_state.init_params()).await;
               println!("First training round has started");
            }


            let training_state_local = training_state.clone();
            tokio::spawn(async move {
                let mut block_storage = DataBlockStorage::new();
                while let Ok(message) = streamer.message().await {
                    if message.is_none() {
                        continue;
                    }
                    let req = message.unwrap();
                    match req.param_message.unwrap() {
                        train_request::ParamMessage::StartMessage(_) => {
                            panic!("This is not covered yet")
                        }
                        train_request::ParamMessage::Params(msg) => {
                            match  msg.param_message.unwrap() {
                                model_parameters::ParamMessage::ParamMeta(meta_msg) => {
                                    block_storage.init(meta_msg)
                                }
                                model_parameters::ParamMessage::ParamBlock(block_msg) => {
                                    if !block_storage.apply(&block_msg) {
                                        panic!("should not happen");
                                    }
                                    if block_storage.done() {
                                        // Handle a completed model transfer
                                        println!("Received model params from Client {}", client_id);
                                        let local_group_state = training_state_local.get_current_round_state();
                                        if local_group_state.is_none() {
                                            todo!();
                                        }
                                        let local_group_state= local_group_state.unwrap();
                                        let local_blocking_group_state = local_group_state. clone();
                                        if !block_storage.verify_round(local_group_state.round_id as u32) {
                                            todo!();
                                        }
                                        let local_enc_params = Arc::new(EncModelParams::deserialize(&training_state_local.aggregation_type, block_storage.data_ref()));
                                        block_storage.reset_mem();

                                        //TODO: maybe use a seperate thread pool for verification
                                        let blocking_enc_params = Arc::clone(&local_enc_params);
                                        if blocking_enc_params.verifiable() {
                                            let res = tokio::task::spawn_blocking(move || {  
                                                let ok = blocking_enc_params.verify();
                                                if ok {
                                                    println!("Model of client {} for round {} is valid!", client_id, local_blocking_group_state.round_id);
                                                    local_blocking_group_state.verifaction_done();
                                                } else {
                                                    println!("Model of client {} for round {} is not valid!", client_id, local_blocking_group_state.round_id);
                                                    local_blocking_group_state.verifaction_error(format!("Verification for client {} round {} failed", client_id, local_blocking_group_state.round_id));
                                                }
                                            });
                                        } else {
                                            local_blocking_group_state.verifaction_done();
                                        }

                                        let local_enc_params_aggr = Arc::clone(&local_enc_params);
                                        let model = if local_group_state.accumulate(&local_enc_params_aggr) {
                                            let done = local_group_state.increment_done_counter();
                                            println!("Aggregate Model of client {} for round {}", client_id, local_group_state.round_id);
                                            let mut model = None;
                                            if done + 1 == local_group_state.expected_clients as i16 {
                                                // Aggregation has finished for this round, start the next round
                                                model = local_group_state.extract_model_data();
                                                if model.is_none() {
                                                    todo!();
                                                }
                                            }
                                            model
                                        } else {
                                            None
                                        };

                                        if model.is_some() {
                                            // wait for verify to complete
                                            // todo: add suport for lacy verification
                                            let last_group_state = training_state_local.get_previous_round_state();
                                            if last_group_state.is_some() {
                                                last_group_state.unwrap().wait_for_verif_completion().await;
                                            }
                                            // round is done, broadcast the new model
                                            println!("All client models received for round {}", local_group_state.round_id);
                                            let params = model.unwrap();
                                            training_state_local.start_new_round(local_group_state.round_id + 1);
                                            training_state_local.broadcast_models(&params).await;
                                        }

                                    }
                                }
                            }
                        }
                        train_request::ParamMessage::StatusMessage(_) => {
                            panic!("This is not covered yet")
                        }
                    }
                }
            });

            Ok(Response::new(rx))
        }

        async fn terminate_model_training(
            &self,
            request: Request<ModelSelection>,
        ) -> Result<Response<StatusMessage>, Status> {
            todo!();
        }

        async fn observer_model_training(
            &self,
            request: Request<Streaming<ModelSelection>>,
        ) -> Result<Response<Self::ObserverModelTrainingStream>, Status> {
            todo!();
        }
}