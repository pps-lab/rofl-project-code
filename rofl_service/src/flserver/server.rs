use super::{
    flservice::flservice_server::Flservice,
    params::{EncModelParamType, EncModelParams, PlainParams},
};
use super::{
    flservice::{
        model_parameters, server_model_data, status_message, train_request, train_response,
    },
    params::GlobalModel,
};
use super::{
    flservice::{
        Config, CryptoConfig, DataBlock, ModelConfig, ModelParameters, ModelSelection,
        ServerModelData, StatusMessage, TrainRequest, TrainResponse,
    },
    logs::TimeState,
};
use crate::flserver::params::EncModelParamsAccumulator;
use crate::flserver::util::DataBlockStorage;
use futures::Stream;
use rofl_crypto::bsgs32::BSGSTable;
use core::panic;
use std::pin::Pin;
use std::sync::atomic::{AtomicI16, Ordering};
use std::sync::{Arc, RwLock};
use std::{collections::HashMap, sync::Condvar};
use tokio::fs;
use tokio::sync::mpsc::Sender;
use tokio::{io::AsyncWriteExt, sync::mpsc, sync::watch};
use tonic::{Request, Response, Status, Streaming};

use log::info;

const CHAN_BUFFER_SIZE: usize = 100;
const NUM_PARAM_BYTES_PER_PACKET: usize = 1 << 20;
const BSGS_TABLE_SIZE: usize = 1 << 16;

#[derive(Clone, PartialEq)]
enum TrainingStatusType {
    Register,
    InProgress,
    Terminate,
    Done,
}

#[derive(Clone)]
pub struct TrainingState {
    model_id: i32,
    model_config: ModelConfig,
    num_params: i32,
    in_memory_rounds: i32,
    train_until_round: i32,
    terminate_on_done: bool,
    bsgs_table: Arc<BSGSTable>,
    training_status: Arc<RwLock<TrainingStatusType>>,
    crypto_config: CryptoConfig,
    global_model: Arc<RwLock<GlobalModel>>,
    aggregation_type: EncModelParamType,
    channels: Arc<RwLock<HashMap<i32, Sender<Result<TrainResponse, Status>>>>>,
    channels_observer: Arc<RwLock<Vec<Sender<Result<TrainResponse, Status>>>>>,
    rounds: Arc<RwLock<Vec<TrainingRoundState>>>,
    do_lazy: bool,
}

impl TrainingState {
    pub fn new(
        model_id: i32,
        model_config: ModelConfig,
        crypto_config: CryptoConfig,
        num_parmas: i32,
        in_memory_rounds: i32,
        train_until_round: i32,
        global_model: GlobalModel,
        terminate_on_done: bool,
        do_lazy_verification: bool,
    ) -> Self {
        TrainingState {
            model_id,
            model_config,
            num_params: num_parmas,
            in_memory_rounds,
            train_until_round,
            terminate_on_done,
            bsgs_table: Arc::new(BSGSTable::new(BSGS_TABLE_SIZE)),
            training_status: Arc::new(RwLock::new(TrainingStatusType::Register)),
            crypto_config: crypto_config.clone(),
            global_model: Arc::new(RwLock::new(global_model)),
            aggregation_type: EncModelParamType::get_type_from_int(&crypto_config.enc_type)
                .unwrap(),
            channels: Arc::new(RwLock::new(HashMap::new())),
            channels_observer: Arc::new(RwLock::new(Vec::new())),
            rounds: Arc::new(RwLock::new(Vec::new())),
            do_lazy: do_lazy_verification,
        }
    }

    fn set_training_status(&self, status: TrainingStatusType) {
        let tmp_state = Arc::clone(&self.training_status);
        let mut mut_ref_state = tmp_state.write().unwrap();
        *mut_ref_state = status;
    }

    fn get_training_status(&self) -> TrainingStatusType {
        let tmp_state = Arc::clone(&self.training_status);
        let ref_state = tmp_state.read().unwrap();
        ref_state.clone()
    }

    fn terminate_on_done(&self) -> bool {
        self.terminate_on_done
    }

    fn register_channel(
        &self,
        client_id: i32,
        sender: Sender<Result<TrainResponse, Status>>,
    ) -> usize {
        let tmp = Arc::clone(&self.channels);
        let mut tmp = tmp.write().unwrap();
        tmp.insert(client_id, sender);
        tmp.len()
    }

    fn register_observer_channel(&self, sender: Sender<Result<TrainResponse, Status>>) -> usize {
        let tmp = Arc::clone(&self.channels_observer);
        let mut tmp = tmp.write().unwrap();
        tmp.push(sender);
        tmp.len()
    }

    fn get_channels_to_broadcast(&self) -> Vec<Sender<Result<TrainResponse, Status>>> {
        let tmp = Arc::clone(&self.channels);
        let tmp = tmp.read().unwrap();
        let mut out: Vec<Sender<Result<TrainResponse, Status>>> = Vec::with_capacity(tmp.len());
        for sender in (*tmp).values() {
            out.push(sender.clone());
        }
        out
    }

    fn get_channels_observer_to_broadcast(&self) -> Vec<Sender<Result<TrainResponse, Status>>> {
        let tmp = Arc::clone(&self.channels_observer);
        let tmp = tmp.read().unwrap();
        let mut out: Vec<Sender<Result<TrainResponse, Status>>> = Vec::with_capacity(tmp.len());
        for sender in &*tmp {
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
        if self.do_lazy {
            if tmp.len() < 2 {
                None
            } else {
                Some(tmp[tmp.len() - 2].clone())
            }
        } else if tmp.len() < 1 {
            None
        } else {
            Some(tmp.last().unwrap().clone())
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
        self.set_training_status(TrainingStatusType::InProgress);
    }

    fn can_client_register(&self) -> bool {
        self.get_training_status() == TrainingStatusType::Register
    }

    fn should_terminate(&self) -> bool {
        self.get_training_status() == TrainingStatusType::Terminate
    }

    fn init_params(&self) -> PlainParams {
        PlainParams::unity(self.num_params as usize)
    }

    fn init_round(&self) {
        self.start_new_round(0);
    }

    fn start_new_round(&self, round_id: i32) {
        let tmp = Arc::clone(&self.rounds);
        let mut tmp = tmp.write().unwrap();

        tmp.push(TrainingRoundState::new(
            round_id,
            self.model_config.num_of_clients,
            EncModelParams::unity(&self.aggregation_type, self.num_params as usize),
        ));

        let to_keep = self.in_memory_rounds as usize;
        if tmp.len() > to_keep {
            for _it in 0..(tmp.len() - to_keep) {
                if tmp.first().unwrap().is_done() {
                    tmp.remove(0);
                } else {
                    break;
                }
            }
        }
    }

    fn update_global_model(&self, mut update: PlainParams) -> bool {
        // info!("Update params {}", update.content[0]);
        update.multiply_inplace(1.0 / self.get_num_clients() as f32);
        let tmp = Arc::clone(&self.global_model);
        let mut tmp = tmp.write().unwrap();
        tmp.update(&update)
    }

    async fn write_global_model_to_file(&self) {
        let file_name = format!("model_{}_round_{}.txt", self.model_id, self.get_round());
        let values = {
            let tmp = Arc::clone(&self.global_model);
            let tmp = tmp.read().unwrap();
            tmp.params.content.clone()
        };
        let mut file = fs::File::create(file_name.as_str()).await.unwrap();
        for result in &values {
            let _ok = file.write_all(&format!("{}\n", result).as_bytes()).await;
        }
    }

    async fn broadcast_done(&self) {
        let done_message = TrainResponse {
            param_message: Some(train_response::ParamMessage::DoneMessage(StatusMessage {
                status: status_message::Status::Done as i32,
            })),
        };
        let channels = self.get_channels_to_broadcast();
        for chan in &channels {
            let out = chan.clone();
            let _res = out.send(Ok(done_message.clone())).await;
        }
        let channels_observer = self.get_channels_observer_to_broadcast();
        for chan in &channels_observer {
            let out = chan.clone();
            let _res = out.send(Ok(done_message.clone())).await;
        }
    }

    async fn broadcast_global_model(&self) {
        let channels = self.get_channels_to_broadcast();
        let config_message = TrainResponse {
            param_message: Some(train_response::ParamMessage::Params(ServerModelData {
                model_message: Some(server_model_data::ModelMessage::Config(Config {
                    model_config: Some(self.model_config.clone()),
                    crypto_config: Some(self.crypto_config.clone()),
                })),
            })),
        };

        //Encode Params
        let buffer = {
            let tmp = Arc::clone(&self.global_model);
            let tmp = tmp.read().unwrap();
            tmp.params.serialize()
        };
        let len_buffer = buffer.len();
        let mut num_packets = len_buffer / NUM_PARAM_BYTES_PER_PACKET;
        if len_buffer % NUM_PARAM_BYTES_PER_PACKET != 0 {
            num_packets += 1;
        }

        let meta_packet = TrainResponse {
            param_message: Some(train_response::ParamMessage::Params(ServerModelData {
                model_message: Some(server_model_data::ModelMessage::ModelBlock(
                    ModelParameters {
                        param_message: Some(model_parameters::ParamMessage::ParamMeta(
                            model_parameters::ModelParametersMeta {
                                model_id: self.model_id,
                                round_id: self.get_round(),
                                num_blocks: num_packets as i32,
                            },
                        )),
                    },
                )),
            })),
        };

        let mut data_messages = Vec::new();
        for packet_num in 0..num_packets {
            let begin = packet_num * NUM_PARAM_BYTES_PER_PACKET;
            let end = (packet_num + 1) * NUM_PARAM_BYTES_PER_PACKET;
            let end = if end > len_buffer { len_buffer } else { end };
            let data_packet = TrainResponse {
                param_message: Some(train_response::ParamMessage::Params(ServerModelData {
                    model_message: Some(server_model_data::ModelMessage::ModelBlock(
                        ModelParameters {
                            param_message: Some(model_parameters::ParamMessage::ParamBlock(
                                DataBlock {
                                    block_number: packet_num as u32,
                                    data: buffer[begin..end].to_vec(),
                                },
                            )),
                        },
                    )),
                })),
            };
            data_messages.push(data_packet);
        }

        // Broadcast Messages
        // TODO: Too much clones happen here :(
        // Could that be improved?
        let data_messages_arc = Arc::new(data_messages);
        let mut join_handlers = Vec::with_capacity(channels.len());
        for chan in &channels {
            let out = chan.clone();
            let data_messages_local = Arc::clone(&data_messages_arc);
            let meta_packet_local = meta_packet.clone();
            let config_packet_local = config_message.clone();
            let fut = tokio::spawn(async move {
                let _res = out.send(Ok(config_packet_local)).await;
                let _res = out.send(Ok(meta_packet_local)).await;
                for data_packet in &*data_messages_local {
                    let _res = out.send(Ok(data_packet.clone())).await;
                }
            });
            join_handlers.push(fut);
        }
        for join_handler in join_handlers {
            let _res = join_handler.await;
        }

        // Broadcast to observers
        let observer_channels = self.get_channels_observer_to_broadcast();
        if !observer_channels.is_empty() {
            let mut join_handlers = Vec::with_capacity(channels.len());
            for chan in &observer_channels {
                let out = chan.clone();
                let data_messages_local = Arc::clone(&data_messages_arc);
                let meta_packet_local = meta_packet.clone();
                let config_packet_local = config_message.clone();
                let fut = tokio::spawn(async move {
                    let _res = out.send(Ok(config_packet_local)).await;
                    let _res = out.send(Ok(meta_packet_local)).await;
                    for data_packet in &*data_messages_local {
                        let _res = out.send(Ok(data_packet.clone())).await;
                    }
                });
                join_handlers.push(fut);
            }
            for join_handler in join_handlers {
                let _res = join_handler.await;
            }
        }
    }
}

fn create_blocking_thread_pool(num_threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap()
}

#[derive(Clone, PartialEq)]
enum RoundState {
    InProgress,
    Error(String),
    Done,
}

#[derive(Clone)]
pub struct TrainingRoundState {
    round_id: i32,
    expected_clients: i32,
    param_aggr: Arc<RwLock<EncModelParamsAccumulator>>,
    verifed_counter: Arc<AtomicI16>,
    done_counter: Arc<AtomicI16>,
    state: Arc<RwLock<RoundState>>,
    notify_aggr: Arc<(watch::Sender<bool>, watch::Receiver<bool>)>,
    notify_verify: Arc<(std::sync::Mutex<bool>, Condvar)>,
    time_state: TimeState,
}

async fn verify_wait(receiver: &mut watch::Receiver<bool>) {
    let done = {
        *receiver.borrow()
    };
    if !done {
        let _res = receiver.changed().await;
    }
}

impl TrainingRoundState {
    pub fn new(
        round_id: i32,
        expected_clients: i32,
        param_aggr: EncModelParamsAccumulator,
    ) -> Self {
        let out = TrainingRoundState {
            round_id,
            expected_clients,
            param_aggr: Arc::new(RwLock::new(param_aggr)),
            verifed_counter: Arc::new(AtomicI16::new(0)),
            done_counter: Arc::new(AtomicI16::new(0)),
            state: Arc::new(RwLock::new(RoundState::InProgress)),
            notify_aggr: Arc::new(watch::channel(false)),
            notify_verify: Arc::new((std::sync::Mutex::new(false), Condvar::new())),
            time_state: TimeState::new(),
        };
        out.time_state.record_instant();
        out
    }

    fn extract_model_data(&self, table: &BSGSTable) -> Option<PlainParams> {
        let tmp = Arc::clone(&self.param_aggr);
        let tmp = tmp.read().unwrap();
        tmp.extract(table).map(|content| PlainParams { content })
    }

    fn increment_done_counter(&self) -> i16 {
        let compl_counter = self.done_counter.clone();
        compl_counter.fetch_add(1, Ordering::SeqCst)
    }

    fn verifaction_done(&self) {
        let compl_counter = self.verifed_counter.clone();
        let done = compl_counter.fetch_add(1, Ordering::SeqCst);
        if done + 1 == self.expected_clients as i16 {
            {
                let (sender, _receivr) = &*Arc::clone(&self.notify_aggr);
                let _res = sender.send(true);
            }
            
            {
                let (tmp_notify, cond) = &*Arc::clone(&self.notify_verify);
                let mut started = tmp_notify.lock().unwrap();
                while !*started {
                    started = cond.wait(started).unwrap();
                }
            }
            let tmp_state = Arc::clone(&self.state);
            let mut mut_ref_state = tmp_state.write().unwrap();
            if let RoundState::InProgress = *mut_ref_state {
                *mut_ref_state = RoundState::Done;
            }
            info!("Update verifications done for round {}", self.round_id);
            self.time_state.record_instant();
            self.time_state.log_bench_times(self.round_id);
        }
    }

    fn verifaction_error(&self, error_msg: String) {
        let compl_counter = self.verifed_counter.clone();
        let tmp_state = Arc::clone(&self.state);
        let mut mut_ref_state = tmp_state.write().unwrap();
        let done = compl_counter.fetch_add(1, Ordering::SeqCst);
        *mut_ref_state = RoundState::Error(error_msg);

        if done + 1 == self.expected_clients as i16 {
            panic!("ERROR occured");
        }
    }

    fn is_done(&self) -> bool {
        let tmp_state = Arc::clone(&self.state);
        let ref_state = tmp_state.read().unwrap();
        *ref_state == RoundState::Done
    }

    async fn wait_for_verif_completion(&self) {
        let (_sender, receivr) = &*Arc::clone(&self.notify_aggr);
        verify_wait(&mut receivr.clone()).await;
    }

    fn notify_model_complete_for_round(&self) {
        let (tmp_notify, cond) = &*Arc::clone(&self.notify_verify);
        let mut started = tmp_notify.lock().unwrap();
        *started = true;
        cond.notify_all();
    }

    pub fn accumulate(&self, param_aggr: &EncModelParams) -> bool {
        let tmp = Arc::clone(&self.param_aggr);
        let mut tmp = tmp.write().unwrap();
        tmp.accumulate_other(param_aggr)
    }
}

pub struct DefaultFlService {
    training_states: Arc<RwLock<Vec<TrainingState>>>,
    verification_pool: Arc<rayon::ThreadPool>,
}

impl DefaultFlService {
    pub fn new(num_verif_threads: usize) -> Self {
        DefaultFlService {
            training_states: Arc::new(RwLock::new(Vec::new())),
            verification_pool: Arc::new(create_blocking_thread_pool(num_verif_threads)),
        }
    }

    pub fn register_new_trainig_state(&self, state: TrainingState) {
        let tmp = Arc::clone(&self.training_states);
        let mut list = tmp.write().unwrap();
        list.push(state);
    }

    pub fn get_training_state_for_model(&self, model_id: i32) -> Option<TrainingState> {
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
    type TrainModelStream =
        Pin<Box<dyn Stream<Item = Result<TrainResponse, Status>> + Send + Sync + 'static>>;
    type ObserverModelTrainingStream =
        Pin<Box<dyn Stream<Item = Result<TrainResponse, Status>> + Send + Sync + 'static>>;

    async fn train_model(
        &self,
        request: Request<Streaming<TrainRequest>>,
    ) -> Result<Response<Self::TrainModelStream>, Status> {
        let mut streamer = request.into_inner();
        let (tx, rx) = mpsc::channel(CHAN_BUFFER_SIZE);
        //info!("", client_id, init.model_id);

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

        info!(
            "Register Messages received from from client {} for model {}",
            client_id, init.model_id
        );

        //Add The client to the Training State
        let training_state = self.get_training_state_for_model(init.model_id).unwrap();

        // If the training is running, register is not possible
        if !training_state.can_client_register() {
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
        if num_channels == training_state.get_num_clients() as usize {
            info!("Initialize model and broadcast it to clients");
            training_state.set_traning_running();
            training_state.init_round();
            training_state.broadcast_global_model().await;
            info!("First training round has started");
        }

        let local_thread_pool = Arc::clone(&self.verification_pool);
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
                        match msg.param_message.unwrap() {
                            model_parameters::ParamMessage::ParamMeta(meta_msg) => {
                                block_storage.init(meta_msg)
                            }
                            model_parameters::ParamMessage::ParamBlock(block_msg) => {
                                if !block_storage.apply(&block_msg) {
                                    panic!("should not happen");
                                }
                                if block_storage.done() {
                                    // Handle a completed model transfer
                                    info!("Received model params from Client {}", client_id);
                                    let local_group_state =
                                        training_state_local.get_current_round_state();
                                    if local_group_state.is_none() {
                                        todo!();
                                    }
                                    let local_group_state = local_group_state.unwrap();
                                    let local_blocking_group_state = local_group_state.clone();
                                    if !block_storage
                                        .verify_round(local_group_state.round_id as u32)
                                    {
                                        todo!();
                                    }
                                    let local_enc_params = Arc::new(EncModelParams::deserialize(
                                        &training_state_local.aggregation_type,
                                        block_storage.data_ref(),
                                    ));
                                    block_storage.reset_mem();

                                    let last_group_state =
                                        training_state_local.get_previous_round_state();

                                    let blocking_enc_params = Arc::clone(&local_enc_params);
                                    let wait_for_last_verify_to_comlete =
                                        training_state_local.do_lazy;

                                    if blocking_enc_params.verifiable() {
                                        let local_thread_pool_m = Arc::clone(&local_thread_pool);
                                        tokio::spawn(async move {
                                            if wait_for_last_verify_to_comlete {
                                                if let Some(last_group_state) = last_group_state {
                                                    last_group_state
                                                    .wait_for_verif_completion()
                                                    .await;
                                                }
                                            }
                                            local_thread_pool_m.as_ref().spawn(move || {
                                                let ok = blocking_enc_params.verify();
                                                if ok {
                                                    info!(
                                                        "Model of client {} for round {} is valid!",
                                                        client_id, local_blocking_group_state.round_id
                                                    );
                                                    local_blocking_group_state.verifaction_done();
                                                } else {
                                                    info!(
                                                        "Model of client {} for round {} is not valid!",
                                                        client_id, local_blocking_group_state.round_id
                                                    );
                                                    local_blocking_group_state.verifaction_error(
                                                        format!(
                                                        "Verification for client {} round {} failed",
                                                        client_id, local_blocking_group_state.round_id
                                                    ),
                                                    );
                                                }
                                            });
                                        });
                                    } else {
                                        tokio::spawn(async move {
                                            local_blocking_group_state.verifaction_done();
                                        });
                                    }

                                    let local_enc_params_aggr = Arc::clone(&local_enc_params);
                                    let model = if local_group_state
                                        .accumulate(&local_enc_params_aggr)
                                    {
                                        let done = local_group_state.increment_done_counter();
                                        info!(
                                            "Aggregate Model of client {} for round {}",
                                            client_id, local_group_state.round_id
                                        );
                                        let mut model = None;
                                        if done + 1 == local_group_state.expected_clients as i16 {
                                            // Aggregation has finished for this round, start the next round
                                            info!(
                                                "Start extracting plaintext model for round {}",
                                                local_group_state.round_id
                                            );
                                            local_group_state.time_state.record_instant();
                                            let bsgs_table_ref =
                                                Arc::clone(&training_state_local.bsgs_table);
                                            model = local_group_state
                                                .extract_model_data(bsgs_table_ref.as_ref());
                                            if model.is_none() {
                                                todo!();
                                            }
                                            info!(
                                                "Plaintext model extracted for round {}",
                                                local_group_state.round_id
                                            );
                                        }
                                        model
                                    } else {
                                        panic!("Should not happen no error handling yet");
                                    };

                                    if let Some(params) = model {
                                        // wait for verify to complete
                                        // todo: add suport for lacy verification
                                        let last_group_state =
                                            training_state_local.get_previous_round_state();
                                        // round is done, broadcast the new model
                                        training_state_local.update_global_model(params);
                                        local_group_state.time_state.record_instant();

                                        local_group_state.notify_model_complete_for_round();

                                        if let Some(last_group_state) = last_group_state {
                                            last_group_state
                                                .wait_for_verif_completion()
                                                .await;
                                        }

                                        // have we reached the end?
                                        if training_state_local.train_until_round
                                            <= local_group_state.round_id + 1
                                            || training_state_local.should_terminate()
                                        {
                                            training_state_local.broadcast_done().await;
                                            training_state_local
                                                .set_training_status(TrainingStatusType::Done);
                                            info!(
                                                "Training for model {} is done, max round reached",
                                                training_state_local.model_id
                                            );
                                            training_state_local.write_global_model_to_file().await;
                                            if training_state_local.terminate_on_done() {
                                                std::process::exit(0);
                                            } else {
                                                break;
                                            }
                                        }

                                        training_state_local
                                            .start_new_round(local_group_state.round_id + 1);
                                        info!(
                                            "Broadcast new model for round {}",
                                            local_group_state.round_id + 1
                                        );
                                        training_state_local.broadcast_global_model().await;
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
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }

    async fn terminate_model_training(
        &self,
        request: Request<ModelSelection>,
    ) -> Result<Response<StatusMessage>, Status> {
        let selection = request.into_inner();
        let training_state = self.get_training_state_for_model(selection.model_id);
        match training_state {
            Some(state) => {
                state.set_training_status(TrainingStatusType::Terminate);
                Ok(Response::new(StatusMessage {
                    status: status_message::Status::Ok as i32,
                }))
            }
            None => Ok(Response::new(StatusMessage {
                status: status_message::Status::Nok as i32,
            })),
        }
    }

    async fn observer_model_training(
        &self,
        request: Request<ModelSelection>,
    ) -> Result<Response<Self::ObserverModelTrainingStream>, Status> {
        let selection = request.into_inner();
        let training_state = self.get_training_state_for_model(selection.model_id);
        let (tx, rx) = mpsc::channel(CHAN_BUFFER_SIZE);
        match training_state {
            Some(state) => {
                state.register_observer_channel(tx);
                Ok(Response::new(Box::pin(
                    tokio_stream::wrappers::ReceiverStream::new(rx),
                )))
            }
            None => Err(Status::invalid_argument("Model does not exists")),
        }
    }
}