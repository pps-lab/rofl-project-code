#[derive(Clone, PartialEq, ::prost::Message)]
pub struct MetaFloatBlockMessage {
    #[prost(int32, tag = "1")]
    pub model_id: i32,
    #[prost(int32, tag = "2")]
    pub round_id: i32,
    #[prost(int32, tag = "3")]
    pub num_blocks: i32,
    #[prost(int32, tag = "4")]
    pub num_floats: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct FloatBlock {
    #[prost(uint32, tag = "1")]
    pub block_number: u32,
    #[prost(float, repeated, tag = "2")]
    pub floats: ::std::vec::Vec<f32>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ClientModelMessage {
    #[prost(oneof = "client_model_message::ModelMessage", tags = "1, 2, 3")]
    pub model_message: ::std::option::Option<client_model_message::ModelMessage>,
}
pub mod client_model_message {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum ModelMessage {
        #[prost(message, tag = "1")]
        Config(super::ModelConfig),
        #[prost(message, tag = "2")]
        MetaBlockMessage(super::MetaFloatBlockMessage),
        #[prost(message, tag = "3")]
        ModelBlock(super::FloatBlock),
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelConfig {
    #[prost(int32, tag = "1")]
    pub num_of_clients: i32,
    #[prost(int32, tag = "2")]
    pub client_batch_size: i32,
    #[prost(int32, tag = "3")]
    pub num_local_epochs: i32,
    #[prost(string, tag = "4")]
    pub optimizer: std::string::String,
    #[prost(float, tag = "5")]
    pub learning_rate: f32,
    #[prost(string, tag = "6")]
    pub loss: std::string::String,
    #[prost(string, tag = "7")]
    pub metrics: std::string::String,
    #[prost(bool, tag = "8")]
    pub image_augmentation: bool,
    #[prost(float, tag = "9")]
    pub lr_decay: f32,
    #[prost(int32, tag = "10")]
    pub model_id: i32,
    #[prost(bool, tag = "11")]
    pub probabilistic_quantization: bool,
    #[prost(int32, tag = "12")]
    pub fp_bits: i32,
    #[prost(int32, tag = "13")]
    pub fp_frac: i32,
    #[prost(int32, tag = "14")]
    pub range_bits: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct CryptoConfig {
    #[prost(int32, tag = "1")]
    pub value_range: i32,
    #[prost(int32, tag = "2")]
    pub n_partition: i32,
    #[prost(int32, tag = "3")]
    pub l2_value_range: i32,
    #[prost(int32, tag = "4")]
    pub check_percentage: i32,
    #[prost(int32, tag = "5")]
    pub enc_type: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct DataBlock {
    #[prost(uint32, tag = "1")]
    pub block_number: u32,
    #[prost(bytes, tag = "2")]
    pub data: std::vec::Vec<u8>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EncRangeData {
    #[prost(bytes, tag = "1")]
    pub enc_values: std::vec::Vec<u8>,
    #[prost(bytes, tag = "2")]
    pub rand_proof: std::vec::Vec<u8>,
    #[prost(bytes, repeated, tag = "3")]
    pub range_proof: ::std::vec::Vec<std::vec::Vec<u8>>,
    #[prost(int32, tag = "4")]
    pub range_bits: i32,
    #[prost(int32, tag = "5")]
    pub check_percentage: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct EncNormData {
    #[prost(bytes, tag = "1")]
    pub enc_values: std::vec::Vec<u8>,
    #[prost(bytes, tag = "2")]
    pub square_proof: std::vec::Vec<u8>,
    #[prost(bytes, repeated, tag = "3")]
    pub range_proof: ::std::vec::Vec<std::vec::Vec<u8>>,
    #[prost(bytes, tag = "4")]
    pub square_range_proof: std::vec::Vec<u8>,
    #[prost(int32, tag = "5")]
    pub range_bits: i32,
    #[prost(int32, tag = "6")]
    pub l2_range_bits: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct Config {
    #[prost(message, optional, tag = "1")]
    pub model_config: ::std::option::Option<ModelConfig>,
    #[prost(message, optional, tag = "2")]
    pub crypto_config: ::std::option::Option<CryptoConfig>,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ServerModelData {
    #[prost(oneof = "server_model_data::ModelMessage", tags = "1, 2")]
    pub model_message: ::std::option::Option<server_model_data::ModelMessage>,
}
pub mod server_model_data {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum ModelMessage {
        #[prost(message, tag = "1")]
        Config(super::Config),
        #[prost(message, tag = "2")]
        ModelBlock(super::ModelParameters),
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct WorkerRegisterMessage {
    #[prost(int32, tag = "1")]
    pub model_id: i32,
    #[prost(int32, tag = "2")]
    pub client_id: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelRegisterResponse {
    #[prost(bool, tag = "1")]
    pub success: bool,
    #[prost(int32, tag = "2")]
    pub model_id: i32,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct StatusMessage {
    #[prost(enumeration = "status_message::Status", tag = "1")]
    pub status: i32,
}
pub mod status_message {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, ::prost::Enumeration)]
    #[repr(i32)]
    pub enum Status {
        Ok = 0,
        Nok = 1,
        Late = 2,
        Done = 3,
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ErrorMessage {
    #[prost(int32, tag = "1")]
    pub version: i32,
    #[prost(string, tag = "2")]
    pub msg: std::string::String,
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelParameters {
    #[prost(oneof = "model_parameters::ParamMessage", tags = "1, 2")]
    pub param_message: ::std::option::Option<model_parameters::ParamMessage>,
}
pub mod model_parameters {
    #[derive(Clone, PartialEq, ::prost::Message)]
    pub struct ModelParametersMeta {
        #[prost(int32, tag = "1")]
        pub model_id: i32,
        #[prost(int32, tag = "2")]
        pub round_id: i32,
        #[prost(int32, tag = "3")]
        pub num_blocks: i32,
    }
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum ParamMessage {
        #[prost(message, tag = "1")]
        ParamMeta(ModelParametersMeta),
        #[prost(message, tag = "2")]
        ParamBlock(super::DataBlock),
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TrainRequest {
    #[prost(oneof = "train_request::ParamMessage", tags = "1, 2, 3")]
    pub param_message: ::std::option::Option<train_request::ParamMessage>,
}
pub mod train_request {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum ParamMessage {
        #[prost(message, tag = "1")]
        StartMessage(super::WorkerRegisterMessage),
        #[prost(message, tag = "2")]
        Params(super::ModelParameters),
        #[prost(message, tag = "3")]
        StatusMessage(super::StatusMessage),
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct TrainResponse {
    #[prost(oneof = "train_response::ParamMessage", tags = "1, 2, 3")]
    pub param_message: ::std::option::Option<train_response::ParamMessage>,
}
pub mod train_response {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum ParamMessage {
        #[prost(message, tag = "1")]
        Params(super::ServerModelData),
        #[prost(message, tag = "2")]
        ErrorMessage(super::ErrorMessage),
        #[prost(message, tag = "3")]
        DoneMessage(super::StatusMessage),
    }
}
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct ModelSelection {
    #[prost(int32, tag = "1")]
    pub model_id: i32,
}
#[doc = r" Generated client implementations."]
pub mod flservice_client {
    #![allow(unused_variables, dead_code, missing_docs)]
    use tonic::codegen::*;
    pub struct FlserviceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl FlserviceClient<tonic::transport::Channel> {
        #[doc = r" Attempt to create a new client by connecting to a given endpoint."]
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: std::convert::TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> FlserviceClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
        T::ResponseBody: Body + HttpBody + Send + 'static,
        T::Error: Into<StdError>,
        <T::ResponseBody as HttpBody>::Error: Into<StdError> + Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_interceptor(inner: T, interceptor: impl Into<tonic::Interceptor>) -> Self {
            let inner = tonic::client::Grpc::with_interceptor(inner, interceptor);
            Self { inner }
        }
        pub async fn train_model(
            &mut self,
            request: impl tonic::IntoStreamingRequest<Message = super::TrainRequest>,
        ) -> Result<tonic::Response<tonic::codec::Streaming<super::TrainResponse>>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static("/flservice.Flservice/TrainModel");
            self.inner
                .streaming(request.into_streaming_request(), path, codec)
                .await
        }
        pub async fn terminate_model_training(
            &mut self,
            request: impl tonic::IntoRequest<super::ModelSelection>,
        ) -> Result<tonic::Response<super::StatusMessage>, tonic::Status> {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/flservice.Flservice/TerminateModelTraining");
            self.inner.unary(request.into_request(), path, codec).await
        }
        pub async fn observer_model_training(
            &mut self,
            request: impl tonic::IntoStreamingRequest<Message = super::ModelSelection>,
        ) -> Result<tonic::Response<tonic::codec::Streaming<super::ServerModelData>>, tonic::Status>
        {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path =
                http::uri::PathAndQuery::from_static("/flservice.Flservice/ObserverModelTraining");
            self.inner
                .streaming(request.into_streaming_request(), path, codec)
                .await
        }
    }
    impl<T: Clone> Clone for FlserviceClient<T> {
        fn clone(&self) -> Self {
            Self {
                inner: self.inner.clone(),
            }
        }
    }
    impl<T> std::fmt::Debug for FlserviceClient<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "FlserviceClient {{ ... }}")
        }
    }
}
#[doc = r" Generated client implementations."]
pub mod fl_client_train_service_client {
    #![allow(unused_variables, dead_code, missing_docs)]
    use tonic::codegen::*;
    pub struct FlClientTrainServiceClient<T> {
        inner: tonic::client::Grpc<T>,
    }
    impl FlClientTrainServiceClient<tonic::transport::Channel> {
        #[doc = r" Attempt to create a new client by connecting to a given endpoint."]
        pub async fn connect<D>(dst: D) -> Result<Self, tonic::transport::Error>
        where
            D: std::convert::TryInto<tonic::transport::Endpoint>,
            D::Error: Into<StdError>,
        {
            let conn = tonic::transport::Endpoint::new(dst)?.connect().await?;
            Ok(Self::new(conn))
        }
    }
    impl<T> FlClientTrainServiceClient<T>
    where
        T: tonic::client::GrpcService<tonic::body::BoxBody>,
        T::ResponseBody: Body + HttpBody + Send + 'static,
        T::Error: Into<StdError>,
        <T::ResponseBody as HttpBody>::Error: Into<StdError> + Send,
    {
        pub fn new(inner: T) -> Self {
            let inner = tonic::client::Grpc::new(inner);
            Self { inner }
        }
        pub fn with_interceptor(inner: T, interceptor: impl Into<tonic::Interceptor>) -> Self {
            let inner = tonic::client::Grpc::with_interceptor(inner, interceptor);
            Self { inner }
        }
        pub async fn train_for_round(
            &mut self,
            request: impl tonic::IntoStreamingRequest<Message = super::ClientModelMessage>,
        ) -> Result<
            tonic::Response<tonic::codec::Streaming<super::ClientModelMessage>>,
            tonic::Status,
        > {
            self.inner.ready().await.map_err(|e| {
                tonic::Status::new(
                    tonic::Code::Unknown,
                    format!("Service was not ready: {}", e.into()),
                )
            })?;
            let codec = tonic::codec::ProstCodec::default();
            let path = http::uri::PathAndQuery::from_static(
                "/flservice.FLClientTrainService/TrainForRound",
            );
            self.inner
                .streaming(request.into_streaming_request(), path, codec)
                .await
        }
    }
    impl<T: Clone> Clone for FlClientTrainServiceClient<T> {
        fn clone(&self) -> Self {
            Self {
                inner: self.inner.clone(),
            }
        }
    }
    impl<T> std::fmt::Debug for FlClientTrainServiceClient<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "FlClientTrainServiceClient {{ ... }}")
        }
    }
}
#[doc = r" Generated server implementations."]
pub mod flservice_server {
    #![allow(unused_variables, dead_code, missing_docs)]
    use tonic::codegen::*;
    #[doc = "Generated trait containing gRPC methods that should be implemented for use with FlserviceServer."]
    #[async_trait]
    pub trait Flservice: Send + Sync + 'static {
        #[doc = "Server streaming response type for the TrainModel method."]
        type TrainModelStream: Stream<Item = Result<super::TrainResponse, tonic::Status>>
            + Send
            + Sync
            + 'static;
        async fn train_model(
            &self,
            request: tonic::Request<tonic::Streaming<super::TrainRequest>>,
        ) -> Result<tonic::Response<Self::TrainModelStream>, tonic::Status>;
        async fn terminate_model_training(
            &self,
            request: tonic::Request<super::ModelSelection>,
        ) -> Result<tonic::Response<super::StatusMessage>, tonic::Status>;
        #[doc = "Server streaming response type for the ObserverModelTraining method."]
        type ObserverModelTrainingStream: Stream<Item = Result<super::ServerModelData, tonic::Status>>
            + Send
            + Sync
            + 'static;
        async fn observer_model_training(
            &self,
            request: tonic::Request<tonic::Streaming<super::ModelSelection>>,
        ) -> Result<tonic::Response<Self::ObserverModelTrainingStream>, tonic::Status>;
    }
    #[derive(Debug)]
    pub struct FlserviceServer<T: Flservice> {
        inner: _Inner<T>,
    }
    struct _Inner<T>(Arc<T>, Option<tonic::Interceptor>);
    impl<T: Flservice> FlserviceServer<T> {
        pub fn new(inner: T) -> Self {
            let inner = Arc::new(inner);
            let inner = _Inner(inner, None);
            Self { inner }
        }
        pub fn with_interceptor(inner: T, interceptor: impl Into<tonic::Interceptor>) -> Self {
            let inner = Arc::new(inner);
            let inner = _Inner(inner, Some(interceptor.into()));
            Self { inner }
        }
    }
    impl<T, B> Service<http::Request<B>> for FlserviceServer<T>
    where
        T: Flservice,
        B: HttpBody + Send + Sync + 'static,
        B::Error: Into<StdError> + Send + 'static,
    {
        type Response = http::Response<tonic::body::BoxBody>;
        type Error = Never;
        type Future = BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, req: http::Request<B>) -> Self::Future {
            let inner = self.inner.clone();
            match req.uri().path() {
                "/flservice.Flservice/TrainModel" => {
                    #[allow(non_camel_case_types)]
                    struct TrainModelSvc<T: Flservice>(pub Arc<T>);
                    impl<T: Flservice> tonic::server::StreamingService<super::TrainRequest> for TrainModelSvc<T> {
                        type Response = super::TrainResponse;
                        type ResponseStream = T::TrainModelStream;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<tonic::Streaming<super::TrainRequest>>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).train_model(request).await };
                            Box::pin(fut)
                        }
                    }
                    let inner = self.inner.clone();
                    let fut = async move {
                        let interceptor = inner.1;
                        let inner = inner.0;
                        let method = TrainModelSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = if let Some(interceptor) = interceptor {
                            tonic::server::Grpc::with_interceptor(codec, interceptor)
                        } else {
                            tonic::server::Grpc::new(codec)
                        };
                        let res = grpc.streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/flservice.Flservice/TerminateModelTraining" => {
                    #[allow(non_camel_case_types)]
                    struct TerminateModelTrainingSvc<T: Flservice>(pub Arc<T>);
                    impl<T: Flservice> tonic::server::UnaryService<super::ModelSelection>
                        for TerminateModelTrainingSvc<T>
                    {
                        type Response = super::StatusMessage;
                        type Future = BoxFuture<tonic::Response<Self::Response>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<super::ModelSelection>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut =
                                async move { (*inner).terminate_model_training(request).await };
                            Box::pin(fut)
                        }
                    }
                    let inner = self.inner.clone();
                    let fut = async move {
                        let interceptor = inner.1.clone();
                        let inner = inner.0;
                        let method = TerminateModelTrainingSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = if let Some(interceptor) = interceptor {
                            tonic::server::Grpc::with_interceptor(codec, interceptor)
                        } else {
                            tonic::server::Grpc::new(codec)
                        };
                        let res = grpc.unary(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                "/flservice.Flservice/ObserverModelTraining" => {
                    #[allow(non_camel_case_types)]
                    struct ObserverModelTrainingSvc<T: Flservice>(pub Arc<T>);
                    impl<T: Flservice> tonic::server::StreamingService<super::ModelSelection>
                        for ObserverModelTrainingSvc<T>
                    {
                        type Response = super::ServerModelData;
                        type ResponseStream = T::ObserverModelTrainingStream;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<tonic::Streaming<super::ModelSelection>>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut =
                                async move { (*inner).observer_model_training(request).await };
                            Box::pin(fut)
                        }
                    }
                    let inner = self.inner.clone();
                    let fut = async move {
                        let interceptor = inner.1;
                        let inner = inner.0;
                        let method = ObserverModelTrainingSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = if let Some(interceptor) = interceptor {
                            tonic::server::Grpc::with_interceptor(codec, interceptor)
                        } else {
                            tonic::server::Grpc::new(codec)
                        };
                        let res = grpc.streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                _ => Box::pin(async move {
                    Ok(http::Response::builder()
                        .status(200)
                        .header("grpc-status", "12")
                        .body(tonic::body::BoxBody::empty())
                        .unwrap())
                }),
            }
        }
    }
    impl<T: Flservice> Clone for FlserviceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self { inner }
        }
    }
    impl<T: Flservice> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone(), self.1.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: Flservice> tonic::transport::NamedService for FlserviceServer<T> {
        const NAME: &'static str = "flservice.Flservice";
    }
}
#[doc = r" Generated server implementations."]
pub mod fl_client_train_service_server {
    #![allow(unused_variables, dead_code, missing_docs)]
    use tonic::codegen::*;
    #[doc = "Generated trait containing gRPC methods that should be implemented for use with FlClientTrainServiceServer."]
    #[async_trait]
    pub trait FlClientTrainService: Send + Sync + 'static {
        #[doc = "Server streaming response type for the TrainForRound method."]
        type TrainForRoundStream: Stream<Item = Result<super::ClientModelMessage, tonic::Status>>
            + Send
            + Sync
            + 'static;
        async fn train_for_round(
            &self,
            request: tonic::Request<tonic::Streaming<super::ClientModelMessage>>,
        ) -> Result<tonic::Response<Self::TrainForRoundStream>, tonic::Status>;
    }
    #[derive(Debug)]
    pub struct FlClientTrainServiceServer<T: FlClientTrainService> {
        inner: _Inner<T>,
    }
    struct _Inner<T>(Arc<T>, Option<tonic::Interceptor>);
    impl<T: FlClientTrainService> FlClientTrainServiceServer<T> {
        pub fn new(inner: T) -> Self {
            let inner = Arc::new(inner);
            let inner = _Inner(inner, None);
            Self { inner }
        }
        pub fn with_interceptor(inner: T, interceptor: impl Into<tonic::Interceptor>) -> Self {
            let inner = Arc::new(inner);
            let inner = _Inner(inner, Some(interceptor.into()));
            Self { inner }
        }
    }
    impl<T, B> Service<http::Request<B>> for FlClientTrainServiceServer<T>
    where
        T: FlClientTrainService,
        B: HttpBody + Send + Sync + 'static,
        B::Error: Into<StdError> + Send + 'static,
    {
        type Response = http::Response<tonic::body::BoxBody>;
        type Error = Never;
        type Future = BoxFuture<Self::Response, Self::Error>;
        fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
        fn call(&mut self, req: http::Request<B>) -> Self::Future {
            let inner = self.inner.clone();
            match req.uri().path() {
                "/flservice.FLClientTrainService/TrainForRound" => {
                    #[allow(non_camel_case_types)]
                    struct TrainForRoundSvc<T: FlClientTrainService>(pub Arc<T>);
                    impl<T: FlClientTrainService>
                        tonic::server::StreamingService<super::ClientModelMessage>
                        for TrainForRoundSvc<T>
                    {
                        type Response = super::ClientModelMessage;
                        type ResponseStream = T::TrainForRoundStream;
                        type Future =
                            BoxFuture<tonic::Response<Self::ResponseStream>, tonic::Status>;
                        fn call(
                            &mut self,
                            request: tonic::Request<tonic::Streaming<super::ClientModelMessage>>,
                        ) -> Self::Future {
                            let inner = self.0.clone();
                            let fut = async move { (*inner).train_for_round(request).await };
                            Box::pin(fut)
                        }
                    }
                    let inner = self.inner.clone();
                    let fut = async move {
                        let interceptor = inner.1;
                        let inner = inner.0;
                        let method = TrainForRoundSvc(inner);
                        let codec = tonic::codec::ProstCodec::default();
                        let mut grpc = if let Some(interceptor) = interceptor {
                            tonic::server::Grpc::with_interceptor(codec, interceptor)
                        } else {
                            tonic::server::Grpc::new(codec)
                        };
                        let res = grpc.streaming(method, req).await;
                        Ok(res)
                    };
                    Box::pin(fut)
                }
                _ => Box::pin(async move {
                    Ok(http::Response::builder()
                        .status(200)
                        .header("grpc-status", "12")
                        .body(tonic::body::BoxBody::empty())
                        .unwrap())
                }),
            }
        }
    }
    impl<T: FlClientTrainService> Clone for FlClientTrainServiceServer<T> {
        fn clone(&self) -> Self {
            let inner = self.inner.clone();
            Self { inner }
        }
    }
    impl<T: FlClientTrainService> Clone for _Inner<T> {
        fn clone(&self) -> Self {
            Self(self.0.clone(), self.1.clone())
        }
    }
    impl<T: std::fmt::Debug> std::fmt::Debug for _Inner<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.0)
        }
    }
    impl<T: FlClientTrainService> tonic::transport::NamedService for FlClientTrainServiceServer<T> {
        const NAME: &'static str = "flservice.FLClientTrainService";
    }
}
