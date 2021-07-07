fn main() {
    tonic_build::compile_protos("proto/roflservice/flservice.proto")
        .unwrap_or_else(|e| panic!("Failed to compile protos {:?}", e));
}
