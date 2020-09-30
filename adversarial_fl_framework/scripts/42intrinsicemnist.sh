
vsize=(100 300 500 750 1000 1250 1500 2000)

# without boosting
for v in "${vsize[@]}"; do
    experiment_name=e42_emnist_mnistcnn_intrinsic_${v}
    cmd="./cil-train.sh ${experiment_name} -c ./train_configs/subspace_emnist_single.yml --model_name mnistcnn_intrinsic --intrinsic_dimension ${v}"
    echo "$cmd"
    set -x
    ${cmd}
    set +x
done