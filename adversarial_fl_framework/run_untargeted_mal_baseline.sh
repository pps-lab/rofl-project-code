#!/usr/bin/env bash

save_files () {
    experiment_dir=$1
    experiment_name=$2

    echo "zipping updates"
    tar -czf updates.tar.gz ${experiment_dir}/${experiment_name}/updates

    echo "putting log files into S3 bucket"
    local_directory=${experiment_dir}/${experiment_name}
    s3_dir=s3://${bucket_name}/${local_directory}
    aws s3 cp ${local_directory}/log.csv ${s3_dir}/log.csv
    aws s3 cp ${local_directory}/config.json ${s3_dir}/config.json
    aws s3 cp updates.tar.gz ${s3_dir}/updates.tar.gz
    echo "log files saved"

    rm -R ${local_directory}
}

echo "activating environment"
source ./venv/bin/activate
echo "environment activated"

bucket_name=attacks-on-federated-learning
experiment_dir=experiments

num_malicious_clients=1
dataset_list=(mnist fmnist)
model_name=mnist_cnn
clients_list=(10 25 50 100)
scale_attack_list=(false true)
attack_type=untargeted

# without boosting
for dataset in "${dataset_list[@]}"; do
    for clients in "${clients_list[@]}"; do
        for scale_attack in "${scale_attack_list[@]}"; do
            experiment_name=${attack_type}_mal_baseline_boosting_${scale_attack}_num_malicious_clients_${num_malicious_clients}_${dataset}_clients_${clients}
            cmd="python -m src.main --attack_type ${attack_type} --num_malicious_clients ${num_malicious_clients} --scale_attack ${scale_attack} --scale_attack_weight ${clients} --dataset ${dataset} --num_clients ${clients} --model_name ${model_name} --experiment_name ${experiment_name}"
            echo "$cmd"
            set -x
            ${cmd}
            set +x
            save_files ${experiment_dir} ${experiment_name}
        done
    done
done


echo "shutting down"
sudo shutdown -h now