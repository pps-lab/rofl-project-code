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

num_malicious_clients=2
clients=25

num_selected_clients_list=(10 12 14 16 18 20)
dataset=mnist
model_name=mnist_cnn
scale_attack=true
attack_type=untargeted
untargeted_after_training=true

# without boosting
for num_selected_clients in "${num_selected_clients_list[@]}"; do
    experiment_name=num_selected_clients_${num_selected_clients}_${attack_type}_${untargeted_after_training}_mal_baseline_boosting_${scale_attack}_num_malicious_clients_${num_malicious_clients}_${dataset}_clients_${clients}
    scale_attack_weight=$((num_selected_clients / num_malicious_clients))
    cmd="python -m src.main --num_selected_clients ${num_selected_clients} --attack_type ${attack_type} --untargeted_after_training ${untargeted_after_training} --num_malicious_clients ${num_malicious_clients} --scale_attack ${scale_attack} --scale_attack_weight ${scale_attack_weight} --dataset ${dataset} --num_clients ${clients} --model_name ${model_name} --experiment_name ${experiment_name}"
    echo "$cmd"
    set -x
    ${cmd}
    set +x
    save_files ${experiment_dir} ${experiment_name}
done



echo "shutting down"
sudo shutdown -h now