#!/usr/bin/env bash

echo "activating environment"
source ./venv/bin/activate
echo "environment activated"

bucket_name=attacks-on-federated-learning
experiment_dir=experiments

dataset=mnist
model_name=mnist_cnn
clients_list=(10 25 50 100)
samples_list=(30000 40000 50000 60000)

for clients in "${clients_list[@]}"; do
    for samples in "${samples_list[@]}"; do
        experiment_name=baseline_${dataset}_clients_${clients}_samples_${samples}
        cmd="python -m src.main --dataset ${dataset} --num_clients ${clients} --model_name ${model_name} --number_of_samples ${samples} --experiment_name ${experiment_name}"

        set -x
        ${cmd}
        set +x

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
    done
done

echo "shutting down"
sudo shutdown -h now