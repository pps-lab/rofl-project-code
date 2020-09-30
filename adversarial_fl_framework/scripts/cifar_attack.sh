
#./cil-train.sh cifar_attack_regression -c ./train_configs/backdoor_bagdas_cifar_background_wall.yml \
#  --load_model ./experiments/base_cifar_resnet32/models/model_800.h5 --attack_type data_poison --attack_after 5 --model_name resnet32 \
#  --attack_frequency 1 \
#  --scale_attack true \
#  --scale_attack_weight 100 \
#  --attack_stop_after 5
#./cil-train.sh cifar_attack_regression_racingstripes -c ./train_configs/backdoor_bagdas_cifar_racing_stripes.yml \
#  --load_model ./experiments/base_cifar_resnet32/models/model_800.h5 --attack_type data_poison --attack_after 5 --model_name resnet32 \
#  --attack_frequency 1 \
#  --scale_attack true \
#  --scale_attack_weight 100 \
#  --attack_stop_after 5
#
#./cil-train.sh cifar_dp_scale_clip_2 -c ./train_configs/backdoor_bagdas_cifar_background_wall.yml \
#  --load_model ./experiments/base_cifar_resnet32/models/model_800.h5 --attack_type data_poison --attack_after 5 --model_name resnet32 \
#  --attack_frequency 1 \
#  --scale_attack true \
#  --scale_attack_weight 10 \
#  --clip 0.02
#./cil-train.sh cifar_ml_scale_clip_100 -c ./train_configs/backdoor_bagdas_cifar_background_wall.yml \
#  --load_model ./experiments/base_cifar_resnet32/models/model_800.h5 --attack_after 5 --model_name resnet32 \
#  --attack_frequency 1 \
#  --scale_attack true \
#  --scale_attack_weight 100 \
#  --clip 0.02

mal_num_batch=(400 450 500)
mal_lr=(0.0008)

# without boosting
for v in "${mal_num_batch[@]}"; do
  for lr in "${mal_lr[@]}"; do

    experiment_name=e44_cifar_attack_memorize_${v}_${lr}
    cmd="./cil-train.sh ${experiment_name} -c ./train_configs/backdoor_bagdas_cifar_racing_stripes.yml --load_model ./models/resnet56v2_single.h5 --attack_after 1 --model_name resnet56_v2 --attack_frequency 1 --attack_stop_after 1 --scale_attack true --scale_attack_weight 100 --num_selected_clients 1 --num_rounds 3 --attacker_full_dataset false --mal_num_epochs 1 --mal_num_batch ${v} --mal_learning_rate ${lr}"
    echo "$cmd"
    set -x
    ${cmd}
    set +x
  done
done
