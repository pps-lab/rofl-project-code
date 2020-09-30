
# Compare if our PGD algorithm works better than basic training and clipping at the end.

./cil-train.sh base_cifar_resnet32_cont -c ./train_configs/backdoor_bagdas_cifar_background_wall.yml \
  --num_malicious_clients 0 --model_name resnet32 --num_rounds 10000 --save_model_at 800 --load_model ./experiments/base_cifar_resnet32/models/model_800.h5


./cil-train.sh base_cifar_resnet32_cont -c ./train_configs/backdoor_bagdas_cifar_background_wall.yml \
  --num_malicious_clients 0 --model_name resnet32 --num_rounds 10000 --save_model_at 800 --load_model ./experiments/base_cifar_resnet32/models/model_800.h5


./cil-train.sh base_google_emnist_federated_dropout -c ./train_configs/google_tasks_emnist.yml --num_rounds 100000 --save_model_at 600 700 800 900 --federated_dropout_rate 0.75


./cil-train.sh base_google_emnist_many_gpu_cont -c ./train_configs/google_tasks_emnist.yml --num_rounds 100000 --save_model_at 600 700 800 900 --load_model ./experiments/base_google_emnist_many_gpu/models/model_500.h5
./cil-train.sh base_google_cifar_many_gpu_cont -c ./train_configs/cifar_fl.yml --num_rounds 100000 --save_model_at 600 700 800 900 --load_model ./experiments/base_google_cifar_many_gpu_lowerlr/models/model_500.h5  --global_learning_rate 10