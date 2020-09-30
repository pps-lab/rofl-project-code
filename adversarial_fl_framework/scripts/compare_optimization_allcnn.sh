
# Compare if our PGD algorithm works better than basic training and clipping at the end.

./cil-train.sh exp_cifar_allcnn -c ./train_configs/cifar_allcnn.yml

./cil-train.sh exp_cifar_allcnn_randommask_25 -c ./train_configs/cifar_allcnn.yml --federated_dropout_rate 0.25
./cil-train.sh exp_cifar_allcnn_subspace_25 -c ./train_configs/cifar_allcnn.yml --model_name allcnn_intrinsic --intrinsic_dimension 325000
