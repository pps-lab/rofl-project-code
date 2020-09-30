
# Compare if our PGD algorithm works better than basic training and clipping at the end.

./cil-train.sh fmnist_clip_lowerfact3_clients10 -c ./train_configs/bhagoji_fmnist_multiple.yml --num_rounds 1000 --clip 0.0005 --num_clients 10
./cil-train.sh fmnist_clip_lowerfact3_clients20 -c ./train_configs/bhagoji_fmnist_multiple.yml --num_rounds 1000 --clip 0.0005 --num_clients 20
#./cil-train.sh fmnist_clip_lowerfact3_clients40 -c ./train_configs/bhagoji_fmnist_multiple.yml --num_rounds 1000 --clip 0.0005 --num_clients 40
./cil-train.sh fmnist_clip_lowerfact3_clients80 -c ./train_configs/bhagoji_fmnist_multiple.yml --num_rounds 1000 --clip 0.0005 --num_clients 80
#./cil-train.sh fmnist_clip_lowerfact3_clients160 -c ./train_configs/bhagoji_fmnist_multiple.yml --num_rounds 1000 --clip 0.0005 --num_clients 160
