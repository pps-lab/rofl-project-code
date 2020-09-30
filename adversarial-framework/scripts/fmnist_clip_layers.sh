
# Compare if our PGD algorithm works better than basic training and clipping at the end.

./cil-train.sh fmnist_clip_lastlayers_20 -c ./train_configs/bhagoji_fmnist_multiple.yml --num_rounds 1000 --clip 0.0005 --num_clients 20 \
  --clip_layers 4 5 6 7
./cil-train.sh fmnist_clip_firstlayers_20 -c ./train_configs/bhagoji_fmnist_multiple.yml --num_rounds 1000 --clip 0.0005 --num_clients 20 \
  --clip_layers 0 1 2 3

./cil-train.sh fmnist_clip_lastlayers_20_higher -c ./train_configs/bhagoji_fmnist_multiple.yml --num_rounds 1000 --clip 0.0015 --num_clients 20 \
  --clip_layers 4 5 6 7