
# Compare if our PGD algorithm works better than basic training and clipping at the end.

./cil-train.sh emnist_nomal_iid -c ./train_configs/google_tasks_emnist.yml \
  --num_malicious_clients 0 --clip 0 --data_distribution IID
./cil-train.sh emnist_nomal_noniid -c ./train_configs/google_tasks_emnist.yml \
  --num_malicious_clients 0 --clip 0

./cil-train.sh emnist_mal_iid -c ./train_configs/google_tasks_emnist.yml \
  --data_distribution IID
./cil-train.sh emnist_mal_noniid -c ./train_configs/google_tasks_emnist.yml