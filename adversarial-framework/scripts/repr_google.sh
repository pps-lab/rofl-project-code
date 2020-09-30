
# Compare if our PGD algorithm works better than basic training and clipping at the end.

./cil-train.sh base_repr_google_emnist_nomal -c ./train_configs/google_tasks_emnist.yml --num_rounds 100000 \
 --save_model_at 600 700 800 900 --num_malicious_clients 0
 ./cil-train.sh base_repr_google_emnist_estnb10 -c ./train_configs/google_tasks_emnist.yml --num_rounds 100000 \
 --save_model_at 600 700 800 900 --clip 0.003

