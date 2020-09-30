
# Compare if our PGD algorithm works better than basic training and clipping at the end.

./cil-train.sh fmnist_compare_pgd_l2_single_base -c ./train_configs/bhagoji_fmnist_multiple.yml --num_malicious_clients 0 --clip 0 --clip_l2 2.67 --num_clients 1 --num_selected_clients 1
./cil-train.sh fmnist_compare_pgd_l2_single_cmp -c ./train_configs/bhagoji_fmnist_multiple.yml --num_malicious_clients 0 --clip 0 --clip_l2 2.67 --pgd l2 --pgd_constraint 2.67  --num_clients 1 --num_selected_clients 1
