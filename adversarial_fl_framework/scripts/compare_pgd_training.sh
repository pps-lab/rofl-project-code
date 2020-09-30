
# Compare if our PGD algorithm works better than basic training and clipping at the end.

./cil-train.sh fmnist_compare_pgd_linf_base -c ./train_configs/bhagoji_fmnist_multiple.yml --num_malicious_clients 0
./cil-train.sh fmnist_compare_pgd_linf_cmp -c ./train_configs/bhagoji_fmnist_multiple.yml --num_malicious_clients 0 --pgd l_inf --pgd_constraint 0.0015

./cil-train.sh fmnist_compare_pgd_l2_base -c ./train_configs/bhagoji_fmnist_multiple.yml --num_malicious_clients 0 --clip 0 --clip_l2 2.67
./cil-train.sh fmnist_compare_pgd_l2_cmp -c ./train_configs/bhagoji_fmnist_multiple.yml --num_malicious_clients 0 --clip 0 --clip_l2 2.67 --pgd l2 --pgd_constraint 2.67
