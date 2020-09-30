
#inf_clipping_bounds=(0.00015 0.0015 0.015 0.15)
#l2_clipping_bounds=(1 3 5 10)
#inf_clipping_bounds=(0.025 0.020 0.010 0.005)
#l2_clipping_bounds=(8 12 14 16 18)
l2_clipping_bounds=(20 25 30 35 40)

# without boosting
for clip in "${inf_clipping_bounds[@]}"; do
    experiment_name=e41_clipinf_${clip}
    cmd="./cil-train.sh ${experiment_name} -c ./train_configs/google_tasks_emnist.yml --clip ${clip}"
    echo "$cmd"
    set -x
    ${cmd}
    set +x
done
# without boosting
for clip in "${l2_clipping_bounds[@]}"; do
    experiment_name=e41_clipl2_${clip}
    cmd="./cil-train.sh ${experiment_name} -c ./train_configs/google_tasks_emnist.yml --clip 0 --clip_l2 ${clip}"
    echo "$cmd"
    set -x
    ${cmd}
    set +x
done