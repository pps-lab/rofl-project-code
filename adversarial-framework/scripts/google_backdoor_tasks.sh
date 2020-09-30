# Script to evaluate google backdoor sizes
# Run locally, will submit to leonhard


#cd ../

num_backdoor_tasks=(10 30 50 100)
model_name=bhagoji

plot_names=""

# without boosting
for backdoor_tasks in "${num_backdoor_tasks[@]}"; do
    experiment_name=google_backdoor_${model_name}_tasks_${backdoor_tasks}
    plot_names+="${experiment_name},"
    cmd="./cil-train.sh -c ./train_configs/google_tasks_emnist.yml --experiment_name ${experiment_name} --backdoor_stealth true --backdoor_tasks ${backdoor_tasks} --model_name ${model_name}"
    echo "$cmd"
    set -x
    ${cmd}
    set +x
done

plot_cmd="./cil-plot.sh adv_benign_compare_all --plot_single true --experiment_name ${plot_names}"
echo "When done, run"
echo "${plot_cmd}"