
#num_clients=(30 60 90)
num_clients=(40 50 120 150)
# without boosting
for c in "${num_clients[@]}"; do

  experiment_name=e45_fmnist_google_bland_att_clients_${c}
  aa=$((3000 / $c))
  attack_after=${aa%.*}
  cmd="./cil-train.sh ${experiment_name} -c ./train_configs/google_tasks_emnist.yml --attack_after ${attack_after} --clip 0 --scale_attack true --scale_attack_weight ${c} --num_selected_clients ${c} --num_rounds 160"
  echo "$cmd"
  set -x
  ${cmd}
  set +x
done
