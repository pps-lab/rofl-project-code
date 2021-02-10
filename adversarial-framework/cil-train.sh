#!/bin/bash
# Shell script to facilitate training our CIL model on the leonhard cluster

leonhard_train() {
    cd attacks_on_federated_learning
    git pull
    module load gcc/6.3.0 python_gpu/3.7.4 hdf5/1.10.1
    pip install --user -r requirements.txt
    experiment_name="$1"
    outfile="lsf.%J.$experiment_name.out"
    job=$(bsub -W 1440 -n 20 -J $experiment_name -oo $outfile -R "rusage[mem=4096,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -N "python -m src.main --experiment_name $experiment_name ${@:2}")
    echo "submitted job $job with params: $@"
}

username="`cat .ethusername`"
if [ -z "$username" ]
then
    echo "Set username in .ethusername file in project root to avoid having to type this every time."
    echo -n "NETHZ username: "
    read username
fi;


ssh $username@login.leonhard.ethz.ch << EOF
    $(typeset -f)
    leonhard_train $@
EOF

curTime=$(date)
commit=$(git rev-parse HEAD)
echo "[$curTime] submitted job at commit $commit with params $@" >> joblog.txt
