#!/bin/bash
# Shell script to facilitate training our CIL model on the leonhard cluster

leonhard_train() {
    cd attacks_on_federated_learning
    git pull
    module load gcc/6.3.0 python_cpu/3.7.4 hdf5/1.10.1
    job=$(bsub -W 15 "python -m plotting.$@")
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
echo "[$curTime] submitted plotting job at commit $commit with params $@" >> joblog.txt
