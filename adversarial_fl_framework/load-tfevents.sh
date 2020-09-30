
username="`cat .ethusername`"
if [ -z "$username" ]
then
    echo "Set username in .ethusername file in project root to avoid having to type this every time."
    echo -n "NETHZ username: "
    read username
fi;

rsync -avz --include="/**/" --include="/**/events/*" --exclude='**/updates/' $username@login.leonhard.ethz.ch:attacks_on_federated_learning/experiments/ experiments/


#scp -r $username@login.leonhard.ethz.ch:attacks_on_federated_learning/experiments/*/events experiments/*/events