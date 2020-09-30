#!/bin/bash

FILE=$1
  
if [ ! -f $FILE ]; then
   echo "File $FILE not found."
   exit 1
fi

CONFIG_FILE="$(readlink -f $FILE)"
source <(grep = $CONFIG_FILE)

cd "$(dirname "$0")"

echo "Loading with $NUM_CLIENTS clients"

fab transfer_config:local_config=$CONFIG_FILE
fab run_data_splitter
for i in $( seq 1 $NUM_CLIENTS )
do
    # add sleep 2 to give server time to start up
    gnome-terminal --tab -- bash -c "sleep 2; fab run_client:id=$i; bash"
done;

fab run_server