#!/bin/bash

FILE=$1
  
if [ ! -f $FILE ]; then
   echo "File $FILE not found."
   exit 1
fi

CONFIG_FILE="$(readlink -f $FILE)"

echo "Config file: $CONFIG_FILE"
source <(grep = $CONFIG_FILE)
shift

cd "$(dirname "$0")"
HELPER_SCRIPT="remote_test_helper.sh"

gnome-terminal --window --maximize -- bash -c "bash $HELPER_SCRIPT $CONFIG_FILE; bash"
