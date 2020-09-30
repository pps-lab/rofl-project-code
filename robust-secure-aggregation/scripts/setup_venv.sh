#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
cd ..

python3 -m venv .

ret=$?
if [ $ret -ne 0 ]; then
    echo "Error activating venv!"
    return 1  
fi

source ./bin/activate
pip3 install -r requirements.txt

rustenv renv
ret=$?
if [ $ret -ne 0 ]; then
    echo "Error activating rustenv!"
    return 1  
fi
. renv/bin/activate

# NOTE mlei: weird linking bug requires reinstallationof python-socketio
# https://github.com/miguelgrinberg/Flask-SocketIO/issues/164#issuecomment-pu
#pip3 uninstall -y python-socketio && pip3 install python-socketio
cd $DIR
