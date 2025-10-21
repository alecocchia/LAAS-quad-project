#!/bin/bash

if [[ ( $@ == "--help") ||  $@ == "-h" ]]
then
        echo "Usage: $0 [IMAGE_NAME]"
        #exit 0
else
        if [[ ($# -eq 0 ) ]]
        then
                echo "Usage: $0 [IMAGE_NAME]"
                #exit 0
        else
                # Aggiungi --network host qui:
                docker build --pull -t $1 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --network host .
        fi
fi
