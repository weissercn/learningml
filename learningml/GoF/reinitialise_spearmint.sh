#! /bin/bash

PROCESS="$(pgrep mongo)"
if [ -z "$PROCESS" ];then
	echo "No mongo process is running"
else
	echo "Mongo process $PROCESS is running, killing this process..."
	kill $PROCESS
fi
