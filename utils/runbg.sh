#!/bin/bash

usage() {
	echo "Usage: $0 [-l] <python_script.py>"
	echo "  -l    Enable logging (output saved to a .log file)"
	exit 1
}

LOGGING=false
while getopts "l" opt; do
	case "$opt" in
		l) LOGGING=true ;;
		*) usage ;;
	esac
done
shift $((OPTIND-1))

if [ -z "$1" ]; then
	usage
fi

SCRIPT_NAME="$1"

if [ "$LOGGING" = true ]; then
	LOG_FILE="${SCRIPT_NAME%.*}.log"
	nohup python -u "$SCRIPT_NAME" > "LOG_FILE" 2>&1 &
else
	nohup python -u "$SCRIPT_NAME" >/dev/null 2>&1 &
fi

PID=$!

disown "$PID"

echo "Started $SCRIPT_NAME in the background (PID: $PID)"
if [ "$LOGGING" = true ]; then
	echo "Logs: $LOG_FILE"
fi
