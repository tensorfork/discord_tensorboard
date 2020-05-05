#!/bin/bash

set -x

start=$1
shift 1
logdir=$1
shift 1
channel=$1
shift 1
waitsec=${1:-600}
shift 1

[ -d /tfk ] || exit 1

logfile=/tfk/discord_tensorboard_start_`echo $logdir | sha1sum | replace - '' | trim`.txt
touch $logfile

start2=`cat $logfile`
[ -z $start2 ] && start2=$start
[ $start -gt $start2 ] && echo $start > $logfile || start=$start2
set -x
prev=$start

trap "exit" INT TERM
trap "kill -9 0" EXIT
trap "kill -9 0" SIGINT

while true; do
        [ -z $start ] && exit 1
        [ $start -lt $prev ] && exit 1
        timeout --signal=SIGKILL 7h bash ~/discord_tensorboard/run_viewer.sh --logdir $logdir --channel $channel --waitsec 0 --logstart $logfile --start $start "$@"
        prev=$start
        start=`cat $logfile`
        echo $logfile
        echo prev=${prev}
        echo "--start $start --logdir $logdir --channel $channel"
        sleep $waitsec
done
