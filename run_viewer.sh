#!/bin/sh
source "${HOME}/discord_tensorboard/.env"
set -ex
exec python3 ~/discord_tensorboard/discord_tensorboard.py "$@"
