#!/bin/bash

source .venv/bin/activate

python pool_simulate/main.py -d debug/subset -k 8 2> /dev/null