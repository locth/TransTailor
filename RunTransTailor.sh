#!/bin/sh

TransTailor_Path=$1

python $TransTailor_Path/TransTailorV2_db.py --root $1 --numworker 8 --batchsize 32 --checkpoint "checkpoint/checkpoint_0.pkl"