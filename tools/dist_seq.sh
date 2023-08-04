#!/usr/bin/env bash

CONFIGDIR=$1
GPUS=$2
WORKDIR=$3
PY_ARGS=${@:4}
EPOCH=${EPOCH:-1}
PORT=${PORT:-29500}

CUR_EP=1
if [ ${EPOCH} -le ${CUR_EP} ];
then
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py "${CONFIGDIR}ep1.py" --launcher pytorch --work_dir=${WORKDIR} ${PY_ARGS}
    sleep 1
fi

for ((i=1; i<10; i++))
do
    if [ ${EPOCH} -le $((${i}+1)) ];
    then
        if [ ! -f "${WORKDIR}epoch_${i}.pth" ];
        then
            echo "FILE NOT EXISTS: ${WORKDIR}epoch_${i}.pth"
            exit -1
        fi
        python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
            $(dirname "$0")/train.py "${CONFIGDIR}ep$((${i}+1)).py" --launcher pytorch \
            --work_dir=${WORKDIR} \
            --resume-from "${WORKDIR}epoch_${i}.pth" \
            ${PY_ARGS}
        sleep 1
    fi
done