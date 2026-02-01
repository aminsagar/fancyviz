#!/bin/bash

TPU_ID=$1
JOB_ID=$2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="bindcraft_logs/bindcraft_job${JOB_ID}_tpu${TPU_ID}_${TIMESTAMP}.log"

export TPU_VISIBLE_DEVICES=$TPU_ID
export TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1"
export TPU_PROCESS_BOUNDS="1,1,1"
export JAX_PLATFORMS="tpu"

python -u /home/amin_sagar/softwares/BindCraft/bindcraft.py \
    --settings ./settings_target/CDCP1_CTD_DefHel_allowCys.json \
    --filters settings_filters/peptide_filters.json \
    --advanced settings_advanced/peptide_3stage_multimer_allowcys.json \
    > $LOGFILE 2>&1
