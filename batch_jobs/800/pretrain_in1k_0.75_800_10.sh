#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuh100
### -- set the job Name -- 
#BSUB -J 800_run10
### -- ask for number of cores (default: 1) -- 
#BSUB -n 8
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 3GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- Select the resources: 2 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o bjob_logs/output_%J.out 
#BSUB -e bjob_logs/output_%J.err 

source /work3/s184202/thesis/thesis-mae-in1k/env/bin/activate

python -m src.pretrain --config-name=pretrain_in1k_0.75_800 resume_from_checkpoint=/work3/s184202/thesis/thesis-mae-in1k/pretrain_outputs/checkpoints/pretrain_in1k_0.75_800/version_9/last.ckpt