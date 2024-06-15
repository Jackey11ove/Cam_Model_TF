#!/bin/bash
if [ ! -d "ret_one/" ]; then
    mkdir -p "ret_one/"
fi 

sbatch --job-name='ssd' -o "ret_one/%x/%j.out" -e "ret_one/%x/%j.err" slurms/ptq_one.slurm
