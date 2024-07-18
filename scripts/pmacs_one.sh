#!/bin/bash
if [ ! -d "param_macs/voc2007" ]; then
    mkdir -p "param_macs/voc2007"
fi 

if [ ! -d "ret_one/param_macs" ]; then
    mkdir -p "ret_one/param_macs"
fi 

if [ ! -d "param_macs/voc0712test" ]; then
    mkdir -p "param_macs/voc0712test"
fi 

if [ ! -d "param_macs/$2" ]; then
    mkdir -p "param_macs/$2"
fi 

sbatch --job-name="param_macs" -o "ret_one/%x/%j.out" -e "ret_one/%x/%j.err" --export=Model=$1,Dataset=$2,Datasize=$3,Classnum=$4, slurms/pmacs_one.slurm