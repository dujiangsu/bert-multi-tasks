#!/bin/bash
#SBATCH --exclusive
#SBATCH --time=99:00:00


CUDA_VISIBLE_DEVICES=0,1,2,3


python bert-glue.py