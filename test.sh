#!/bin/bash
#SBATCH -J train_test
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -p kshdnormal02
python test.py