#!/bin/bash
#SBATCH --mail-user=nima.akbarzadeh@mail.mcgill.ca
#SBATCH --account=rrg-adulyasa
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=~/projects/def-adulyasa/mcnima/bandits/SafeWhittleIndex/output.txt
#SBATCH --time=00:10:00

python main.py
