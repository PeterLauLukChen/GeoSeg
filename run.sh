#!/bin/bash
#SBATCH --account=iicd
#SBATCH --job-name=RunImplementation
#SBATCH --output=ts_%j.log
#SBATCH --error=ts_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G                        
#SBATCH --gres=gpu:2                     
#SBATCH --time=12:00:00                 

module load python37
source /burg/home/lc3826/GPUrun/myenv/bin/activate
python /burg/home/lc3826/GPUrun/TS/tiltedseg.py --PENALTY 1.3 --MODE minmax