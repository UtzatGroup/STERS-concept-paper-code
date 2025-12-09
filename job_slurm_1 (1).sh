#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=fc_utzatgroup
#SBATCH --partition=savio3_htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:05:30
#SBATCH --array=1-10
#SBATCH --output=/global/home/users/ccobbbruno/Python_files/Job_array_output_files/array_job_%A_task_%a.out
#SBATCH --error=/global/home/users/ccobbbruno/Python_files/Job_array_output_files/array_job_%A_task_%a.err

module load python
sleep $((RANDOM%30+1))

python simulation_for_mean_FWHM_arrays_two_values.py $SLURM_ARRAY_TASK_ID
