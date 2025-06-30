#!/bin/bash
#SBATCH --job-name=adrenal      # Specify job name
#SBATCH --partition=gpu        # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --mem=48G                # Use entire memory of node
#SBATCH --gres=gpu:1           # Generic resources; 1 GPU   
#SBATCH --exclusive            # Do not share node
#SBATCH --time=24:00:00        # Set a limit on the total run time  
#SBATCH --mail-type=FAIL       # Notify user by email in case of job failur
#SBATCH --account=sc-users     # Charge resources on this project accoun
#SBATCH --output=/home/jabareen/logs/adrenal.o%j    # File name for standard output
#SBATCH --error=/home/jabareen/logs/adrenal.e%j     # File name for standard error output

srun python /sc-projects/sc-proj-gbm-radiomics/posenc/run_mnist.py
