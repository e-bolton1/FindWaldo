

#!/bin/bash

#SBATCH --job-name=waldo_detection
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=waldo_detection_%j_%a.out
#SBATCH --error=waldo_detection_%j_%a.err


# Load modules (adjust as needed for Aire)
module load python/3.9
module load tensorflow/2.8

# Activate virtual environment if needed
# source ~/myenv/bin/activate

# Run the script
python object_detection_localization.py
