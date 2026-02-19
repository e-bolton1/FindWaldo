#!/bin/bash
# Batch script to run object detection and localization on Aire supercomputer
#SBATCH --job-name=waldo_detection
#SBATCH --output=waldo_detection.out
#SBATCH --error=waldo_detection.err
#SBATCH --time=04:00:00
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load modules (adjust as needed for Aire)
module load python/3.9
module load tensorflow/2.8

# Activate virtual environment if needed
# source ~/myenv/bin/activate

# Run the script
python object_detection_localization.py
