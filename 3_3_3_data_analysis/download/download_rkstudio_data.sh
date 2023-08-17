#!/bin/zsh

source ~/opt/miniconda3/etc/profile.d/conda.sh
conda activate web_scrape

python /Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_3_data_analysis/python_scripts/download_rkstudio_data.py

python /Users/djw/Documents/pCloud_synced/Academics/Projects/2020_thesis/thesis_experiments/3_experiments/3_3_experience_sampling/3_3_3_data_analysis/python_scripts/track_adherence.py

conda deactivate
