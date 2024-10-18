#!/bin/bash
#SBATCH --job-name=sex_outrider
#SBATCH --output=logs/sex_outrider_%A.out
#SBATCH --error=logs/sex_outrider_%A.err
#SBATCH --mem-per-cpu=7G
#SBATCH --exclude=ouga[01-04]
#SBATCH --cpus-per-task 128

cd '/s/project/promoter_prediction/sex_analysis/bcv'
py_outrider --input gtexv8_rnaseq_counts.csv --profile outrider --output py_outrider_results.h5ad --output_res_table py_outrider_results.csv  --num_cpus 128 --sample_anno sample_annotation.csv --covariates subtissue sex --noise_factor "0" --encod_dim 45
