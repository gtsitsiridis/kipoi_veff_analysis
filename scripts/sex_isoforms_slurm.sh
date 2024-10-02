#!/bin/bash
#SBATCH --job-name=sex_isoforms
#SBATCH --output=logs/sex_isoforms_%A_%a.out
#SBATCH --error=logs/sex_isoforms_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G

# Define variables for file paths
GENES_PATH="/data/nasif12/home_if12/tsi/projects/kipoi_veff_analysis/assets/genes.txt"
GTEX_ANNOTATION_PATH="/s/project/gtex_genetic_diagnosis/v8/sample_annotation.tsv"
GTF_PATH="/s/project/rep/processed/training_results_v15/gtex_v8_old_dna/gtf_transcripts.parquet"
GTEX_TRANSCRIPT_TPM_PATH="/s/project/rep/processed/training_results_v15/general/gtex_transcript_tpms.zarr/"
OUTPUT_PATH="/s/project/promoter_prediction/sex_analysis/isoform_proportions"

# Create log directory if it doesn't exist
mkdir -p logs



echo "Slurm Job Id SLURM_ARRAY_JOB_ID is ${SLURM_ARRAY_JOB_ID}"
echo "Slurm job array index SLURM_ARRAY_TASK_ID value is ${SLURM_ARRAY_TASK_ID}"

# Run the Python script with the gene index
python scripts/sex_isoforms.py --gtex_annotation_path ${GTEX_ANNOTATION_PATH} \
  --gtf_path ${GTF_PATH} --gtex_transcript_tpm_path ${GTEX_TRANSCRIPT_TPM_PATH} \
  --genes_path ${GENES_PATH} --output_path ${OUTPUT_PATH} --gene_index $SLURM_ARRAY_TASK_ID
