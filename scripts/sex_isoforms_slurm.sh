#!/bin/bash

# Define variables for file paths
GENES_PATH="/data/nasif12/home_if12/tsi/projects/kipoi_veff_analysis/assets/genes.txt"
GTEX_ANNOTATION_PATH="/s/project/gtex_genetic_diagnosis/v8/sample_annotation.tsv"
GTF_PATH="/s/project/rep/processed/training_results_v15/gtex_v8_old_dna/gtf_transcripts.parquet"
GTEX_TRANSCRIPT_TPM_PATH="/s/project/rep/processed/training_results_v15/general/gtex_transcript_tpms.zarr/"
OUTPUT_PATH="/s/project/promoter_prediction/sex_analysis/isoform_proportions"
LOG_DIR="logs"

# Read the number of genes
NUM_GENES=$(wc -l < "$GENES_PATH")

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Submit a job array
sbatch --array=0-$(($NUM_GENES-1)) --ntasks=200 --partition=lowprio<< 'EOF'
#!/bin/bash
#SBATCH --job-name=sex_isoforms
#SBATCH --output=${LOG_DIR}/sex_isoforms_%A_%a.out
#SBATCH --error=${LOG_DIR}/sex_isoforms_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G

# Run the Python script with the gene index
python scripts/sex_isoforms.py --gtex_annotation_path ${GTEX_ANNOTATION_PATH} --gtf_path ${GTF_PATH} --gtex_transcript_tpm_path ${GTEX_TRANSCRIPT_TPM_PATH} --genes_path ${GENES_PATH} --output_path ${OUTPUT_PATH} --gene_index $SLURM_ARRAY_TASK_ID
EOF