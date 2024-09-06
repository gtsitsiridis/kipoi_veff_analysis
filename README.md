# Variant effect prediction using Kipoi

## Setup:

### Profiles

#### Local profile

To run the test workflow locally use the following profile:

```bash
workflow/profiles/dev
```

#### Slurm profile

To run the production workflow on a cluster with a SLURM scheduler use the following profile:

```bash
workflow/profiles/prod
```

### Create conda environments

```bash
conda env create -f workflow/envs/kipoi-veff-analysis.yml
conda activate kipoi-veff-analysis
# create conda envs on the cluster
snakemake --workflow-profile=workflow/profiles/<profile> --conda-create-envs-only
```

### Run Snakemake workflow

```bash
conda activate kipoi-veff-analysis
# Note: when running the prod profile, there is no need to execute the command in a server with GPU support,
# if the conda envs have already been created on a server with GPU support.
snakemake --workflow-profile=workflow/profiles/<profile>
# to keep temporary files
#snakemake --workflow-profile=workflow/profiles/<profile> --notemp

# to run in high priority:
#snakemake --workflow-profile=workflow/profiles/<profile> --default-resources slurm_partition=urgent slurm_extra="'--exclude=ouga[01-04] --no-requeue'"

# example:
# high priority and keep temp files
#snakemake --workflow-profile=workflow/profiles/prod --notemp --default-resources slurm_partition=urgent slurm_extra="'--exclude=ouga[01-04] --no-requeue'"

# To avoid re-running the workflow, for existing files, "touch" the existing files using the --touch directive.
# example:
#snakemake --workflow-profile=workflow/profiles/prod --touch /s/project/promoter_prediction/kipoi_expression_prediction/process/common/genomes/GRCh37.parquet
#snakemake --workflow-profile=workflow/profiles/prod --touch /s/project/promoter_prediction/kipoi_expression_prediction/process/enformer/alternative/raw/GRCh37_short__gtexv8_2000-500.parquet/part-0*
#snakemake --workflow-profile=workflow/profiles/prod --touch /s/project/promoter_prediction/kipoi_expression_prediction/process/enformer/reference/raw/GRCh37_short.parquet/*/data.parquet

# To speed up the construction of the DAG when running snakemake, the --batch option on the benchmark rule can be used.
# example:
# first batch
#snakemake --workflow-profile=workflow/profiles/<profile> --batch benchmark=1/10
# nth batch
#snakemake --workflow-profile=workflow/profiles/<profile> --batch benchmark=n/10
```

## Predictors

### Enformer

TODO add package dev info

Karollus et al. (cite Karollus) have shown that Enformer is able to capture gene expression determinants in promoters.
Enformer is a transformer-based model that takes as input one-hot encoded DNA sequence of 393,216 bp and predicts 5,313
Human and 1,643 Mouse genomic tracks for the central sequence of 114,688 bp, aggregated into 128 bp bins. To calculate
a variant effect score for a given variant on neighboring transcripts, we used the following approach:

For all protein-coding transcripts (todo canonical or all), we define as regions of interest, promoters that are XXX bp
upstream and YYY bp downstream of the transcription start site (TSS). For all variants in these regions, we predict the
gene expression score both for the reference and alternative sequences using Enformer. To account for the case that
regulatory elements fall on bin boundaries, we predict the gene expression score for the central sequence 3 times,
centered at the TSS and with small shifts upstream and downstream of the TSS. We then extract the 3 central bins from
each prediction and average them to obtain a single prediction for each track. To map the predicted tracks into a single
gene expression score in a GTEx-tissue specific manner, we ... (todo tissue mapper) (cite Karollus). (todo if not
canonical, include isoform proportions). Finally, we calculate the log2 fold change in gene expression score between the
reference and alternative sequences.