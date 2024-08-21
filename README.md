# Variant effect prediction using Kipoi

## Setup:

### Create main conda environment
```bash
conda env create -f workflow/envs/kipoi-veff-analysis
conda activate kipoi-veff-analysis
```

### Snakemake workflow
#### Local profile

To run the snakemake workflow locally using the dev config, execute the following command:

```bash
conda activate kipoi-veff-analysis
snakemake --workflow-profile=workflow/profiles/dev
```

#### Slurm profile

To run the snakemake workflow on a slurm cluster using the production config, execute the following command:

```bash
conda activate kipoi-veff-analysis
CONDA_OVERRIDE_CUDA="11.8" snakemake --workflow-profile=workflow/profiles/prod
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