import pathlib

# the path to the output folder
output_path = pathlib.Path(config["output_path"]) / 'process' / 'enformer'

module common_workflow:
    snakefile: "common.smk"
    config: config

use rule genome from common_workflow


rule predict_reference:
    priority: 5
    resources:
        gpu=1,
        mem_mb=lambda wildcards, attempt, threads: 12000 + (1000 * attempt)
    output:
        prediction_path=output_path / 'reference/raw/{ref_key}.parquet' / 'chrom={chromosome}/data.parquet',
    input:
        genome_path=expand(rules.genome.output[0],
            genome=lookup(dpath="enformer/references/{ref_key}/genome",within=config))
    params:
        type='reference'
    script:
        '../scripts/enformer/predict_expression.py'


rule aggregate_prediction:
    priority: 4
    resources:
        mem_mb=lambda wildcards, attempt, threads: 6000 + (1000 * attempt)
    output:
        aggregated_path=temp(output_path / '{allele_type}/aggregated/{num_agg_bins}/{key}.parquet/{subpath}'),
    input:
        prediction_path=output_path / '{allele_type}/raw/{key}.parquet/{subpath}',
    script:
        '../scripts/enformer/aggregate_tracks.py'


def train_mapper_input(wildcards):
    mapper_key, ref_key = wildcards['mapper_key'], wildcards['ref_key']
    mapper_config = config['enformer']['mappers'][mapper_key]
    reference_config = config['enformer']['references'][ref_key]
    genome_config = config['genomes'][reference_config['genome']]
    return expand(rules.aggregate_prediction.output['aggregated_path'],
        allele_type='reference',
        num_agg_bins=mapper_config['num_agg_bins'],
        key=ref_key,
        subpath=[f'chrom={chr}/data.parquet' for chr in genome_config['chromosomes']])


rule train_mapper:
    priority: 4
    resources:
        mem_mb=lambda wildcards, attempt, threads: 50000 + (10000 * attempt)
    threads: 30
    output:
        tissue_mapper_path=output_path / 'mappers/{mapper_key}_{ref_key}.pkl',
    input:
        enformer_paths=train_mapper_input
    script:
        '../scripts/enformer/train_mapper.py'


def predict_alternative_input(wildcards):
    alt_key = wildcards['alt_key']
    ref_key = lookup(f'enformer/alternatives/{alt_key}/reference',within=config)
    return expand(rules.genome.output[0],
        genome=lookup(dpath=f"enformer/references/{ref_key}/genome",within=config))


rule tissue_expression:
    priority: 4
    resources:
        mem_mb=lambda wildcards, attempt, threads: 6000 + (1000 * attempt)
    output:
        prediction_path=output_path / '{allele_type}/expression/{mapper_key}_{ref_key}/{key}.parquet/{subpath}',
    input:
        enformer_path=expand(rules.aggregate_prediction.output['aggregated_path'],
            allele_type='{allele_type}',
            num_agg_bins=lookup(dpath='enformer/mappers/{mapper_key}/num_agg_bins',within=config),
            key='{key}',
            subpath='{subpath}'),
        tissue_mapper_path=expand(rules.train_mapper.output[0],mapper_key='{mapper_key}',ref_key='{ref_key}')
    script:
        '../scripts/enformer/tissue_expression.py'


rule predict_alternative:
    priority: 3
    resources:
        gpu=1,
        mem_mb=lambda wildcards, attempt, threads: 12000 + (1000 * attempt)
    output:
        prediction_path=output_path / 'alternative/raw/{alt_key}.parquet' / '{vcf_name}.parquet',
    input:
        genome_path=predict_alternative_input
    wildcard_constraints:
        vcf_name='.*\.vcf\.gz'
    params:
        type='alternative'
    script:
        '../scripts/enformer/predict_expression.py'
