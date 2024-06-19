import pathlib

# the path to the output folder
output_path = pathlib.Path(config["output_path"]) / 'enformer'

module common_workflow:
    snakefile: "common.smk"
    config: config

use rule genome from common_workflow


# todo
# performance evaluation
# runs/performance/<run>.parquet

# performance benchmarking
# benchmark.parquet


rule predict_reference:
    priority: 5
    resources:
        gpu=1,
        ntasks=1,
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


def predict_alternative_input(wildcards):
    alt_key = wildcards['alt_key']
    ref_key = lookup(f'enformer/alternatives/{alt_key}/reference',within=config)
    return expand(rules.genome.output[0],
        genome=lookup(dpath=f"enformer/references/{ref_key}/genome",within=config))


rule predict_alternative:
    priority: 3
    resources:
        gpu=1,
        ntasks=1,
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


rule aggregate_prediction:
    priority: 2
    resources:
        ntasks=1,
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
    priority: 2
    resources:
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 50000 + (10000 * attempt)
    threads: 30
    output:
        tissue_mapper_path=output_path / 'mappers/{mapper_key}_{ref_key}.pkl',
    input:
        enformer_paths=train_mapper_input
    script:
        '../scripts/enformer/train_mapper.py'


rule tissue_expression:
    priority: 3
    resources:
        ntasks=1,
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


def variant_effect_input(wildcards):
    run_key, vcf_name = wildcards['run_key'], wildcards['vcf_name']
    run_config = config['runs']['enformer'][run_key]
    alternative_config = config['enformer']['alternatives'][run_config['alternative']]
    reference_config = config['enformer']['references'][alternative_config['reference']]
    genome_config = config['genomes'][reference_config['genome']]
    return {
        'ref_paths': expand(rules.tissue_expression.output[0],
            allele_type='reference',
            mapper_key=run_config['mapper'],
            key=alternative_config['reference'],
            ref_key=alternative_config['reference'],
            subpath=[f'chrom={chr}/data.parquet' for chr in genome_config['chromosomes']]),
        'alt_path': expand(rules.tissue_expression.output[0],
            allele_type='alternative',
            mapper_key=run_config['mapper'],
            key=run_config['alternative'],
            ref_key=alternative_config['reference'],
            subpath=f'{vcf_name}.parquet')

    }


rule variant_effect:
    priority: 4
    resources:
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 10000 + (1000 * attempt)
    output:
        veff_path=output_path / 'runs/{run_key}/veff.parquet/{vcf_name}.parquet',
    input:
        unpack(variant_effect_input)
    wildcard_constraints:
        vcf_name='.*\.vcf\.gz'
    params:
        isoform_file=lookup(dpath='runs/enformer/{run_key}/isoform_file',within=config),
        aggregation_mode=lookup(dpath='runs/enformer/{run_key}/aggregation_mode',within=config)
    script:
        '../scripts/enformer/veff.py'