import pathlib

# the path to the output folder
output_path = pathlib.Path(config["output_path"]) / 'process' / 'aparent2'
veff_path = pathlib.Path(config["output_path"]) / 'veff.parquet'

module common_workflow:
    snakefile: "common.smk"
    config: config

use rule genome from common_workflow

def get_genome_path(ref_key):
    apa_annot = config['aparent2']['references'][ref_key]['apa_annotation']
    genome = config['apa_annotation'][apa_annot]['genome']
    return expand(rules.genome.output[0],genome=genome)

rule predict_reference:
    priority: 5
    resources:
        gpu=1,
        mem_mb=lambda wildcards, attempt, threads: 12000 + (1000 * attempt)
    output:
        prediction_path=output_path / 'reference/{ref_key}.parquet' / 'chrom={chromosome}/data.parquet',
    input:
        genome_path=lambda wildcards: get_genome_path(wildcards['ref_key'])
    params:
        type='reference'
    conda:
        f'../envs/kipoi-aparent2{"" if not config.get("aparent2_use_gpu", False) else "-gpu"}.yml'
    script:
        '../scripts/aparent2/predict_cleavage.py'

def predict_alternative_input(wildcards):
    alt_key = wildcards['alt_key']
    ref_key = lookup(f'aparent2/alternatives/{alt_key}/reference',within=config)
    return get_genome_path(ref_key)

rule predict_alternative:
    priority: 3
    resources:
        gpu=1,
        mem_mb=lambda wildcards, attempt, threads: 12000 + (1000 * attempt)
    output:
        prediction_path=output_path / 'alternative/{alt_key}.parquet' / '{vcf_name}.parquet',
    input:
        genome_path=predict_alternative_input
    wildcard_constraints:
        vcf_name=r'.*\.vcf\.gz'
    params:
        type='alternative'
    conda:
        f'../envs/kipoi-aparent2{"" if not config.get("aparent2_use_gpu", False) else "-gpu"}.yml'
    script:
        '../scripts/aparent2/predict_cleavage.py'


def variant_effect_input(wildcards):
    run_key, vcf_name = wildcards['run_key'], wildcards['vcf_name']
    run_config = config['runs'][run_key]
    alternative_config = config['aparent2']['alternatives'][run_config['alternative']]
    reference_config = config['aparent2']['references'][alternative_config['reference']]
    apa_config = config['apa_annotation'][reference_config['apa_annotation']]
    genome_config = config['genomes'][apa_config['genome']]
    return {
        'ref_paths': expand(rules.predict_reference.output[0],
            ref_key=alternative_config['reference'],
            chromosome=genome_config['chromosomes']),
        'alt_path': expand(rules.predict_alternative.output[0],
            alt_key=run_config['alternative'],
            ref_key=alternative_config['reference'],
            vcf_name=vcf_name),
        'genome_path': expand(rules.genome.output[0],genome=apa_config['genome']),
    }

rule variant_effect:
    priority: 2
    resources:
        tasks=1,
        mem_mb=lambda wildcards, attempt, threads: 10000 + (1000 * attempt)
    output:
        veff_path=veff_path / 'model=aparent2/run={run_key}/{vcf_name}.parquet',
    input:
        unpack(variant_effect_input)
    wildcard_constraints:
        vcf_name=r'.*\.vcf\.gz'
    conda:
        f'../envs/kipoi-aparent2{"" if not config.get("aparent2_use_gpu", False) else "-gpu"}.yml'
    script:
        '../scripts/aparent2/veff.py'
