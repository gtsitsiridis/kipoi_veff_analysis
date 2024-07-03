import pathlib

# the path to the output folder
veff_path = pathlib.Path(config["output_path"]) / 'veff.parquet'

module enfomer_workflow:
    snakefile: "../rules/enformer.smk"
    config: config

use rule * from enfomer_workflow as enformer_ *


def enformer_variant_effect_input(wildcards):
    run_key, vcf_name = wildcards['run_key'], wildcards['vcf_name']
    run_config = config['runs']['enformer'][run_key]
    alternative_config = config['enformer']['alternatives'][run_config['alternative']]
    reference_config = config['enformer']['references'][alternative_config['reference']]
    genome_config = config['genomes'][reference_config['genome']]
    return {
        'ref_paths': expand(rules.enformer_tissue_expression.output[0],
            allele_type='reference',
            mapper_key=run_config['mapper'],
            key=alternative_config['reference'],
            ref_key=alternative_config['reference'],
            subpath=[f'chrom={chr}/data.parquet' for chr in genome_config['chromosomes']]),
        'alt_path': expand(rules.enformer_tissue_expression.output[0],
            allele_type='alternative',
            mapper_key=run_config['mapper'],
            key=run_config['alternative'],
            ref_key=alternative_config['reference'],
            subpath=f'{vcf_name}.parquet'),
        'genome_path': expand(rules.enformer_genome.output[0],genome=reference_config['genome']),
    }


rule enformer_variant_effect:
    priority: 2
    resources:
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 10000 + (1000 * attempt)
    output:
        veff_path=veff_path / 'predictor=enformer/run={run_key}/{vcf_name}.parquet',
    input:
        unpack(enformer_variant_effect_input)
    wildcard_constraints:
        vcf_name='.*\.vcf\.gz'
    params:
        isoform_file=lookup(dpath='runs/enformer/{run_key}/isoform_file',within=config),
        aggregation_mode=lookup(dpath='runs/enformer/{run_key}/aggregation_mode',within=config),
        upstream_tss=lookup(dpath='runs/enformer/{run_key}/upstream_tss',within=config),
        downstream_tss=lookup(dpath='runs/enformer/{run_key}/downstream_tss',within=config),
    script:
        '../scripts/enformer/veff.py'
