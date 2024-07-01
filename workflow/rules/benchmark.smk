import pathlib

# the path to the output folder
benchmark_path = pathlib.Path(config["output_path"]) / 'benchmark.parquet'
veff_path = pathlib.Path(config["output_path"]) / 'veff.parquet'

module veff_workflow:
    snakefile: "../rules/veff.smk"
    config: config

use rule * from veff_workflow


def vcfs(vcf_key):
    # extract name from vcf file
    return [f.name for f in pathlib.Path(config["vcfs"][vcf_key]["path"]).glob('*.vcf.gz')]


def benchmark_input(wildcards):
    predictor, run_key = wildcards['predictor'], wildcards['run_key']

    run_config = config['runs'][predictor][run_key]
    vcf_key = config[predictor]['alternatives'][run_config['alternative']]['vcf']

    return [
        veff_path / f'predictor={predictor}/run={run_key}/{vcf}.parquet'
        for vcf in vcfs(vcf_key)
    ]


rule benchmark:
    priority: 1
    resources:
        mem_mb=lambda wildcards, attempt, threads: 20000 + (1000 * attempt)
    output:
        benchmark_path=benchmark_path / 'predictor={predictor}/run={run_key}/data.parquet'
    input:
        veff_path=benchmark_input
    script:
        '../scripts/common/benchmark.py'
