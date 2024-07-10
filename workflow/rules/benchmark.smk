import pathlib

# the path to the output folder
benchmark_path = pathlib.Path(config["output_path"]) / 'benchmark.parquet'
veff_path = pathlib.Path(config["output_path"]) / 'veff.parquet'
evaluation_path = pathlib.Path(config["output_path"]) / 'evaluation'

module veff_workflow:
    snakefile: "veff.smk"
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
        mem_mb=lambda wildcards, attempt, threads: 120000 + (1000 * attempt)
    output:
        benchmark_path=benchmark_path / 'predictor={predictor}/run={run_key}/data.parquet'
    input:
        veff_path=benchmark_input
    script:
        '../scripts/common/benchmark.py'

rule evaluation:
    priority: 1
    resources:
        mem_mb=lambda wildcards, attempt, threads: 120000 + (1000 * attempt)
    log:
        notebook=evaluation_path / 'notebooks/{predictor}-{run_key}.ipynb'
    output:
        prc_path = evaluation_path / 'prc.parquet' / 'predictor={predictor}/run={run_key}/data.parquet',
        prc_tissue_path = evaluation_path / 'prc_tissue.parquet' / 'predictor={predictor}/run={run_key}/data.parquet',
        prc_tissue_type_path = evaluation_path / 'prc_tissue_type.parquet' / 'predictor={predictor}/run={run_key}/data.parquet',
        prc_fold_path = evaluation_path / 'prc_fold.parquet' / 'predictor={predictor}/run={run_key}/data.parquet',
        r2_path= evaluation_path / 'r2.parquet' / 'predictor={predictor}/run={run_key}/data.parquet',
        r2_tissue_path= evaluation_path / 'r2_tissue.parquet' / 'predictor={predictor}/run={run_key}/data.parquet',
        r2_tissue_type_path= evaluation_path / 'r2_tissue_type.parquet' / 'predictor={predictor}/run={run_key}/data.parquet',
        r2_fold_path= evaluation_path / 'r2_fold.parquet' / 'predictor={predictor}/run={run_key}/data.parquet',
    input:
        benchmark_path=rules.benchmark.output.benchmark_path
    notebook:
        "../notebooks/evaluation.py.ipynb"
