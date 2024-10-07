import pathlib

# the path to the output folder
benchmark_path = pathlib.Path(config["output_path"]) / 'benchmark.parquet'
veff_path = pathlib.Path(config["output_path"]) / 'veff.parquet'
evaluation_path = pathlib.Path(config["output_path"]) / 'evaluation'
comparison_path = pathlib.Path(config["output_path"]) / 'comparison'

module enfomer_workflow:
    snakefile: "enformer.smk"
    config: config

module aparent2_workflow:
    snakefile: "aparent2.smk"
    config: config

use rule * from enfomer_workflow as enformer_*
use rule * from aparent2_workflow as aparent2_*


def vcfs(vcf_key):
    # extract name from vcf file
    return [f.name for f in pathlib.Path(config["vcfs"][vcf_key]["path"]).glob('*.vcf.gz')]


def benchmark_input(wildcards):
    run_key = wildcards['run_key']
    run_config = config['runs'][run_key]
    predictor = run_config['predictor']
    vcf_key = config[predictor]['alternatives'][run_config['alternative']]['vcf']

    if predictor == 'enformer':
        return [*expand(rules.enformer_variant_effect.output,run_key=run_key,vcf_name=vcfs(vcf_key))]
    elif predictor == 'aparent2':
        return [*expand(rules.aparent2_variant_effect.output,run_key=run_key,vcf_name=vcfs(vcf_key))]
    elif predictor == 'custom':
        return veff_path / f'{run_key}.parquet/{vcf_name}.parquet'


rule benchmark:
    priority: 1
    resources:
        mem_mb=lambda wildcards, attempt, threads: 120000 + (1000 * attempt)
    output:
        benchmark_path=benchmark_path / 'run={run_key}/data.parquet'
    input:
        veff_path=benchmark_input
    script:
        '../scripts/common/benchmark.py'

rule evaluation:
    priority: 1
    resources:
        mem_mb=lambda wildcards, attempt, threads: 120000 + (1000 * attempt)
    log:
        notebook=evaluation_path / 'notebooks/{run_key}.py.ipynb'
    output:
        prc_path=evaluation_path / 'prc.parquet' / 'run={run_key}/data.parquet',
        prc_tissue_path=evaluation_path / 'prc_tissue.parquet' / 'run={run_key}/data.parquet',
        prc_tissue_type_path=evaluation_path / 'prc_tissue_type.parquet' / 'run={run_key}/data.parquet',
        prc_fold_path=evaluation_path / 'prc_fold.parquet' / 'run={run_key}/data.parquet',
        r2_path=evaluation_path / 'r2.parquet' / 'run={run_key}/data.parquet',
        r2_tissue_path=evaluation_path / 'r2_tissue.parquet' / 'run={run_key}/data.parquet',
        r2_tissue_type_path=evaluation_path / 'r2_tissue_type.parquet' / 'run={run_key}/data.parquet',
        r2_fold_path=evaluation_path / 'r2_fold.parquet' / 'run={run_key}/data.parquet',
    input:
        benchmark_path=rules.benchmark.output.benchmark_path
    notebook:
        "../notebooks/evaluation.py.ipynb"


rule comparison:
    priority: 1
    resources:
        mem_mb=lambda wildcards, attempt, threads: 6000 + (1000 * attempt)
    log:
        notebook=comparison_path / 'notebooks/{comparison_id}.r.ipynb'
    input:
        expand(rules.evaluation.output, run_key=lookup('comparisons/{comparison_id}', within=config))
    notebook:
        "../notebooks/comparison.r.ipynb"
