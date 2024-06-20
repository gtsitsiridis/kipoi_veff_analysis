import pathlib

# the path to the output folder
output_path = pathlib.Path(config["output_path"])


def vcfs(vcf_key):
    # extract name from vcf file
    return [f.name for f in pathlib.Path(config["vcfs"][vcf_key]["path"]).glob('*.vcf.gz')]


def benchmark_input(wildcards):
    predictor, run_key = wildcards['predictor'], wildcards['run_key']

    run_config = config['runs'][predictor][run_key]
    vcf_key = config[predictor]['alternatives'][run_config['alternative']]['vcf']

    return [
        output_path / f'{predictor}/veff.parquet/run={run_key}/{vcf}.parquet'
        for vcf in vcfs(vcf_key)
    ]


rule benchmark:
    priority: 1
    resources:
        mem_mb=lambda wildcards, attempt, threads: 20000 + (1000 * attempt)
    output:
        benchmark_path=output_path / 'benchmark.parquet/predictor={predictor}/run={run_key}/data.parquet'
    input:
        veff_path=benchmark_input
    script:
        '../scripts/common/benchmark.py'
