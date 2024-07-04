import pathlib

# the path to the output folder
output_path = pathlib.Path(config["output_path"]) / 'process' / 'common'

rule genome:
    priority: 5
    resources:
        mem_mb=lambda wildcards, attempt, threads: 20000 + (1000 * attempt)
    output:
        output_path / 'genomes/{genome}.parquet'
    input:
        gtf_file=(lambda wildcards: config['genomes'][wildcards['genome']]["gtf_file"])
    script:
        '../scripts/common/genome.py'
