import pathlib

assert len(config) > 0, "The config file has not been defined or is empty"

output_dir = pathlib.Path(config["output_dir"])


def vcf_file(wildcards):
    return str(pathlib.Path(config['vcf']["path"]) / f'{wildcards.vcf_name}')


def tissue_mapper_path():
    config_path = config['enformer']['tissue_mapper'].get('model_path', None)
    if config_path:
        return config_path
    else:
        return output_dir / 'tissue_mapper.pkl'


rule gtf:
    priority: 5
    resources:
        mem_mb=lambda wildcards, attempt, threads: 20000 + (1000 * attempt)
    output:
        temp(output_dir / 'temp/gtf.parquet')
    input:
        gtf_file=config["genome"]["gtf_file"],
    script:
        'scripts/gtf.py'

rule enformer_ref:
    priority: 5
    resources:
        gpu=1,
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 12000 + (1000 * attempt)
    output:
        prediction_path=output_dir / 'raw/ref.parquet' / 'chrom={chromosome}/data.parquet',
    input:
        gtf=rules.gtf.output[0],
        fasta_file=config["genome"]["fasta_file"],
    params:
        type='ref'
    script:
        'scripts/enformer.py'

rule enformer_alt:
    priority: 2
    resources:
        gpu=1,
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 12000 + (1000 * attempt)
    output:
        prediction_path=output_dir / 'raw/alt.parquet' / 'vcf={vcf_name}/data.parquet',
    input:
        gtf=rules.gtf.output[0],
        fasta_file=config["genome"]["fasta_file"],
        vcf_file=vcf_file,
    wildcard_constraints:
        vcf_name='.*\.vcf\.gz'
    params:
        type='vcf'
    script:
        'scripts/enformer.py'

rule train_tissue_mapper:
    priority: 3
    resources:
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 50000 + (1000 * attempt)
    output:
        tissue_mapper_path=tissue_mapper_path(),
    input:
        enformer_paths = expand(output_dir / 'raw/ref.parquet' / 'chrom={chromosome}/data.parquet',
            chromosome=config['genome']['chromosomes']),
        tracks_path=config['enformer']['tissue_mapper']['tracks_path'],
        expression_path=config['enformer']['tissue_mapper']['expression_path']
    params:
        enformer_path=output_dir / 'raw/ref.parquet'
    script:
        'scripts/train_tissue_mapper.py'

rule enformer_to_tissue:
    priority: 3
    resources:
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 6000 + (1000 * attempt)
    output:
        prediction_path=output_dir / 'tissue' / '{path}',
    input:
        enformer_path=output_dir / 'raw' / '{path}',
        tracks_path=config['enformer']['tissue_mapper']['tracks_path'],
        tissue_mapper_path=rules.train_tissue_mapper.output
    script:
        'scripts/enformer_to_tissue.py'

rule transcript_veff:
    priority: 4
    resources:
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 8000 + (1000 * attempt)
    output:
        transcript_veff_path=output_dir / 'tissue/transcript_veff.parquet' / 'vcf={vcf_name}/data.parquet',
    input:
        ref_tissue_pred_paths=expand(output_dir / 'tissue/ref.parquet' / 'chrom={chromosome}/data.parquet',
            chromosome=config['genome']['chromosomes']),
        alt_tissue_pred_path=output_dir / 'tissue/alt.parquet' / 'vcf={vcf_name}/data.parquet',
    params:
        ref_tissue_pred_paths=output_dir / 'tissue/ref.parquet/**/data.parquet',
    wildcard_constraints:
        vcf_name='.*\.vcf\.gz'
    script:
        'scripts/transcript_veff.py'

rule gene_veff:
    priority: 4
    resources:
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 10000 + (1000 * attempt)
    output:
        gene_veff_path=output_dir / 'tissue/gene_veff.parquet' / 'vcf={vcf_name}/data.parquet',
    input:
        transcript_veff_path=rules.transcript_veff.output
    wildcard_constraints:
        vcf_name='.*\.vcf\.gz'
    script:
        'scripts/gene_veff.py'


def vcf_names():
    # extract name from vcf file
    return [f.name for f in pathlib.Path(config["vcf"]["path"]).glob('*.vcf.gz')]


rule all:
    default_target: True
    input:
        expand(rules.gene_veff.output,vcf_name=vcf_names())
    # expand(rules.enformer_ref.output,chromosome=config['genome']['chromosomes']),
    # expand(rules.veff.output, vcf_name=vcf_names())


    # CONDA_OVERRIDE_CUDA="11.8" SBATCH_ARGS="--partition=standard --exclude=ouga[01-04]" \
    # N_CORES=2 MEM_MB=30000 N_JOBS=23 N_GPUS=2 \
    # slurm_scripts/run_slurm_jobs.sh --rerun-incomplete --rerun-triggers mtime -k --restart-times 3 --use-conda --show-failed-logs \
    # --configfile config/config.prod.yaml
