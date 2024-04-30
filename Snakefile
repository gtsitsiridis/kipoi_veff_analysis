import pathlib

assert len(config) > 0, "The config file has not been defined or is empty"

output_dir = pathlib.Path(config["output_dir"])


def vcf_file(wildcards):
    return str(pathlib.Path(config['vcf']["path"]) / f'{wildcards.vcf_name}')


rule gtf_chrom_store:
    priority: 5
    resources:
        mem_mb=lambda wildcards, attempt, threads: 20000 + (1000 * attempt)
    output:
        temp(f'{output_dir}/temp/gtf_chrom_store.h5')
    input:
        gtf_file=config["genome"]["gtf_file"],
    script:
        'scripts/gtf_chrom_store.py'

rule enformer_ref:
    priority: 5
    resources:
        gpu=1,
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 12000 + (1000 * attempt)
    output:
        prediction_path=f'{output_dir}/raw/ref/reference.parquet/' + 'chrom={chromosome}/data.parquet',
    input:
        gtf_chrom_store=rules.gtf_chrom_store.output[0],
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
        prediction_path=f'{output_dir}/raw/alt/' + '{vcf_name}.parquet',
    input:
        gtf_file=config["genome"]["gtf_file"],
        fasta_file=config["genome"]["fasta_file"],
        vcf_file=vcf_file,
    wildcard_constraints:
        vcf_name='.*\.vcf\.gz'
    params:
        type='vcf'
    script:
        'scripts/enformer.py'

rule tissue_mapper:
    priority: 3
    resources:
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 6000 + (1000 * attempt)
    output:
        prediction_path=f'{output_dir}/tissue/' + '{path}',
    input:
        enformer_path=f'{output_dir}/raw/' + '{path}',
        tracks_path=config['enformer']['tissue_matcher']['tracks_path'],
        tissue_matcher_path=config['enformer']['tissue_matcher']['model_path'],
    script:
        'scripts/tissue_mapper.py'


rule veff:
    priority: 4
    resources:
        ntasks=1,
        mem_mb=lambda wildcards, attempt, threads: 2000 + (1000 * attempt)
    output:
        veff=f'{output_dir}/tissue/veff/' + '{vcf_name}.parquet',
    input:
        ref_tissue_pred=expand(f'{output_dir}/tissue/ref/reference.parquet/' + 'chrom={chromosome}/data.parquet',
            chromosome=config['genome']['chromosomes']),
        alt_tissue_pred=f'{output_dir}/tissue/alt/' + '{vcf_name}.parquet',
    params:
        ref_tissue_pred_dir=f'{output_dir}/tissue/ref/reference.parquet'
    wildcard_constraints:
        vcf_name='.*\.vcf\.gz'
    script:
        'scripts/veff.py'


def vcf_names():
    # extract name from vcf file
    return [f.name for f in pathlib.Path(config["vcf"]["path"]).glob('*.vcf.gz')]


rule all:
    default_target: True
    input:
        expand(rules.veff.output, vcf_name=vcf_names())
        # expand(rules.enformer_ref.output,chromosome=config['genome']['chromosomes']),
        # expand(rules.veff.output, vcf_name=vcf_names())


        # CONDA_OVERRIDE_CUDA="11.8" SBATCH_ARGS="--partition=standard --exclude=ouga[01-04]" \
        # N_CORES=2 MEM_MB=30000 N_JOBS=23 N_GPUS=2 \
        # slurm_scripts/run_slurm_jobs.sh --rerun-incomplete --rerun-triggers mtime -k --restart-times 3 --use-conda --show-failed-logs \
        # --configfile config/config.prod.yaml
