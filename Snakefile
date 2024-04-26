import pathlib

assert len(config) > 0, "The config file has not been defined or is empty"

output_dir = pathlib.Path(config["output_dir"])
output_dir.mkdir(parents=True,exist_ok=True)
(output_dir / 'temp').mkdir(exist_ok=True)

def vcf_file(wildcards):
    return str(pathlib.Path(config['vcf']["path"]) / f'{wildcards.vcf_name}.vcf.gz')


rule gtf_chrom_store:
    resources:
        mem_mb=lambda wildcards, attempt, threads: 20000 + (1000 * attempt)
    output:
        temp(f'{output_dir}/temp/gtf_chrom_store.h5')
    input:
        gtf_file=config["genome"]["gtf_file"],
        gtf_file_index=config["genome"]["gtf_file"] + '.tbi',
    script:
        'scripts/gtf_chrom_store.py'

rule enformer_ref:
    resources:
        gpu=1,
        ntasks=1,
        mem_mb= lambda wildcards,attempt,threads: 12000 + (1000 * attempt)
    output:
        prediction_dir=directory(f'{output_dir}/raw/ref/reference.parquet/' + 'chrom={chromosome}',)
    input:
        gtf_chrom_store=rules.gtf_chrom_store.output[0],
        fasta_file=config["genome"]["fasta_file"],
        fasta_file_index=config["genome"]["fasta_file"] + '.fai',
    script:
        'scripts/enformer.py'

rule enformer_alt:
    resources:
        gpu=1,
        ntasks=1
    output:
        prediction_dir=directory(f'{output_dir}/raw/alt/' + '{vcf_name}.parquet',)
    input:
        gtf_file=config["genome"]["gtf_file"],
        gtf_file_index=config["genome"]["gtf_file"] + '.tbi',
        fasta_file=config["genome"]["fasta_file"],
        fasta_file_index=config["genome"]["fasta_file"] + '.fai',
        vcf_file=vcf_file,
        vcf_file_index=lambda wildcards: vcf_file(wildcards) + '.tbi',
    script:
        'scripts/enformer.py'

rule tissue_mapper:
    output:
        prediction_dir=directory(f'{output_dir}/tissue/' + '{path}',)
    input:
        enformer_dir=directory(f'{output_dir}/raw/' + '{path}',),
        tracks_path=config['enformer']['tissue_matcher']['tracks_path'],
        tissue_matcher_path=config['enformer']['tissue_matcher']['model_path'],
    script:
        'scripts/tissue_mapper.py'


#
# rule veff:
#     output:
#         veff=directory(f'{output_dir}/enformer/tissue/veff/' + '{alt_path}.parquet'),
#     input:
#         ref_tissue_pred=directory(f'{output_dir}/enformer/tissue/ref/reference.parquet',),
#         alt_tissue_pred=directory(f'{output_dir}/enformer/tissue/alt/' + '{alt_path}.parquet',)
#     script:
#         'scripts/veff.py'


def vcf_names():
    # extract name from vcf file
    names = []
    for f in pathlib.Path(config["vcf"]["path"]).glob('*.vcf.gz'):
        while f.suffix:
            f = f.with_suffix('')
        names.append(f.name)
    return names


rule all:
    default_target: True
    input:
        expand(rules.enformer_ref.output,chromosome=config['genome']['chromosomes']),
        # expand(rules.enformer_alt.output,vcf_name=vcf_names()),
        # expand(rules.tissue_mapper.output,
        #     path=expand('ref/reference.parquet/chrom={chromosome}',
        #         chromosome=config['genome']['chromosomes'])
        # ),
        # expand(rules.tissue_mapper.output,
        #     path=expand('alt/{vcf_name}.parquet',
        #         vcf_name=vcf_names())
        # ),


        # todo figure out resources
        # around 11GB per job

        # CONDA_OVERRIDE_CUDA="11.8" SBATCH_ARGS="--partition=standard --exclude=ouga[01-04]" \
        # N_CORES=2 MEM_MB=30000 N_JOBS=23 N_GPUS=2 \
        # slurm_scripts/run_slurm_jobs.sh --rerun-incomplete --rerun-triggers mtime -k --restart-times 3 --use-conda --show-failed-logs \
        # --configfile config/config.prod.yaml
