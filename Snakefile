import pathlib

configfile: "config.dev.yaml"

output_dir = pathlib.Path(config["output_dir"])
(output_dir / 'enformer/raw/ref').mkdir(parents=True,exist_ok=True)
(output_dir / 'enformer/raw/alt').mkdir(parents=False,exist_ok=True)
(output_dir / 'enformer/tissue/ref').mkdir(parents=True,exist_ok=True)
(output_dir / 'enformer/tissue/alt').mkdir(parents=False,exist_ok=True)
(output_dir / 'enformer/tissue/veff').mkdir(parents=False,exist_ok=True)


rule enformer:
    resources:
        gpu=1,
        ntasks=1
    output:
        prediction_dir=directory(f'{output_dir}/enformer/raw/' + '{allele}/{name}.parquet',)
    input:
        gtf_file=config["genome"]["gtf_file"],
        fasta_file=config["genome"]["fasta_file"],
    script:
        'scripts/enformer.py'

rule tissue_mapper:
    output:
        prediction_dir=directory(f'{output_dir}/enformer/tissue/' + '{path}.parquet',)
    input:
        enformer_dir=directory(f'{output_dir}/enformer/raw/' + '{path}.parquet',),
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
        expand(rules.enformer.output,allele='ref',name='reference'),
        expand(rules.enformer.output,allele='alt',name=vcf_names())

    # expand(rules.tissue_mapper.output,path='ref/reference'),
    # alt_tissue_mapper_files(),
    # veff_files(),


    # todo figure out resources
    # CONDA_OVERRIDE_CUDA="11.8" SBATCH_ARGS="--partition=standard --exclude=ouga[01-04]"
    # N_CORES=4096 MEM_MB=8192000 N_JOBS=200 N_GPUS=128
    # run_slurm_jobs.sh --rerun-incomplete --rerun-triggers mtime -k --restart-times 3 --use-conda --show-failed-logs
