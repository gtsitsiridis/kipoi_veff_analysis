import pathlib
from snakemake.io import directory, expand

configfile: "config.dev.yaml"

output_dir = pathlib.Path(config["output_dir"])
(output_dir / 'enformer/raw/ref').mkdir(parents=True,exist_ok=True)
(output_dir / 'enformer/raw/alt').mkdir(parents=False,exist_ok=True)
(output_dir / 'enformer/tissue/ref').mkdir(parents=True,exist_ok=True)
(output_dir / 'enformer/tissue/alt').mkdir(parents=False,exist_ok=True)
(output_dir / 'enformer/tissue/veff').mkdir(parents=False,exist_ok=True)

rule enformer_reference:
    output:
        prediction_dir=directory(f'{output_dir}/enformer/raw/ref/reference.parquet',)
    input:
        gtf_file=config["genome"]["gtf_file"],
        fasta_file=config["genome"]["fasta_file"],
    script:
        'scripts/enformer_reference.py'


rule enformer_vcf:
    output:
        prediction_dir=directory(f'{output_dir}/enformer/raw/alt/' + '{vcf_name}/{vcf_id}.parquet')
    input:
        gtf_file=config["genome"]["gtf_file"],
        fasta_file=config["genome"]["fasta_file"],
        vcf_file=lambda wildcards: f'{config["vcfs"][wildcards.vcf_name]["dir"]}/{wildcards.vcf_id}.vcf'
    wildcard_constraints:
        vcf_name="[^/]+",
        vcf_id="[^/]+",
    script:
        'scripts/enformer_vcf.py'


rule tissue_mapper:
    output:
        prediction_dir=directory(f'{output_dir}/enformer/tissue/' + '{path}.parquet',)
    input:
        enformer_dir=directory(f'{output_dir}/enformer/raw/' + '{path}.parquet',),
        tracks_path=config['enformer']['tissue_matcher']['tracks_path'],
        tissue_matcher_path=config['enformer']['tissue_matcher']['model_path'],
    script:
        'scripts/tissue_mapper.py'


rule veff:
    output:
        veff=directory(f'{output_dir}/enformer/tissue/veff/' + '{alt_path}.parquet'),
    input:
        ref_tissue_pred=directory(f'{output_dir}/enformer/tissue/ref/reference.parquet',),
        alt_tissue_pred=directory(f'{output_dir}/enformer/tissue/alt/' + '{alt_path}.parquet',)
    script:
        'scripts/veff.py'


def vcf_files():
    files = []
    for vcf_name, vcf in config["vcfs"].items():
        for vcf_file in pathlib.Path(vcf["dir"]).glob('*.vcf'):
            files.extend(expand(rules.enformer_vcf.output,vcf_name=vcf_name,vcf_id=vcf_file.stem))
    return files


def raw_alt_files():
    files = []
    for vcf_name, vcf in config["vcfs"].items():
        for vcf_file in pathlib.Path(vcf["dir"]).glob('*.vcf'):
            files.extend(expand(rules.tissue_mapper.output,path=f'alt/{vcf_name}/{vcf_file.stem}'))
    return files


def veff_files():
    files = []
    for vcf_name, vcf in config["vcfs"].items():
        for vcf_file in pathlib.Path(vcf["dir"]).glob('*.vcf'):
            files.extend(expand(rules.veff.output,alt_path=f'{vcf_name}/{vcf_file.stem}'))
    return files


rule all:
    default_target: True
    input:
        expand(rules.tissue_mapper.output,path='ref/reference'),
        rules.enformer_reference.output,
        vcf_files(),
        raw_alt_files(),
        veff_files(),
