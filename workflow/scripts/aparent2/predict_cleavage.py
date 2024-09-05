import pathlib

from kipoi_aparent2.aparent2 import Aparent2
from kipoi_aparent2.dataloader import ApaDataloader
from kipoi_aparent2.dataloader.apa_annotation import EnsemblAPAAnnotation
from kipoi_aparent2.logger import setup_logger
from kipoi_aparent2 import constants
import pandas as pd

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params

logger = setup_logger()
test_config = config.get('test', None)

genome_df = pd.read_parquet(input_['genome_path'])
dl_args = {}
chromosome = None
if params['type'] == 'reference':
    allele = constants.AlleleType.REF
    logger.info('Allele type: %s', allele)
    chromosome = wildcards['chromosome']
    logger.info('Predicting for chromosome: %s', chromosome)

    # load reference config
    ref_config = config['aparent2']['references'][wildcards['ref_key']]
elif params.type == 'alternative':
    allele = constants.AlleleType.ALT
    logger.info('Allele type: %s', allele)
    # load alternative config
    alt_config = config['aparent2']['alternatives'][wildcards['alt_key']]
    # load reference config
    ref_config = config['aparent2']['references'][alt_config['reference']]
    # load VCF config
    vcf_config = config['vcfs'][alt_config['vcf']]
    vcf_file = pathlib.Path(vcf_config['path']) / f'{wildcards["vcf_name"]}'

    logger.info('Using VCF file: %s', vcf_file)
    dl_args.update({'vcf_file': vcf_file,
                    'variant_upstream_cse': alt_config['variant_upstream_cse'],
                    'variant_downstream_cse': alt_config['variant_downstream_cse'],
                    'vcf_lazy': True})
else:
    raise ValueError(f'invalid allele type {params["type"]}')

apa_config = config['apa_annotation'][ref_config['apa_annotation']]
# load genome config
genome_config = config['genomes'][apa_config['genome']]
if apa_config['type'] == 'ensembl':
    apa_annotation = EnsemblAPAAnnotation(genome_df, chromosome=chromosome,
                                          canonical_only=genome_config['canonical_only'],
                                          protein_coding_only=genome_config['protein_coding_only'])
else:
    raise ValueError(f'invalid APA annotation {ref_config["apa_annotation"]}')

# update dataloader args
dl_args = {**dl_args, **{'fasta_file': genome_config['fasta_file'],
                         'size': None if test_config is None else test_config['dataloader_size'],
                         'apa_annotation': apa_annotation}}

dl = ApaDataloader.from_allele_type(allele, **dl_args, )
# load model config
model_config = config['aparent2']['models'][ref_config['model']]
aparent2 = Aparent2(model_path=model_config['path'])
aparent2.predict(dl, batch_size=model_config['batch_size'], filepath=pathlib.Path(output['prediction_path']))
