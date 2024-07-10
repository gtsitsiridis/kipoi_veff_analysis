import pathlib

from kipoi_enformer.enformer import Enformer, EnformerAggregator
from kipoi_enformer.dataloader import TSSDataloader
from kipoi_enformer.logger import setup_logger
from kipoi_enformer import constants
import logging
import pandas as pd

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params

if config.get('debug', False):
    logger = setup_logger(logging.DEBUG)
else:
    logger = setup_logger()

test_config = config.get('test', None)

genome_df = pd.read_parquet(input_['genome_path'])
dl_args = {}
if params['type'] == 'reference':
    allele = constants.AlleleType.REF
    logger.info('Allele type: %s', allele)

    chromosome = wildcards['chromosome']
    dl_args['chromosome'] = chromosome
    logger.info('Predicting for chromosome: %s', chromosome)

    # load reference config
    ref_config = config['enformer']['references'][wildcards['ref_key']]
    # load model config
    model_config = config['enformer']['models'][ref_config['model']]
    # load genome config
    genome_config = config['genomes'][ref_config['genome']]
elif params.type == 'alternative':
    allele = constants.AlleleType.ALT
    logger.info('Allele type: %s', allele)

    # load alternative config
    alt_config = config['enformer']['alternatives'][wildcards['alt_key']]
    # load reference config
    ref_config = config['enformer']['references'][alt_config['reference']]
    # load model config
    model_config = config['enformer']['models'][ref_config['model']]
    # load genome config
    genome_config = config['genomes'][ref_config['genome']]
    # load VCF config
    vcf_config = config['vcfs'][alt_config['vcf']]
    vcf_file = pathlib.Path(vcf_config['path']) / f'{wildcards["vcf_name"]}'

    logger.info('Using VCF file: %s', vcf_file)
    dl_args.update({'vcf_file': vcf_file,
                    'variant_upstream_tss': vcf_config['variant_upstream_tss'],
                    'variant_downstream_tss': vcf_config['variant_downstream_tss'],
                    'vcf_lazy': True})
else:
    raise ValueError(f'invalid allele type {params["type"]}')

dl_args = dl_args | {'fasta_file': genome_config['fasta_file'],
                     'shift': model_config['shift'],
                     'protein_coding_only': genome_config['protein_coding_only'],
                     'canonical_only': genome_config['canonical_only'],
                     'size': None if test_config is None else test_config['dataloader_size'],
                     'gtf': genome_df}

dl = TSSDataloader.from_allele_type(allele, **dl_args, )
# for development purposes and testing
if test_config is not None and test_config['is_random_enformer']:
    if allele == constants.AlleleType.ALT:
        enformer = Enformer(is_random=True, lamda=2)
    else:
        enformer = Enformer(is_random=True, lamda=10)
else:
    enformer = Enformer()
enformer.predict(dl, batch_size=model_config['batch_size'], filepath=pathlib.Path(output['prediction_path']),
                 num_output_bins=model_config['num_output_bins'])
