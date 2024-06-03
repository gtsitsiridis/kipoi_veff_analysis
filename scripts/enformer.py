import pathlib

from kipoi_enformer.enformer import Enformer
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

gtf = pd.read_parquet(input_['gtf'])
args = {'fasta_file': input_['fasta_file'],
        'shift': config['enformer']['shift'], 'protein_coding_only': config['genome']['protein_coding_only'],
        'canonical_only': config['genome']['canonical_only'],
        'size': None if test_config is None else test_config['dataloader_size'],
        'gtf': gtf}

# Check if VCF file is provided
# If VCF file is provided, predict for ALT allele
# If VCF file is not provided, predict for REF allele
if params.type == 'vcf':
    allele = constants.AlleleType.ALT
    vcf = config['vcf']
    vcf_file = input_['vcf_file']
    args.update({'vcf_file': vcf_file, 'variant_upstream_tss': vcf['variant_upstream_tss'],
                 'variant_downstream_tss': vcf['variant_downstream_tss'], 'vcf_lazy': True})
    logger.info('Allele type: %s', allele)
    logger.info('Using VCF file: %s', vcf_file)
elif params.type == 'ref':
    allele = constants.AlleleType.REF
    logger.info('Allele type: %s', allele)
    chromosome = wildcards['chromosome']
    args['chromosome'] = chromosome
    logger.info('Predicting for chromosome: %s', chromosome)
else:
    raise ValueError('Invalid allele type: %s' % params.type)

dl = TSSDataloader.from_allele_type(allele, **args, )
enformer = Enformer(is_random=False if test_config is None else test_config['is_random_enformer'])
enformer.predict(dl, batch_size=config['enformer']['batch_size'], filepath=pathlib.Path(output['prediction_path']),
                 num_output_bins=config['enformer']['num_output_bins'])
