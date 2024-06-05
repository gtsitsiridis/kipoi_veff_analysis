from kipoi_enformer.enformer import aggregate_veff
from kipoi_enformer.logger import setup_logger
import logging

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

if config['genome']['canonical_only']:
    isoforms_path = None
else:
    isoforms_path = config['genome']['isoform_file']

logger.info('isoforms path: ' + str(isoforms_path))

aggregate_veff(veff_path=str(input_['transcript_veff']), output_path=str(output['gene_veff']),
               isoforms_path=isoforms_path, mode='mean')
