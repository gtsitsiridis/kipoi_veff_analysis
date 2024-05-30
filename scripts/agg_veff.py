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

aggregate_veff(str(input_['veff']), config['genome']['isoform_file'], str(output['agg_veff']))
