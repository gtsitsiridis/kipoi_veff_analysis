from kipoi_enformer.enformer import calculate_veff
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

calculate_veff(params['ref_tissue_pred_dir'] + '/**/data.parquet', input_['alt_tissue_pred'], output['veff'])
