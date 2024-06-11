import pyranges as pr
import pandas as pd
import logging
from kipoi_enformer.logger import setup_logger

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards

if config.get('debug', False):
    logger = setup_logger(logging.DEBUG)
else:
    logger = setup_logger()

logger.info('Reading GTF file: %s', input_['gtf_file'])
gtf = pr.read_gtf(input_['gtf_file'], as_df=True, duplicate_attr=True)
chromosomes = config['genome']['chromosomes']
gtf = gtf.query('`Chromosome`.isin(@chromosomes)')
gtf.to_parquet(output[0])