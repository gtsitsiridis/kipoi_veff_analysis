import pyranges as pr
import pandas as pd
import logging
from kipoi_enformer.logger import setup_logger

# SNAKEMAKE SCRIPT
params = snakemake.params
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
config = snakemake.config
genome_config = config['genomes'][wildcards['genome']]

if config.get('debug', False):
    logger = setup_logger(logging.DEBUG)
else:
    logger = setup_logger()

logger.info('Reading GTF file: %s', input_['gtf_file'])
gtf = pr.read_gtf(input_['gtf_file'], as_df=True, duplicate_attr=True)
chromosomes = genome_config['chromosomes']
gtf = gtf.query('`Chromosome`.isin(@chromosomes)')
gtf.to_parquet(output[0])
