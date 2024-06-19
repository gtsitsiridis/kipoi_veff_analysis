from kipoi_enformer.enformer import EnformerAggregator
from kipoi_enformer.logger import setup_logger
import logging

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards

if config.get('debug', False):
    logger = setup_logger(logging.DEBUG)
else:
    logger = setup_logger()

aggregator = EnformerAggregator()
aggregator.aggregate(input_['prediction_path'], output_path=output['aggregated_path'],
                     num_bins=int(wildcards['num_agg_bins']))
