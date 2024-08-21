from kipoi_enformer.enformer import EnformerAggregator
from kipoi_enformer.logger import setup_logger

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards

setup_logger()
aggregator = EnformerAggregator()
aggregator.aggregate(input_['prediction_path'], output_path=output['aggregated_path'],
                     num_bins=int(wildcards['num_agg_bins']))
