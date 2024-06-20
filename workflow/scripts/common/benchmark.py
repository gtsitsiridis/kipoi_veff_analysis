import pyranges as pr
import logging
from kipoi_enformer.logger import setup_logger
from kipoi_enformer.benchmark import VeffBenchmark

# SNAKEMAKE SCRIPT
params = snakemake.params
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
config = snakemake.config

if config.get('debug', False):
    logger = setup_logger(logging.DEBUG)
else:
    logger = setup_logger()

benchmark = VeffBenchmark(annotation_path=config['benchmark']['annotation_path'],
                          genotypes_path=config['benchmark']['genotypes_path'])

benchmark.run(list(input_['veff_path']), output_path=output['benchmark_path'])
