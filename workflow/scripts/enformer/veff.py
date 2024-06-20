from kipoi_enformer.enformer import EnformerVeff
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

logger.info(f'Running veff on {input_["alt_path"]}')
logger.info(params)
veff = EnformerVeff(isoforms_path=params.get('isoform_file', None),
                    aggregation_mode=params['aggregation_mode'],
                    upstream_tss=params.get('upstream_tss', None),
                    downstream_tss=params.get('downstream_tss', None), )
veff.run(input_['ref_paths'], input_['alt_path'], output['veff_path'])
