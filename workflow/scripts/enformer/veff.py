from kipoi_enformer.enformer import EnformerVeff
from kipoi_enformer.logger import setup_logger
import logging
import pandas as pd

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = config['runs'][wildcards['run_key']]

logger = setup_logger()
logger.info(f'Running veff on {input_["alt_path"]}')
logger.info(params)

genome_df = pd.read_parquet(input_['genome_path'])
isoforms_path = wildcards['run_key']['alternative']['reference']['genome'].get('isoform_proportion_file', None)
veff = EnformerVeff(isoforms_path=isoforms_path, gtf=genome_df)
veff.run(input_['ref_paths'], input_['alt_path'], output['veff_path'],
         aggregation_mode=params['aggregation_mode'],
         upstream_tss=params.get('upstream_tss', None),
         downstream_tss=params.get('downstream_tss', None))
