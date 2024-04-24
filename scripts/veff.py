from kipoi_enformer.enformer import calculate_veff
from kipoi_enformer.logger import setup_logger
import logging

logger = setup_logger()

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards

calculate_veff(input_['ref_tissue_pred'], input_['alt_tissue_pred'], output['veff'])
