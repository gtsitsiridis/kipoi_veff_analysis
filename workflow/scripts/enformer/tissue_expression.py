from kipoi_enformer.enformer import EnformerTissueMapper
from kipoi_enformer.logger import setup_logger
import logging

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards

logger = setup_logger()

mapper_config = config['enformer']['mappers'][wildcards['mapper_key']]
tissue_mapper = EnformerTissueMapper(tracks_path=mapper_config['tracks_path'],
                                     tissue_mapper_path=str(input_['tissue_mapper_path']))
tissue_mapper.predict(input_['enformer_path'], output_path=output['prediction_path'])
