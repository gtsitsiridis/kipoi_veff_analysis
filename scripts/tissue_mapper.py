from kipoi_enformer.enformer import EnformerTissueMapper
from kipoi_enformer.logger import setup_logger
from pathlib import Path

logger = setup_logger()

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards

tissue_mapper = EnformerTissueMapper(tracks_path=input_['tracks_path'],
                                     tissue_matcher_path=input_['tissue_matcher_path'])
tissue_mapper.predict(input_['enformer_dir'], output_path=output['prediction_dir'],
                      num_workers=config['enformer']['tissue_matcher']['num_workers'])
