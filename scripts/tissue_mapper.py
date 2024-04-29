from kipoi_enformer.enformer import EnformerTissueMapper
from kipoi_enformer.logger import setup_logger

logger = setup_logger()

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards


tissue_mapper = EnformerTissueMapper(tracks_path=input_['tracks_path'],
                                     tissue_matcher_path=input_['tissue_matcher_path'])
tissue_mapper.predict(input_['enformer_path'], output_path=output['prediction_path'],
                      num_bins=config['enformer']['tissue_matcher']['nbins'])
