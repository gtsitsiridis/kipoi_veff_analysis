from kipoi_enformer.enformer import EnformerTissueMapper
from kipoi_enformer.logger import setup_logger
import pathlib

logger = setup_logger()

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards

pathlib.Path(output['prediction_dir']).parent.mkdir(parents=True, exist_ok=True)

tissue_mapper = EnformerTissueMapper(tracks_path=input_['tracks_path'],
                                     tissue_matcher_path=input_['tissue_matcher_path'])
tissue_mapper.predict(input_['enformer_dir'], output_path=output['prediction_dir'],
                      num_bins=config['enformer']['tissue_matcher']['nbins'])
