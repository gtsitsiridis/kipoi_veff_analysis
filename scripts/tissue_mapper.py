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

for chr_file in input_['enformer_dir'].glob('*/'):
    tissue_mapper.predict(chr_file, output_path=output['prediction_dir'] / chr_file.name)
