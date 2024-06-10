from kipoi_enformer.enformer import EnformerTissueMapper
from kipoi_enformer.logger import setup_logger
import logging
import sklearn

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards
params = snakemake.params
tm_config = config['enformer']['tissue_mapper']

if config.get('debug', False):
    logger = setup_logger(logging.DEBUG)
else:
    logger = setup_logger()

tissue_mapper = EnformerTissueMapper(tracks_path=input_['tracks_path'], tissue_mapper_path=None)

model = sklearn.linear_model.ElasticNetCV(cv=5 if not tm_config['cv'] else tm_config['cv'],)
tissue_mapper.train(enformer_scores_path=params['enformer_path'], expression_path=input_['expression_path'],
                    output_path=str(output['tissue_mapper_path']),num_bins=tm_config['nbins'], model=model)
