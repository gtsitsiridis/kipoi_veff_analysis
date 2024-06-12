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

model = sklearn.linear_model.ElasticNetCV(cv=tm_config.get('cv', 5), max_iter=tm_config.get('max_iter', 1000), )
tissue_mapper.train(agg_enformer_path=params['enformer_path'], expression_path=input_['expression_path'],
                    output_path=str(output['tissue_mapper_path']), model=model)
