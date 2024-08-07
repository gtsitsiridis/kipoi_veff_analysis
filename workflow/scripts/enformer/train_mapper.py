from kipoi_veff_analysis.enformer import EnformerTissueMapper
from kipoi_veff_analysis.logger import setup_logger
import logging
import sklearn
import lightgbm as lgb
import pickle

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

test_config = config.get('test', None)

# For developing and testing purposes, we can use a precomputed mapper
if test_config is not None and test_config.get('precomputed_enformer_mapper_path', False):
    logger.info('Using precomputed enformer mapper')
    with open(test_config['precomputed_enformer_mapper_path'], 'rb') as f:
        tissue_mapper = pickle.load(f)

    with open(output['tissue_mapper_path'], 'wb') as f:
        pickle.dump(tissue_mapper, f, protocol=pickle.HIGHEST_PROTOCOL)
    exit(0)

mapper_config = config['enformer']['mappers'][wildcards['mapper_key']]
tissue_mapper = EnformerTissueMapper(tracks_path=mapper_config['tracks_path'], tissue_mapper_path=None)

model_params = mapper_config.get('params', {})
if mapper_config['type'] == 'ElasticNet':
    model = sklearn.linear_model.ElasticNetCV(cv=model_params.get('cv', 5),
                                              max_iter=model_params.get('max_iter', 1000), )
elif mapper_config['type'] == 'Ridge':
    model = sklearn.linear_model.RidgeCV(**model_params)
elif mapper_config['type'] == 'LightGBM':
    model = lgb.LGBMRegressor(**model_params)
else:
    raise ValueError('Invalid mapper type: %s' % mapper_config['type'])

tissue_mapper.train(agg_enformer_paths=list(input_['enformer_paths']), expression_path=mapper_config['expression_path'],
                    output_path=str(output['tissue_mapper_path']), model=model)
