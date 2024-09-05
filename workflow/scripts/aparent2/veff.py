from kipoi_aparent2.aparent2 import Aparent2Veff
from kipoi_aparent2.dataloader.apa_annotation import EnsemblAPAAnnotation
from kipoi_aparent2.logger import setup_logger
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

run_config = config['runs'][wildcards['run_key']]
alternative_config = config['aparent2']['alternatives'][run_config['alternative']]
reference_config = config['aparent2']['references'][alternative_config['reference']]
apa_config = config['apa_annotation'][reference_config['apa_annotation']]
genome_config = config['genomes'][apa_config['genome']]
if apa_config['type'] == 'ensembl':
    apa_annotation = EnsemblAPAAnnotation(genome_df, canonical_only=genome_config['canonical_only'],
                                          protein_coding_only=genome_config['protein_coding_only'],
                                          isoform_usage_path=genome_config.get('isoform_proportion_file', None))
else:
    raise ValueError(f'invalid APA annotation {apa_config["type"]}')
veff = Aparent2Veff(apa_annotation)
veff.run([str(x) for x in input_['ref_paths']], str(input_['alt_path']), output['veff_path'],
         aggregation_mode=params['aggregation_mode'], use_narrow_score=params['use_narrow_score'],
         upstream_cse=params.get('upstream_cse', None), downstream_cse=params.get('downstream_cse', None), )
