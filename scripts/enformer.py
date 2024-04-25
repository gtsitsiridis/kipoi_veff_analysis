import pathlib

from kipoi_enformer.enformer import Enformer
from kipoi_enformer.dataloader import TSSDataloader
from kipoi_enformer.logger import setup_logger
from kipoi_enformer import constants
import pyranges as pr
import logging

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards

output_dir = pathlib.Path(config["output_dir"]) / 'raw'
output_dir.mkdir(parents=False, exist_ok=True)

if config.get('debug', False):
    logger = setup_logger(logging.DEBUG)
else:
    logger = setup_logger()

test_config = config.get('test', None)

# preload gtf file to save time
logger.info('Loading GTF file')
gtf = pr.read_gtf(input_['gtf_file'], as_df=True, duplicate_attr=True)

args = {'fasta_file': input_['fasta_file'],
        'shift': config['enformer']['shift'], 'protein_coding_only': True,
        'canonical_only': True, 'size': None if test_config is None else test_config['dataloader_size']}

if input_.get('vcf_file', None):
    (output_dir / 'alt').mkdir(parents=False, exist_ok=True)
    allele = constants.AlleleType.ALT
    vcf = config['vcf']
    vcf_file = input_['vcf_file']
    args = {'vcf_file': vcf_file, 'variant_upstream_tss': vcf['variant_upstream_tss'],
            'variant_downstream_tss': vcf['variant_downstream_tss'], 'vcf_lazy': True,
            **args}
    logger.info('Allele type: %s', allele)
    logger.info('Using VCF file: %s', vcf_file)
else:
    (output_dir / 'ref').mkdir(parents=False, exist_ok=True)
    allele = constants.AlleleType.REF
    logger.info('Allele type: %s', allele)

base_path = pathlib.Path(output['prediction_dir'])
base_path.mkdir(parents=False, exist_ok=True)
for chromosome in config['genome']['chromosomes']:
    logger.info('Predicting for chromosome: %s', chromosome)
    output_path = base_path / chromosome
    dl = TSSDataloader.from_allele_type(allele, **args, chromosome=chromosome, gtf=gtf.copy())
    enformer = Enformer(is_random=False if test_config is None else test_config['is_random_enformer'])
    enformer.predict(dl, batch_size=config['enformer']['batch_size'], filepath=output_path)
