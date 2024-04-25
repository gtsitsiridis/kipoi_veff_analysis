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

if config.get('debug', False):
    logger = setup_logger(logging.DEBUG)
else:
    logger = setup_logger()

is_test = config['enformer'].get('is_test', False)

# preload gtf file to save time
logger.info('Loading GTF file')
gtf = pr.read_gtf(input_['gtf_file'], as_df=True, duplicate_attr=True)

args = {'fasta_file': input_['fasta_file'],
        'shift': config['enformer']['shift'], 'protein_coding_only': True,
        'canonical_only': True, 'size': 5 if is_test else None}

if input_.get('vcf_file', None):
    allele = constants.AlleleType.ALT
    vcf = config['vcf']
    vcf_file = input_['vcf_file']
    args = {'vcf_file': vcf_file, 'variant_upstream_tss': vcf['variant_upstream_tss'],
            'variant_downstream_tss': vcf['variant_downstream_tss'], 'vcf_lazy': True,
            **args}
    logger.info('Allele type: %s', allele)
    logger.info('Using VCF file: %s', vcf_file)
else:
    allele = constants.AlleleType.REF
    logger.info('Allele type: %s', allele)

base_path = pathlib.Path(output['prediction_dir'])
for chromosome in constants.Chromosome:
    logger.info('Predicting for chromosome: %s', chromosome)
    chromosome = chromosome.value
    output_path = base_path / chromosome
    dl = TSSDataloader.from_allele_type(allele, **args, chromosome=chromosome, gtf=gtf.copy())
    enformer = Enformer(is_test=is_test)
    enformer.predict(dl, batch_size=config['enformer']['batch_size'], filepath=output_path)
