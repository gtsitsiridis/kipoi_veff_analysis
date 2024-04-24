import pathlib

from kipoi_enformer.enformer import Enformer
from kipoi_enformer.dataloader import TSSDataloader
from kipoi_enformer.logger import setup_logger
from kipoi_enformer import constants
import logging

logger = setup_logger(logging.DEBUG)

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards

allele = constants.AlleleType(wildcards.allele)
is_test = config['enformer'].get('is_test', False)

args = {'gtf_file': input_['gtf_file'], 'fasta_file': input_['fasta_file'],
        'shift': config['enformer']['shift'], 'protein_coding_only': True,
        'canonical_only': True, 'size': 5 if is_test else None}

if allele == constants.AlleleType.ALT:
    vcf = config['vcf']
    vcf_file = pathlib.Path(vcf["path"]) / f'{wildcards.name}.vcf.gz'
    args = {'vcf_file': vcf_file, 'variant_upstream_tss': vcf['variant_upstream_tss'],
            'variant_downstream_tss': vcf['variant_downstream_tss'], 'vcf_lazy': True,
            **args}

base_path = pathlib.Path(output['prediction_dir'])
for chromosome in [constants.Chromosome.chr21, constants.Chromosome.chr22, ]:
    logger.info('Predicting for chromosome: %s', chromosome)
    chromosome = chromosome.value
    output_path = base_path / chromosome
    dl = TSSDataloader.from_allele_type(allele, **args, chromosome=chromosome)
    enformer = Enformer(is_test=is_test)
    enformer.predict(dl, batch_size=config['enformer']['batch_size'], filepath=output_path)
