from pathlib import Path
from kipoi_enformer.enformer import Enformer
from kipoi_enformer.dataloader import VCFTSSDataloader
from kipoi_enformer.logger import setup_logger

logger = setup_logger()

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output
wildcards = snakemake.wildcards

vcf = config['vcfs'][wildcards.vcf_name]

is_test = config['enformer'].get('is_test', False)
enformer = Enformer(is_test=is_test)

base_dir = Path(output['prediction_dir']).parent
base_dir.mkdir(exist_ok=True, parents=False)

dl = VCFTSSDataloader(gtf_file=input_['gtf_file'], fasta_file=input_['fasta_file'],
                      shift=config['enformer']['shift'], vcf_lazy=True, vcf_file=input_['vcf_file'],
                      variant_upstream_tss=vcf['variant_upstream_tss'],
                      variant_downstream_tss=vcf['variant_downstream_tss'],
                      protein_coding_only=True, canonical_only=True, size=5 if is_test else None)
enformer.predict(dl, batch_size=config['enformer']['batch_size'], filepath=output['prediction_dir'])
