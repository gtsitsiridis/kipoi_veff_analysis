from kipoi_enformer.enformer import Enformer
from kipoi_enformer.dataloader import RefTSSDataloader
from kipoi_enformer.logger import setup_logger

logger = setup_logger()

# SNAKEMAKE SCRIPT
config = snakemake.config
input_ = snakemake.input
output = snakemake.output

is_test = config['enformer'].get('is_test', False)
enformer = Enformer(is_test=is_test)
dl = RefTSSDataloader(gtf_file=input_['gtf_file'], fasta_file=input_['fasta_file'],
                      shift=config['enformer']['shift'], protein_coding_only=True,
                      canonical_only=True, size=5 if is_test else None)
enformer.predict(dl, batch_size=config['enformer']['batch_size'], filepath=output['prediction_dir'])
