import pytest
import logging
from kipoi_enformer.logger import logger
from pathlib import Path


@pytest.fixture(autouse=True)
def setup_logger():
    # Use `-p no:logging -s` in Pycharm's additional arguments to view logs
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(logging.DEBUG)


@pytest.fixture
def chr22_example_files():
    base = Path("assets/example_files")
    return {
        'fasta': base / "seq.fa",
        'gtf': base / "annot.gtf.gz",
        'vcf': base / "vcf" / "chr22_var.vcf.gz",
        'isoform_proportions': base / "isoform_proportions.tsv",
        'gtex_expression': base / 'gtex_transcripts_tpms.zarr'
    }
