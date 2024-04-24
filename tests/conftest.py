import pytest
import logging
from kipoi_enformer.logger import logger
from pathlib import Path


@pytest.fixture(autouse=True)
def setup_logger():
    # Use `-p no:logging -s` in Pycharm's additional arguments to view logs
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)


@pytest.fixture
def chr22_example_files():
    base = Path("assets/example_files")
    return {
        'fasta': base / "seq.fa",
        'gtf': base / "annot.gtf.gz",
        'vcf': base / "vcf" / "promoter_var_1.vcf.gz",
    }
