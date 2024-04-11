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
    base = Path("assets/example_files/chr22")
    return {
        'fasta': base / "seq.chr22.fa",
        'gtf': base / "annot.chr22.gtf",
        'vcf': base / "promoter_variants.chr22.vcf",
    }
