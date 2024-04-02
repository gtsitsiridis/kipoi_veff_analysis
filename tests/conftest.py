import pytest
import logging
from kipoi_enformer.logger import logger


@pytest.fixture(autouse=True)
def setup_logger():
    # Use `-p no:logging -s` in Pycharm's additional arguments to view logs
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
