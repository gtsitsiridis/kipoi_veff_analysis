import logging

__all__ = ['logger']

logger = logging.getLogger('kipoi_enformer')


def setup_logger(level=logging.INFO):
    logging.basicConfig()
    logger.setLevel(level)
    return logger
