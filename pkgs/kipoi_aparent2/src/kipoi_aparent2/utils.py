from __future__ import annotations

import pyranges as pr
import pathlib


def gtf_to_pandas(gtf: str | pathlib.Path):
    """
    Read GTF file to pandas DataFrame
    :param gtf: Path to GTF file
    :return:
    """
    return pr.read_gtf(gtf, as_df=True, duplicate_attr=True)