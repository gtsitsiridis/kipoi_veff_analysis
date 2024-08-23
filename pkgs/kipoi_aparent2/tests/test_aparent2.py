from __future__ import annotations

import pytest
from kipoi_aparent2.dataloader import CSEDataloader, RefCSEDataloader, VCFCSEDataloader
from kipoi_aparent2.aparent2 import Aparent2
import pyarrow.parquet as pq
from kipoi_aparent2.logger import logger
from kipoi_aparent2.constants import AlleleType
from shutil import rmtree
from pathlib import Path
import numpy as np


def run_aparent2(dl: CSEDataloader, aparent2_model_path, output_path, size, batch_size):
    model = Aparent2(aparent2_model_path)

    model.predict(dl, batch_size=batch_size, filepath=output_path)
    table = pq.read_table(output_path, partitioning=None)
    logger.info(table.schema)

    assert table.shape == (size, 1 + len(dl.pyarrow_metadata_schema.names))

    x = table['tracks'].to_pylist()
    x = np.array(x)
    assert x.shape == (size, 1, 206)


def get_aparent2_path(output_dir: Path, size: int, allele_type: AlleleType, rm=False):
    if allele_type == AlleleType.REF:
        path = output_dir / f'aparent2_{size}/raw/ref.parquet/chrom=chr22/data.parquet'
    else:
        path = output_dir / f'aparent2_{size}/raw/alt.parquet'

    if rm and path.exists():
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("size, batch_size", [
    (3, 1), (5, 3), (10, 5),
    (3, 1), (5, 3), (10, 5),
])
def test_aparent2_ref(chr22_example_files, aparent2_model_path, output_dir: Path, size, batch_size):
    args = {
        'fasta_file': chr22_example_files['fasta'],
        'gtf': chr22_example_files['gtf'],
        'size': size,
        'chromosome': 'chr22',
        'canonical_only': True,
        'protein_coding_only': True,
    }

    aparent2_filepath = get_aparent2_path(output_dir, size, AlleleType.REF, rm=True)
    dl = RefCSEDataloader(**args)
    run_aparent2(dl, aparent2_model_path, aparent2_filepath, size, batch_size=batch_size)


@pytest.mark.parametrize("size, batch_size", [
    (3, 1), (5, 3), (10, 5),
    (3, 1), (5, 3), (10, 5),
])
def test_aparent2_alt(chr22_example_files, aparent2_model_path, output_dir: Path, size, batch_size):
    args = {
        'fasta_file': chr22_example_files['fasta'],
        'gtf': chr22_example_files['gtf'],
        'size': size,
        'vcf_file': chr22_example_files['vcf'],
        'variant_downstream_cse': 500,
        'variant_upstream_cse': 500,
        'canonical_only': True,
        'protein_coding_only': True,
    }

    aparent2_filepath = get_aparent2_path(output_dir, size, AlleleType.ALT, rm=True)
    dl = VCFCSEDataloader(**args)
    run_aparent2(dl, aparent2_model_path, aparent2_filepath, size, batch_size=batch_size)
