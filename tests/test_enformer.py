import pytest

from kipoi_enformer.dataloader import TSSDataloader
import tensorflow as tf
from kipoi_enformer.enformer import Enformer, EnformerTissueMapper, calculate_veff
from pathlib import Path
import pyarrow.parquet as pq
from kipoi_enformer.logger import logger
import numpy as np
import pickle
import polars as pl
from kipoi_enformer.constants import AlleleType
from shutil import rmtree


@pytest.fixture
def output_dir():
    output_dir = Path('output/test')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def gtex_tissue_matcher_path():
    return Path('assets/gtex_enformer_lm_models_pseudocount1.pkl')


@pytest.fixture
def enformer_tracks_path():
    return Path('assets/cage_nonuniversal_enformer_tracks.yaml')


def run_enformer(dl: TSSDataloader, output_path, size, batch_size):
    enformer = Enformer(is_random=True)

    enformer.predict(dl, batch_size=batch_size, filepath=output_path)
    table = pq.read_table(output_path)
    logger.info(table.schema)

    assert table.shape == (size, 1 + len(dl.pyarrow_metadata_schema.names))

    x = table['tracks'].to_pylist()
    x = np.array(x)
    assert x.shape == (size, 3, 896, 5313)


@pytest.mark.parametrize("size, batch_size, allele_type", [
    (3, 1, 'REF'), (5, 3, 'REF'), (10, 5, 'REF'),
    (3, 1, 'ALT'), (5, 3, 'ALT'), (10, 5, 'ALT')
])
def test_enformer(chr22_example_files, output_dir: Path, size, batch_size, allele_type: str):
    args = {
        'fasta_file': chr22_example_files['fasta'],
        'gtf': chr22_example_files['gtf'],
        'shift': 43,
        'seq_length': 393_216,
        'size': size
    }

    if allele_type == 'ALT':
        args.update({'vcf_file': chr22_example_files['vcf'],
                     'variant_downstream_tss': 500,
                     'variant_upstream_tss': 500, })

    dl = TSSDataloader.from_allele_type(AlleleType[allele_type], **args)
    enformer_filepath = output_dir / f'enformer_{size}_raw_{allele_type.lower()}.parquet'
    if enformer_filepath.exists():
        rmtree(enformer_filepath)
    run_enformer(dl, enformer_filepath, size, batch_size=batch_size)


@pytest.mark.parametrize("allele_type", [
    'REF', 'ALT'
])
def test_enformer_tissue_mapper(allele_type: str, chr22_example_files, output_dir: Path,
                                enformer_tracks_path: Path, gtex_tissue_matcher_path: Path, size=10, batch_size=5):
    enformer_filepath = output_dir / f'enformer_{size}_raw_{allele_type.lower()}.parquet'
    if not enformer_filepath.exists():
        logger.debug(f'Creating file: {enformer_filepath}')
        test_enformer(chr22_example_files, output_dir, size, batch_size, allele_type)
    else:
        logger.debug(f'Using existing file: {enformer_filepath}')

    tissue_mapper = EnformerTissueMapper(tracks_path=enformer_tracks_path,
                                         tissue_matcher_path=gtex_tissue_matcher_path)

    enformer_tissue_filepath = output_dir / f'enformer_{size}_tissue_{allele_type.lower()}.parquet'
    if enformer_tissue_filepath.exists():
        rmtree(enformer_tissue_filepath)
    logger.debug(f'Creating file: {enformer_tissue_filepath}')
    tissue_mapper.predict(enformer_filepath, output_path=enformer_tissue_filepath)

    with open(gtex_tissue_matcher_path, 'rb') as f:
        num_tissues = len(pickle.load(f))

    tbl = pl.DataFrame(pq.ParquetDataset(enformer_tissue_filepath).read())
    if allele_type == 'REF':
        assert tbl.shape == (num_tissues * size, 9 + 2)
    elif allele_type == 'ALT':
        assert tbl.shape == (num_tissues * size, 13 + 2)


def test_calculate_veff(chr22_example_files, output_dir: Path,
                        enformer_tracks_path: Path, gtex_tissue_matcher_path: Path, size=10):
    ref_filepath = output_dir / f'enformer_{size}_tissue_ref.parquet'
    if ref_filepath.exists():
        logger.debug(f'Using existing file: {ref_filepath}')
    else:
        logger.debug(f'Creating file: {ref_filepath}')
        test_enformer_tissue_mapper('REF', chr22_example_files, output_dir,
                                    enformer_tracks_path, gtex_tissue_matcher_path, size=size)

    alt_filepath = output_dir / f'enformer_{size}_tissue_alt.parquet'
    if alt_filepath.exists():
        logger.debug(f'Using existing file: {alt_filepath}')
    else:
        logger.debug(f'Creating file: {alt_filepath}')
        test_enformer_tissue_mapper('ALT', chr22_example_files, output_dir,
                                    enformer_tracks_path, gtex_tissue_matcher_path, size=size)
    output_path = output_dir / f'enformer_{size}_veff.parquet'
    if output_path.exists():
        rmtree(output_path)
    calculate_veff(ref_filepath, alt_filepath, output_path)
