import pytest

from kipoi_enformer.dataloader import TSSDataloader, RefTSSDataloader, VCFTSSDataloader
from kipoi_enformer.enformer import Enformer, EnformerAggregator, EnformerTissueMapper, calculate_veff, aggregate_veff
from pathlib import Path
import pyarrow.parquet as pq
from kipoi_enformer.logger import logger
import numpy as np
import pickle
import polars as pl
from kipoi_enformer.constants import AlleleType
from shutil import rmtree
import tempfile


@pytest.fixture
def output_dir():
    output_dir = Path('output/test')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def gtex_tissue_mapper_path():
    return Path('assets/gtex_enformer_lm_models_pseudocount1.pkl')


@pytest.fixture
def enformer_tracks_path():
    return Path('assets/cage_nonuniversal_enformer_tracks.yaml')


def run_enformer(dl: TSSDataloader, output_path, size, batch_size, num_output_bins):
    enformer = Enformer(is_random=True)

    enformer.predict(dl, batch_size=batch_size, filepath=output_path, num_output_bins=num_output_bins)
    table = pq.read_table(output_path)
    logger.info(table.schema)

    assert table.shape == (size, 1 + len(dl.pyarrow_metadata_schema.names))

    x = table['tracks'].to_pylist()
    x = np.array(x)
    assert x.shape == (size, 3, num_output_bins, 5313)


def get_enformer_path(output_dir: Path, size: int, allele_type: AlleleType, rm=False):
    if allele_type == AlleleType.REF:
        path = output_dir / f'enformer_{size}/raw/ref.parquet/chrom=chr22/data.parquet'
    else:
        path = output_dir / f'enformer_{size}/raw/alt.parquet'

    if rm and path.exists():
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_tissue_path(output_dir: Path, size: int, allele_type: AlleleType, rm=False):
    if allele_type == AlleleType.REF:
        path = output_dir / f'enformer_{size}/tissue/ref.parquet/chrom=chr22/data.parquet'
    else:
        path = output_dir / f'enformer_{size}/tissue/alt.parquet'

    if rm and path.exists():
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_veff_path(output_dir: Path, size: int, rm=False):
    path = output_dir / f'enformer_{size}/tissue/veff.parquet'
    if rm and path.exists():
        if path.is_dir():
            rmtree(path)
        else:
            path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("size, batch_size, num_output_bins", [
    (3, 1, 896), (5, 3, 896), (10, 5, 896),
    (3, 1, 21), (5, 3, 21), (10, 5, 21),
])
def test_enformer_ref(chr22_example_files, output_dir: Path, size, batch_size, num_output_bins):
    args = {
        'fasta_file': chr22_example_files['fasta'],
        'gtf': chr22_example_files['gtf'],
        'shift': 43,
        'seq_length': 393_216,
        'size': size,
        'chromosome': 'chr22'
    }

    enformer_filepath = get_enformer_path(output_dir, size, AlleleType.REF, rm=True)
    dl = RefTSSDataloader(**args)
    run_enformer(dl, enformer_filepath, size, batch_size=batch_size, num_output_bins=num_output_bins)


@pytest.mark.parametrize("size, batch_size, num_output_bins", [
    (3, 1, 896), (5, 3, 896), (10, 5, 896),
    (3, 1, 21), (5, 3, 21), (10, 5, 21),
])
def test_enformer_alt(chr22_example_files, output_dir: Path, size, batch_size, num_output_bins):
    args = {
        'fasta_file': chr22_example_files['fasta'],
        'gtf': chr22_example_files['gtf'],
        'shift': 43,
        'seq_length': 393_216,
        'size': size,
        'vcf_file': chr22_example_files['vcf'],
        'variant_downstream_tss': 500,
        'variant_upstream_tss': 500,
    }

    enformer_filepath = get_enformer_path(output_dir, size, AlleleType.ALT, rm=True)
    dl = VCFTSSDataloader(**args)
    run_enformer(dl, enformer_filepath, size, batch_size=batch_size, num_output_bins=num_output_bins)


@pytest.mark.parametrize("allele_type", [
    'REF', 'ALT'
])
def test_predict_tissue_mapper(allele_type: str, chr22_example_files, output_dir: Path,
                               enformer_tracks_path: Path, gtex_tissue_mapper_path: Path, size=10, batch_size=5,
                               num_output_bins=21):
    enformer_filepath = get_enformer_path(output_dir, size, AlleleType[allele_type])
    if not enformer_filepath.exists():
        logger.debug(f'Creating file: {enformer_filepath}')
        if allele_type == 'REF':
            test_enformer_ref(chr22_example_files, output_dir, size, batch_size, num_output_bins)
        elif allele_type == 'ALT':
            test_enformer_alt(chr22_example_files, output_dir, size, batch_size, num_output_bins)
    else:
        logger.debug(f'Using existing file: {enformer_filepath}')

    enformer_aggregator = EnformerAggregator()
    agg_path = output_dir / f'enformer_{size}/tmp_aggregated.parquet'
    enformer_aggregator.aggregate(enformer_filepath, agg_path)

    tissue_mapper = EnformerTissueMapper(tracks_path=enformer_tracks_path,
                                         tissue_mapper_path=gtex_tissue_mapper_path)
    enformer_tissue_filepath = get_tissue_path(output_dir, size, AlleleType[allele_type])
    tissue_mapper.predict(agg_path, output_path=enformer_tissue_filepath)

    with open(gtex_tissue_mapper_path, 'rb') as f:
        num_tissues = len(pickle.load(f))

    tbl = pl.read_parquet(enformer_tissue_filepath)

    if allele_type == 'REF':
        assert tbl.shape == (num_tissues * size, 9 + 2)
    elif allele_type == 'ALT':
        assert tbl.shape == (num_tissues * size, 13 + 2)


def test_calculate_veff(chr22_example_files, output_dir: Path,
                        enformer_tracks_path: Path, gtex_tissue_mapper_path: Path, size=10):
    ref_filepath = get_tissue_path(output_dir, size, AlleleType.REF)
    if ref_filepath.exists():
        logger.debug(f'Using existing file: {ref_filepath}')
    else:
        logger.debug(f'Creating file: {ref_filepath}')
        test_predict_tissue_mapper('REF', chr22_example_files, output_dir,
                                   enformer_tracks_path, gtex_tissue_mapper_path, size=size)

    alt_filepath = get_tissue_path(output_dir, size, AlleleType.ALT)
    if alt_filepath.exists():
        logger.debug(f'Using existing file: {alt_filepath}')
    else:
        logger.debug(f'Creating file: {alt_filepath}')
        test_predict_tissue_mapper('ALT', chr22_example_files, output_dir,
                                   enformer_tracks_path, gtex_tissue_mapper_path, size=size)

    output_path = output_dir / f'enformer_{size}/tissue/veff.parquet'
    if output_path.exists():
        output_path.unlink()
    calculate_veff(ref_filepath, alt_filepath, output_path)


@pytest.mark.parametrize("is_isoform", [
    True, False
])
def test_calculate_agg_veff(chr22_example_files, output_dir: Path,
                            enformer_tracks_path: Path, gtex_tissue_mapper_path: Path, is_isoform, size=100):
    veff_filepath = get_veff_path(output_dir, size)
    if veff_filepath.exists():
        logger.debug(f'Using existing file: {veff_filepath}')
    else:
        logger.debug(f'Creating file: {veff_filepath}')
        test_calculate_veff(chr22_example_files, output_dir, enformer_tracks_path, gtex_tissue_mapper_path, size=size)

    output_path = output_dir / f'enformer_{size}/tissue/veff_agg.parquet'
    if output_path.exists():
        output_path.unlink()
    aggregate_veff(veff_path=veff_filepath, output_path=output_path,
                   isoforms_path=None if not is_isoform else chr22_example_files['isoform_proportions'])


def test_train_tissue_mapper(chr22_example_files, gtex_tissue_mapper_path, enformer_tracks_path, output_dir, size=10,
                             batch_size=5, num_output_bins=21):
    enformer_filepath = get_enformer_path(output_dir, size, AlleleType.REF)
    if not enformer_filepath.exists():
        logger.debug(f'Creating file: {enformer_filepath}')
        test_enformer_ref(chr22_example_files, output_dir, size, batch_size, num_output_bins)
    else:
        logger.debug(f'Using existing file: {enformer_filepath}')

    enformer_aggregator = EnformerAggregator()
    agg_path = output_dir / f'enformer_{size}/tmp_aggregated.parquet'
    enformer_aggregator.aggregate(enformer_filepath, agg_path)

    tissue_mapper = EnformerTissueMapper(tracks_path=enformer_tracks_path,
                                         tissue_mapper_path=gtex_tissue_mapper_path)
    tissue_mapper.train(agg_path, output_path=output_dir / 'tissue_mapper',
                        expression_path=chr22_example_files['gtex_expression'])
