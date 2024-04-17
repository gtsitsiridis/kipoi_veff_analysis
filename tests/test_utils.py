import pytest

from kipoi_enformer.dataloader import VCFTSSDataloader, TSSDataloader, RefTSSDataloader
import tensorflow as tf
from kipoi_enformer.utils import Enformer, EnformerTissueMapper
from pathlib import Path
import pyarrow.parquet as pq
from kipoi_enformer.logger import logger
import numpy as np
import pickle
import polars as pl


@pytest.fixture
def random_model():
    class RandomModel(tf.keras.Model):
        def predict_on_batch(self, input_tensor):
            return {'human': tf.abs(tf.random.normal((input_tensor.shape[0], 896, 5313))),
                    'mouse': tf.abs(tf.random.normal((input_tensor.shape[0], 896, 1643))),
                    }

    return RandomModel()


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


def run_enformer(dl: TSSDataloader, model, output_path, size, batch_size):
    enformer = Enformer(model=model)

    enformer.predict(dl, batch_size=batch_size, filepath=output_path)
    table = pq.read_table(output_path)
    logger.info(table.schema)

    assert table.shape == (size, 1 + len(dl.pyarrow_metadata_schema.names))

    x = table['tracks'].to_pylist()
    x = np.array(x)
    assert x.shape == (size, 3, 896, 5313)


@pytest.mark.parametrize("size, batch_size", [(3, 1), (5, 3), (10, 5)])
def test_random_vcf_enformer(chr22_example_files, random_model, output_dir: Path, size, batch_size):
    dl = VCFTSSDataloader(
        fasta_file=chr22_example_files['fasta'],
        gtf_file=chr22_example_files['gtf'],
        vcf_file=chr22_example_files['vcf'],
        variant_downstream_tss=500,
        variant_upstream_tss=500,
        shift=43,
        seq_length=393_216,
        size=size
    )
    random_enformer_filepath = output_dir / f'random_vcf_enformer_{size}.parquet'
    run_enformer(dl, random_model, random_enformer_filepath, size, batch_size=batch_size)


@pytest.mark.parametrize("size, batch_size", [(3, 1), (5, 3), (10, 5)])
def test_random_ref_enformer(chr22_example_files, random_model, output_dir: Path, size, batch_size):
    dl = RefTSSDataloader(
        fasta_file=chr22_example_files['fasta'],
        gtf_file=chr22_example_files['gtf'],
        shift=43,
        seq_length=393_216,
        size=size
    )
    random_enformer_filepath = output_dir / f'random_ref_enformer_{size if size is not None else "full"}.parquet'
    run_enformer(dl, random_model, random_enformer_filepath, size, batch_size=batch_size)


def test_estimate_ref_scores(chr22_example_files, random_model, output_dir: Path, enformer_tracks_path: Path,
                             gtex_tissue_matcher_path: Path, size=10, batch_size=5):
    random_enformer_filepath = output_dir / f'random_ref_enformer_{size}.parquet'
    if not random_enformer_filepath.exists():
        logger.debug(f'Creating file: {random_enformer_filepath}')
        run_enformer(chr22_example_files, random_model, random_enformer_filepath, size=size, batch_size=batch_size)
    else:
        logger.debug(f'Using existing file: {random_enformer_filepath}')

    tissue_mapper = EnformerTissueMapper(tracks_path=enformer_tracks_path,
                                         tissue_matcher_path=gtex_tissue_matcher_path)

    random_tissue_filepath = output_dir / f'random_ref_tissue_{size}.parquet'
    tissue_mapper.predict(random_enformer_filepath, output_path=random_tissue_filepath, batch_size=batch_size)

    with open(gtex_tissue_matcher_path, 'rb') as f:
        num_tissues = len(pickle.load(f))

    tbl = pl.read_parquet(random_tissue_filepath)
    assert tbl.shape == (size * num_tissues, 10 + 2)
