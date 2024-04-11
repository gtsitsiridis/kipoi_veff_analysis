import pytest

from kipoi_enformer.dataloader import VCFEnformerDL
import tensorflow as tf
from kipoi_enformer.utils import Enformer, estimate_veff
from pathlib import Path
import pyarrow.parquet as pq
from kipoi_enformer.logger import logger
import numpy as np
import pickle
import yaml


@pytest.fixture
def random_model():
    class RandomModel(tf.keras.Model):
        def predict_on_batch(self, input_tensor):
            return {'human': tf.abs(tf.random.normal((input_tensor.shape[0], 896, 5313))),
                    'mouse': tf.abs(tf.random.normal((input_tensor.shape[0], 896, 1643))),
                    }

    return RandomModel()


@pytest.fixture
def random_enformer_filepath():
    output_dir = Path('output/test')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / 'random_enformer.parquet'


@pytest.fixture
def random_veff_filepath():
    output_dir = Path('output/test')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / 'random_veff.parquet'


@pytest.fixture
def gtex_tissue_matcher_lm_dict():
    filepath = Path('assets/gtex_enformer_lm_models_pseudocount1.pkl')
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return {k: v['ingenome'] for k, v in data.items()}


@pytest.fixture
def enformer_tracks_dict():
    filepath = Path('assets/enformer_tracks.yaml')
    with open(filepath, 'rb') as f:
        return yaml.safe_load(f)


def run_enformer(example_files, model, output_path):
    batch_size = 3
    size = 3

    dl = VCFEnformerDL(
        fasta_file=example_files['fasta'],
        gtf_file=example_files['gtf'],
        vcf_file=example_files['vcf'],
        variant_downstream_tss=500,
        variant_upstream_tss=500,
        shift=43,
        seq_length=393_216,
        size=size
    )

    enformer = Enformer(model=model)

    enformer.predict(dl, batch_size=batch_size, filepath=output_path)
    table = pq.read_table(output_path)
    logger.info(table.schema)

    assert table.shape == (size, 6 + 13)

    x = table['ref_0'].to_pylist()
    x = np.array(x)
    assert x.shape == (size, 896, 5313)


def test_random_enformer(chr22_example_files, random_model, random_enformer_filepath: Path):
    run_enformer(chr22_example_files, random_model, random_enformer_filepath)


def test_estimate_veff(chr22_example_files, random_model, random_enformer_filepath: Path,
                       gtex_tissue_matcher_lm_dict: dict, random_veff_filepath: Path, enformer_tracks_dict: dict):
    if not random_enformer_filepath.exists():
        run_enformer(chr22_example_files, random_model, random_enformer_filepath)

    estimate_veff(random_enformer_filepath, tissue_matcher_lm_dict=gtex_tissue_matcher_lm_dict,
                  output_path=random_veff_filepath, enformer_tracks_dict=enformer_tracks_dict)
