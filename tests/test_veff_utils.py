import pytest

from kipoi_enformer.dataloader import VCFEnformerDL, get_tss_from_genome_annotation
import tensorflow as tf
from kipoi_enformer.utils import Enformer
from pathlib import Path
import pyarrow.parquet as pq
from kipoi_enformer.logger import logger
import numpy as np


@pytest.fixture
def chr22_example_files():
    base = Path("example_files/chr22")
    return {
        'fasta': base / "seq.chr22.fa",
        'gtf': base / "annot.chr22.gtf",
        'vcf': base / "promoter_variants.chr22.vcf",
    }


@pytest.fixture
def random_model():
    class RandomModel(tf.keras.Model):
        def predict_on_batch(self, input_tensor):
            return {'human': tf.random.normal((input_tensor.shape[0], 896, 5313)),
                    'mouse': tf.random.normal((input_tensor.shape[0], 896, 1643)),
                    }

    return RandomModel()


def run_enformer(example_files, model):
    batch_size = 3
    size = 10

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
    output_dir = Path('output/test')
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / 'random_enformer.parquet'
    enformer.predict(dl, batch_size=batch_size, filepath=filepath)

    table = pq.read_table(filepath)
    logger.info(table.schema)

    assert table.shape == (size, 6 + 13)

    x = table['ref_0'].to_pylist()
    x = np.array(x)
    assert x.shape == (size, 896, 5313)


def test_random_enformer(chr22_example_files, random_model):
    run_enformer(chr22_example_files, random_model)
