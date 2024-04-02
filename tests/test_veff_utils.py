import pytest

from kipoi_enformer.veff.dataloader import VCF_Enformer_DL, get_tss_from_genome_annotation
import tensorflow as tf
from kipoi_enformer.veff.utils import Enformer
from pathlib import Path


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

    dl = VCF_Enformer_DL(
        fasta_file=example_files['fasta'],
        gtf_file=example_files['gtf'],
        vcf_file=example_files['vcf'],
        is_onehot=True,
        downstream_tss=500,
        upstream_tss=500,
        shift=43,
        seq_length=393_216,
        size=size
    )

    enformer = Enformer(model=model)
    results = enformer.predict(dl, batch_size=batch_size)
    assert len(results) == size

    output_dir = Path('output/test')
    output_dir.mkdir(exist_ok=True, parents=True)
    Enformer.to_parquet(results, output_dir / 'random_enformer.parquet')


def test_random_enformer(chr22_example_files, random_model):
    run_enformer(chr22_example_files, random_model)
