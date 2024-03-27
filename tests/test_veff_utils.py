import pytest

from kipoi_enformer.veff.dataloader import VCF_Enformer_DL, get_tss_from_genome_annotation
from pathlib import Path
import pyranges as pr
import tensorflow as tf
from kipoi_enformer.veff.utils import Enformer

MODEL_PATH = 'https://tfhub.dev/deepmind/enformer/1'


@pytest.fixture
def chr22_example_files():
    base = Path("example_files/chr22")
    return {
        'fasta': base / "seq.chr22.fa",
        'gtf': base / "annot.chr22.gtf",
        'vcf': base / "promoter_variants.chr22.vcf",
    }


def test_enformer(chr22_example_files):
    dl = VCF_Enformer_DL(
        fasta_file=chr22_example_files['fasta'],
        gtf_file=chr22_example_files['gtf'],
        vcf_file=chr22_example_files['vcf'],
        is_onehot=True,
        downstream_tss=10,
        upstream_tss=10,
        seq_length=393_216
    )
    batch_size = 1

    res = [x for x in dl][batch_size]
    input_tensor = tf.convert_to_tensor([x['sequence'] for x in res])
    assert input_tensor.shape == (batch_size, 393_216, 4)
    enformer = Enformer(MODEL_PATH)
    predictions = enformer.predict_on_batch(input_tensor)
    assert predictions['human'].shape == (batch_size, 896, 5313)
    assert predictions['mouse'].shape == (batch_size, 896, 1643)
