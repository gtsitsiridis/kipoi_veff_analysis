import pytest
import logging
from kipoi_aparent2.logger import logger
from pathlib import Path


@pytest.fixture(autouse=True)
def setup_logger():
    # Use `-p no:logging -s` in Pycharm's additional arguments to view logs
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(logging.DEBUG)


@pytest.fixture
def output_dir():
    output_dir = Path('../../../output/test/')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def aparent2_model_path():
    return Path('../../../assets/aparent_all_libs_resnet_no_clinvar_wt_ep_5_var_batch_size_inference_mode_no_drop.h5')

@pytest.fixture
def chr22_example_files():
    base = Path("../../../example_files")
    return {
        'fasta': base / "seq.fa",
        'gtf': base / "annot.gtf.gz",
        'vcf': base / "vcf" / "chr22_var.vcf.gz",
        'isoform_proportions': base / "isoform_proportions.tsv",
        'gtex_expression': base / 'gtex_samples/transcripts_tpms.zarr',
        'gtex_variants': base / 'gtex_samples/rare_variants.vcf.parquet',
        'gtex_annotation': base / 'gtex_samples/benchmark_with_annotation.parquet',
        'gtex_folds': base / 'gtex_samples/folds.parquet',
    }
