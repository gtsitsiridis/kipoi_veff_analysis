from __future__ import annotations

import pytest
from kipoi_aparent2.dataloader import ApaDataloader, RefApaDataloader, VCFApaDataloader
from kipoi_aparent2.dataloader.apa_annotation import EnsemblAPAAnnotation
from kipoi_aparent2.aparent2 import Aparent2, Aparent2Veff
import pyarrow.parquet as pq
from kipoi_aparent2.logger import logger
from kipoi_aparent2.constants import AlleleType
from shutil import rmtree
from pathlib import Path
import polars as pl


def run_aparent2(dl: ApaDataloader, aparent2_model_path, output_path, size, batch_size,
                 num_cut_sites=50):
    model = Aparent2(aparent2_model_path)

    model.predict(dl, batch_size=batch_size, filepath=output_path, num_cut_sites=num_cut_sites)
    table = pq.read_table(output_path, partitioning=None)
    logger.info(table.schema)

    if size is not None:
        assert table.shape == (size, 3 + len(dl.pyarrow_metadata_schema.names))
    else:
        assert table.shape[1] == 3 + len(dl.pyarrow_metadata_schema.names)

    assert 'cleavage_prob_full' in table.column_names
    assert 'cleavage_prob_narrow' in table.column_names
    assert 'cleavage_prob_bp' in table.column_names


def get_aparent2_path(output_dir: Path, size: int, allele_type: AlleleType, rm=False):
    if allele_type == AlleleType.REF:
        path = output_dir / f'aparent2_{"full" if not size else size}/raw/ref.parquet/chrom=chr22/data.parquet'
    else:
        path = output_dir / f'aparent2_{"full" if not size else size}/raw/alt.parquet'

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
        'apa_annotation': EnsemblAPAAnnotation(chr22_example_files['gtf'], chromosome='chr22', canonical_only=True,
                                               protein_coding_only=True),
        'size': size,
    }

    aparent2_filepath = get_aparent2_path(output_dir, size, AlleleType.REF, rm=True)
    dl = RefApaDataloader(**args)
    run_aparent2(dl, aparent2_model_path, aparent2_filepath, size, batch_size=batch_size)


@pytest.mark.parametrize("size, batch_size", [
    (3, 1), (5, 3), (10, 5),
    (3, 1), (5, 3), (10, 5),
])
def test_aparent2_alt(chr22_example_files, aparent2_model_path, output_dir: Path, size, batch_size):
    args = {
        'fasta_file': chr22_example_files['fasta'],
        'apa_annotation': EnsemblAPAAnnotation(chr22_example_files['gtf'], chromosome='chr22', canonical_only=True,
                                               protein_coding_only=True),
        'size': size,
        'vcf_file': chr22_example_files['vcf'],
        'variant_downstream_cse': 134,
        'variant_upstream_cse': 70,
        'canonical_only': True,
        'protein_coding_only': True,
    }

    aparent2_filepath = get_aparent2_path(output_dir, size, AlleleType.ALT, rm=True)
    dl = VCFApaDataloader(**args)
    run_aparent2(dl, aparent2_model_path, aparent2_filepath, size, batch_size=batch_size)


@pytest.mark.parametrize("aggregation_mode, upstream_cse, downstream_cse", [
    ('delta_pdui', 70, 134),
    ('lor', 70, 134),
])
def test_calculate_veff(chr22_example_files, output_dir: Path, aparent2_model_path,
                        aggregation_mode, downstream_cse,
                        upstream_cse, size=None, batch_size=5):
    ref_filepath = get_aparent2_path(output_dir, size, AlleleType.REF, rm=True)
    if ref_filepath.exists():
        logger.debug(f'Using existing file: {ref_filepath}')
    else:
        logger.debug(f'Creating file: {ref_filepath}')
        test_aparent2_ref(chr22_example_files, aparent2_model_path, output_dir, batch_size=batch_size, size=size)

    alt_filepath = get_aparent2_path(output_dir, size, AlleleType.ALT, rm=True)
    if alt_filepath.exists():
        logger.debug(f'Using existing file: {alt_filepath}')
    else:
        logger.debug(f'Creating file: {alt_filepath}')
        test_aparent2_alt(chr22_example_files, aparent2_model_path, output_dir, batch_size=batch_size, size=size)

    output_path = output_dir / f'aparent2_{"full" if not size else size}/{aggregation_mode}_{upstream_cse}_{downstream_cse}_veff.parquet'
    if output_path.exists():
        output_path.unlink()
    apa_annotation = EnsemblAPAAnnotation(chr22_example_files['gtf'], chromosome='chr22', canonical_only=True,
                                          protein_coding_only=True,
                                          isoform_usage_path=chr22_example_files['isoform_proportions'])
    enformer_veff = Aparent2Veff(
        apa_annotation=apa_annotation if aggregation_mode == 'delta_pdui' else None,
    )
    enformer_veff.run([ref_filepath], alt_filepath, output_path, aggregation_mode=aggregation_mode,
                      downstream_cse=downstream_cse, upstream_cse=upstream_cse)

    df = pl.read_parquet(output_path)

    assert 'veff_score' in df.columns
    if aggregation_mode == 'delta_pdui':
        assert 'pdui_ref' in df.columns
        assert df.select(['chrom', 'strand', 'gene_id', 'tissue', 'variant_start', 'variant_end', 'ref', 'alt']). \
            unique().shape[0] == df.shape[0]
    else:
        assert df.select(['chrom', 'strand', 'gene_id', 'variant_start', 'variant_end', 'ref', 'alt']). \
            unique().shape[0] == df.shape[0]

    assert df['veff_score'].dtype == pl.Float64
    assert df.filter(pl.col('veff_score').is_null()).shape[0] == 0
    assert df.filter(pl.col('veff_score').is_nan()).shape[0] == 0

    return output_path
