from __future__ import annotations

import pytest

from kipoi_aparent2.dataloader import VCFApaDataloader, RefApaDataloader
from kipoi_aparent2.dataloader.apa_annotation import EnsemblAPAAnnotation
from kipoiseq.transforms.functional import one_hot2string
import polars as pl

pl.Config.with_columns_kwargs = True

UPSTREAM_TSS = 10
DOWNSTREAM_TSS = 10


@pytest.fixture
def variants():
    return {
        # Positive strand
        # SNP
        # ===|CSE|===|Var|=======================
        'chr22:17097977:T>C:ENST00000438850.1_2': {
            'chrom': 'chr22',
            'strand': '+',
            'cse': 17097973,  # 0-based
            'ref_start': 17097963,  # 0-based
            'ref_stop': 17097985,  # 1-based
            'var_start': 17097976,  # 0-based
            'var_stop': 17097977,  # 1-based
            'ref': 'T',
            'alt': 'C',
            'ref_seq': 'GTGTTTTAATGTTTTCATCTT',
            'alt_seq': 'GTGTTCTAATGTTTTCATCTT',
        },
        # Positive strand
        # SNP
        # ===|Var|===|CSE|=======================
        'chr22:17119839:G>A:ENST00000585784.1_2': {
            'chrom': 'chr22',
            'strand': '+',
            'cse': 17119839,  # 0-based
            'ref_start': 17119829,  # 0-based
            'ref_stop': 17119851,  # 1-based
            'var_start': 17119838,  # 0-based
            'var_stop': 17119839,  # 1-based
            'ref': 'G',
            'alt': 'A',
            'ref_seq': 'GGAGAATCTTTTGAACCTGGG',
            'alt_seq': 'GAAGAATCTTTTGAACCTGGG',
        },
        # Negative strand
        # SNP
        # ===|Var|===|CSE|=======================
        'chr22:16255375:T>C:ENST00000417657.1': {
            'chrom': 'chr22',
            'strand': '-',
            'cse': 16255384,  # 0-based
            'ref_start': 16255374,  # 0-based
            'ref_stop': 16255396,  # 1-based
            'var_start': 16255374,  # 0-based
            'var_stop': 16255375,  # 1-based
            'ref': 'T',
            'alt': 'C',
            # complement: GTGGACAATGGAGGGGCCTGA
            'ref_seq': 'TCAGGCCCCTCCATTGTCCAC',
            # complement: GTGGACAACGGAGGGGCCTGA
            'alt_seq': 'TCAGGCCCCTCCGTTGTCCAC',
        },
        # Negative strand
        # Deletion
        # ===|Var|===|CSE|=======================
        'chr22:20640916:TCGGCGGCCTCGTTAGCGATGCCGTGGA>T:ENST00000428139.1': {
            'chrom': 'chr22',
            'strand': '-',
            'cse': 20640918,  # 0-based
            'ref_start': 20640908,  # 0-based
            'ref_end': 20640930,  # 1-based
            'var_start': 20640915,  # 0-based
            'var_end': 20640943,  # 1-based
            'ref': 'TCGGCGGCCTCGTTAGCGATGCCGTGGA',
            'alt': 'T',
            # complement: TTGGCGATGCCCTTGTCGGCG
            'ref_seq': 'CGCCGACAAGGGCATCGCCAA',
            # complement: CGTTGGCGATGCCCTTGTCAG
            'alt_seq': 'CTGACAAGGGCATCGCCAACG',
        },
    }


@pytest.fixture()
def references():
    return {
        # Positive strand
        'chr22:ENST00000398242.2': {
            'chrom': 'chr22',
            'strand': '+',
            'cse': 16123737,  # 0-based
            'start': 16123727,  # 0-based
            'end': 16123749,  # 1-based
            'seq': 'TTCCAGTGTGGCTTTGGACTC',
        },
        # Negative strand
        'chr22:ENST00000413156.1': {
            'chrom': 'chr22',
            'strand': '-',
            'cse': 16084278,  # 0-based
            'start': 16084268,  # 0-based
            'end': 16084290,  # 1-based
            # complement: GATGCCTAGCTGAGGGCAAAC
            'seq': 'GTTTGCCCTCAGCTAGGCATC',
        },
    }


@pytest.fixture
def ensemble_apa_anotation(chr22_example_files):
    return EnsemblAPAAnnotation(chr22_example_files['gtf'])


def test_get_cse_from_ensembl(chr22_example_files):
    roi = EnsemblAPAAnnotation(chr22_example_files['gtf'], chromosome='chr22', protein_coding_only=False,
                               canonical_only=False).get_annotation().to_pandas()

    # are the extracted ROIs correct
    # criteria are the strand and transcript start and end
    # roi start is zero based
    # roi end is 1 based
    def test_cse(row):
        # make tss zero-based
        if row.Strand == '-':
            cse = row.pas_pos + 30
        else:
            cse = row.pas_pos - 30

        assert row.Start == cse, \
            f'Transcript {row.transcript_id}; Strand {row.Strand}: {row.Start} != {cse}'
        assert row.End == cse + 1, \
            f'Transcript {row.transcript_id}; Strand {row.Strand}: {row.End} != ({cse} + 1)'

    roi.apply(test_cse, axis=1)

    # check the extracted ROI for a negative strand transcript
    roi_i = roi.set_index('pas_id').loc['chr22:16076051:-']
    assert roi_i.Start == (16076052 + 30 - 1)
    assert roi_i.End == 16076052 + 30

    # check the extracted ROI for a positive strand transcript
    roi_i = roi.set_index('pas_id').loc['chr22:16063235:+']
    assert roi_i.Start == (16063236 - 30 - 1)
    assert roi_i.End == 16063236 - 30


def test_vcf_dataloader(chr22_example_files, variants):
    dl = VCFApaDataloader(
        fasta_file=chr22_example_files['fasta'],
        apa_annotation=EnsemblAPAAnnotation(chr22_example_files['gtf']),
        vcf_file=chr22_example_files['vcf'],
        variant_downstream_cse=10,
        variant_upstream_cse=2,
        seq_length=21,
        cse_pos_index=2,  # In this case the CSE is placed on position 2 of the sequence (zero-based)
    )
    total = 0
    checked_variants = dict()
    for i in dl:
        total += 1
        metadata = i['metadata']
        # example: chr22:16364873:G>A_
        var_id = (f'{metadata["chrom"]}:{metadata["variant_start"] + 1}:'
                  f'{metadata["ref"]}>{metadata["alt"]}:'
                  f'{metadata["transcript_id"]}')
        variant = variants.get(var_id, None)
        if variant is not None:
            assert one_hot2string(i['sequence'][None, :, :])[0] == variant['alt_seq']
            checked_variants[var_id] = 2

    # check that all variants in my list were found and checked
    assert set(checked_variants.keys()) == set(variants.keys())
    print(total)


def test_ref_dataloader(chr22_example_files, references):
    dl = RefApaDataloader(
        fasta_file=chr22_example_files['fasta'],
        apa_annotation=EnsemblAPAAnnotation(chr22_example_files['gtf'], chromosome='chr22'),
        seq_length=21,
        cse_pos_index=2,  # In this case the CSE is placed on position 2 of the sequence (zero-based)
    )
    total = 0
    checked_refs = dict()
    for i in dl:
        total += 1
        metadata = i['metadata']
        # example: chr22:16364873:G>A_
        ref_id = f'chr22:{metadata["transcript_id"]}'
        ref = references.get(ref_id)
        if ref is not None:
            assert one_hot2string(i['sequence'][None, :, :])[0] == ref['seq']
            checked_refs[ref_id] = 2

    # check that all variants in my list were found and checked
    assert set(checked_refs.keys()) == set(references.keys())
    print(total)


def test_ensembl_isoform_usage(chr22_example_files):
    apa_annotation = EnsemblAPAAnnotation(chr22_example_files['gtf'],
                                          isoform_usage_path=chr22_example_files['isoform_proportions'])
    isoform_usage_df = apa_annotation.get_isoform_usage()
    assert 'isoform_proportion' in isoform_usage_df.columns
    assert 'pas_id' in isoform_usage_df.columns
    assert isoform_usage_df.with_columns(
        transcript_id=pl.col('transcript_id').arr.join(';')).unique().shape == isoform_usage_df.shape
    for row in isoform_usage_df.rows(named=True):
        print(row)
        # todo
        # isoform_usage_df.apply(lambda x: x['isoform_proportion'] >= 0 and x['isoform_proportion'] <= 1)
