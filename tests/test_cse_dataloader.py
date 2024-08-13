import pytest

from kipoi_veff_analysis.dataloader import VCFCSEDataloader, RefCSEDataloader
from kipoi_veff_analysis.dataloader.common import get_cse_from_genome_annotation
from kipoiseq.transforms.functional import one_hot2string

UPSTREAM_TSS = 10
DOWNSTREAM_TSS = 10


# todo add the the CSEs to the variants
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


# todo add the the CSEs to the reference
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


def test_get_cse_from_genome_annotation(chr22_example_files):
    roi = get_cse_from_genome_annotation(chr22_example_files['gtf'], chromosome='chr22', protein_coding_only=False,
                                         canonical_only=False)

    # check the number of transcripts
    # grep -v '^#' annot.chr22.gtf | cut -f 3 | grep transcript | wc -l
    assert len(roi) == 5279

    # are the extracted ROIs correct
    # criteria are the strand and transcript start and end
    # roi start is zero based
    # roi end is 1 based
    def test_cse(row):
        # make tss zero-based
        if row.Strand == '-':
            cse = row.transcript_start + 30
        else:
            cse = row.transcript_end - 1 - 30

        assert row.Start == cse, \
            f'Transcript {row.transcript_id}; Strand {row.Strand}: {row.Start} != {cse}'
        assert row.End == cse + 1, \
            f'Transcript {row.transcript_id}; Strand {row.Strand}: {row.End} != ({cse} + 1)'

    roi.apply(test_cse, axis=1)

    # check the extracted ROI for a negative strand transcript
    roi_i = roi.set_index('transcript_id').loc['ENST00000448070.1']
    assert roi_i.Start == (16076052 + 30 - 1)
    assert roi_i.End == 16076052 + 30

    # check the extracted ROI for a positive strand transcript
    roi_i = roi.set_index('transcript_id').loc['ENST00000424770.1']
    assert roi_i.Start == (16063236 - 30 - 1)
    assert roi_i.End == 16063236 - 30


def test_genome_annotation_protein_canonical(chr22_example_files):
    # Ground truth bash:
    # grep -E '^\S+\s+\S+\s+transcript' annot.chr22.gtf | grep -E 'tag\s+"Ensembl_canonical"' |
    # grep -E 'transcript_type\s+"protein_coding"' | wc -l
    chromosome = 'chr22'
    # not all genes have a canonical transcript if the genome annotation is not current (e.g. GRCh37 is not current)
    roi = get_cse_from_genome_annotation(chr22_example_files['gtf'], chromosome=chromosome, protein_coding_only=True,
                                         canonical_only=True)
    assert len(roi) == 441
    roi = get_cse_from_genome_annotation(chr22_example_files['gtf'], chromosome=chromosome, protein_coding_only=True,
                                         canonical_only=True, gene_ids=['ENSG00000172967'])
    assert roi.gene_id.iloc[0] == 'ENSG00000172967.8_5'
    roi = get_cse_from_genome_annotation(chr22_example_files['gtf'], chromosome=chromosome, protein_coding_only=False,
                                         canonical_only=False)
    assert len(roi) == 5279
    roi = get_cse_from_genome_annotation(chr22_example_files['gtf'], chromosome=chromosome, protein_coding_only=True,
                                         canonical_only=False)
    assert len(roi) == 3623
    roi = get_cse_from_genome_annotation(chr22_example_files['gtf'], chromosome=chromosome, protein_coding_only=False,
                                         canonical_only=True)
    assert len(roi) == 1212


def test_vcf_dataloader(chr22_example_files, variants):
    dl = VCFCSEDataloader(
        fasta_file=chr22_example_files['fasta'],
        gtf=chr22_example_files['gtf'],
        vcf_file=chr22_example_files['vcf'],
        variant_downstream_tss=10,
        variant_upstream_tss=10,
        seq_length=21,
        shifts=[8],  # 21 // 2 - 2; In this case the CSE is placed on position 2 of the sequence
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
            assert one_hot2string(i['sequences'])[0] == variant['alt_seq']
            checked_variants[var_id] = 2

    # check that all variants in my list were found and checked
    assert set(checked_variants.keys()) == set(variants.keys())
    print(total)


def test_ref_dataloader(chr22_example_files, references):
    dl = RefCSEDataloader(
        fasta_file=chr22_example_files['fasta'],
        gtf=chr22_example_files['gtf'],
        seq_length=21,
        shifts=[8],
        chromosome='chr22',
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
            assert one_hot2string(i['sequences'])[0] == ref['seq']
            checked_refs[ref_id] = 2

    # check that all variants in my list were found and checked
    assert set(checked_refs.keys()) == set(references.keys())
    print(total)
