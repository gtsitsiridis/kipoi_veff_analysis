import pytest

from kipoi_enformer.veff.dataloader import VCF_Enformer_DL, get_roi_from_genome_annotation
from pathlib import Path
import pyranges as pr
import traceback

UPSTREAM_TSS = 10
DOWNSTREAM_TSS = 10


@pytest.fixture
def chr22_example_files():
    base = Path("example_files/chr22")
    return {
        'fasta': base / "seq.chr22.fa",
        'gtf': base / "annot.chr22.gtf",
        'vcf': base / "promoter_variants.chr22.vcf",
    }


@pytest.fixture
def variants():
    return {
        'chr22:17702426:C>T': {
            'chrom': 'chr22',
            'start': 17702425,
            'end': 17702426,
            'ref': 'C',
            'alt': 'T',
        },
    }


def test_get_roi_from_genome_annotation(chr22_example_files):
    upstream_tss = 10
    downstream_tss = 10

    genome_annotation = pr.read_gtf(chr22_example_files['gtf'], as_df=True)
    roi = get_roi_from_genome_annotation(genome_annotation, upstream_tss=upstream_tss, downstream_tss=downstream_tss)

    # check the number of transcripts
    # grep -v '^#' annot.chr22.gtf | cut -f 3 | grep transcript | wc -l
    assert len(roi) == 4511

    # are the extracted ROIs correct
    # criteria are the strand and transcript start and end
    # roi start is zero based
    # roi end is 1 based
    def test_roi(row):
        # make tss zero-based
        if row.Strand == '-':
            tss = row.transcript_end - 1
        else:
            tss = row.transcript_start

        assert row.Start == tss - upstream_tss, \
            f'Transcript {row.transcript_id}; Strand {row.Strand}: {row.Start} != ({tss} - {upstream_tss})'
        assert row.End == tss + downstream_tss + 1, \
            f'Transcript {row.transcript_id}; Strand {row.Strand}: {row.End} != ({tss} + {downstream_tss + 1})'

    roi.apply(test_roi, axis=1)

    # check the extracted ROI for a negative strand transcript
    roi_i = roi.set_index('transcript_id').loc['ENST00000615943.1']
    assert roi_i.Start == (10736283 - (upstream_tss + 1))
    assert roi_i.End == (10736283 + downstream_tss)

    # check the extracted ROI for a positive strand transcript
    roi_i = roi.set_index('transcript_id').loc['ENST00000624155.1']
    assert roi_i.Start == (11066501 - (upstream_tss + 1))
    assert roi_i.End == (11066501 + downstream_tss)


def test_dataloader(chr22_example_files):
    dl = VCF_Enformer_DL(
        fasta_file=chr22_example_files['fasta'],
        gtf_file=chr22_example_files['gtf'],
        vcf_file=chr22_example_files['vcf'],
    )
    total = 0
    for i in dl:
        total += 1
        print(i)
