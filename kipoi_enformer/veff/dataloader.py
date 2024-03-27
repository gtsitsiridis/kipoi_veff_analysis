from kipoi.data import SampleIterator
from kipoiseq import Interval, Variant
from kipoiseq.transforms import OneHot
from kipoiseq.extractors import VariantSeqExtractor, SingleVariantMatcher, BaseExtractor, FastaStringExtractor
from kipoi.metadata import GenomicRanges
import math
import pandas as pd
import pyranges as pr

from kipoiseq.variant_source import VariantFetcher

# length of sequence which enformer gets as input
# ═════┆═════┆════════════════════════┆═════┆═════
SEQUENCE_LENGTH = 393_216
# length of central sequence which enformer actually sees (1536 bins)
# ─────┆═════┆════════════════════════┆═════┆─────
SEEN_SEQUENCE_LENGTH = 1_536 * 128
# length of central sequence for which enformer gives predictions (896 bins)
# ─────┆─────┆════════════════════════┆─────┆─────
PRED_SEQUENCE_LENGTH = 896 * 128

# padding (only one side!) until the PRED_SEQUENCE_LENGTH window
# ═════┆═════┆────────────────────────┆═════┆═════
PADDING = (SEQUENCE_LENGTH - PRED_SEQUENCE_LENGTH) // 2
# padding (only one side!) until the SEEN_SEQUENCE_LENGTH window
# ═════┆─────┆────────────────────────┆─────┆═════
PADDING_UNTIL_SEEN = (SEQUENCE_LENGTH - SEEN_SEQUENCE_LENGTH) // 2
# padding (only one side!) from PADDING_UNTIL_SEEN to PRED_SEQUENCE_LENGTH
# ─────┆═════┆────────────────────────┆═════┆─────
PADDING_SEEN = PADDING - PADDING_UNTIL_SEEN

assert 2 * (PADDING_UNTIL_SEEN + PADDING_SEEN) + PRED_SEQUENCE_LENGTH == SEQUENCE_LENGTH, \
    "All parts should add up to SEQUENCE_LENGTH"
assert PADDING_UNTIL_SEEN + PADDING_SEEN == PADDING, \
    "All padding parts should add up to PADDING"
assert PRED_SEQUENCE_LENGTH + 2 * (PADDING_SEEN) == SEEN_SEQUENCE_LENGTH, \
    "All SEEN_SEQUENCE parts should add up to SEEN_SEQUENCE_LENGTH"


class Enformer_DL(SampleIterator):
    def __init__(
            self,
            roi_regions: pr.PyRanges,
            reference_sequence: BaseExtractor,
            variants: VariantFetcher,
            seq_length: int = SEQUENCE_LENGTH,
        is_onehot: bool = True,
    ):
        interval_attrs = ['gene_id', 'transcript_id', 'landmark', 'transcript_start', 'transcript_end']
        for attr in interval_attrs:
            assert attr in roi_regions.columns, f"attr must be in {roi_regions.columns}"
        self.seq_length = seq_length
        self.roi_regions = roi_regions
        self.reference_sequence = reference_sequence
        self.variants = variants

        if not self.reference_sequence.use_strand:
            raise ValueError(
                "Reference sequence fetcher does not use strand but this is needed to obtain correct sequences!")
        self.variant_seq_extractor = VariantSeqExtractor(reference_sequence=reference_sequence)

        self.matcher = SingleVariantMatcher(
            variant_fetcher=self.variants,
            pranges=self.roi_regions,
            interval_attrs=interval_attrs
        )

        self.one_hot = None
        if is_onehot:
            self.one_hot = OneHot()

    def _extract_seq(self, interval: Interval, variant: Variant, allele: str, shift: int = 0):
        five_end_len = math.ceil(self.seq_length / 2) + shift
        three_end_len = math.floor(self.seq_length / 2) - shift

        interval = Interval(chrom=interval.chrom,
                            start=interval.attrs['landmark'] - five_end_len,
                            end=interval.attrs['landmark'] + three_end_len)

        assert allele in ['ref', 'alt'], f"allele must be one of ['ref', 'alt'] but got {allele}"
        if allele == 'ref':
            seq = self.reference_sequence.extract(interval)
        else:
            seq = self.variant_seq_extractor.extract(
                interval,
                [variant],
                anchor=interval.start + five_end_len if not interval.neg_strand
                else interval.end - three_end_len
            )

        if self.one_hot is not None:
            return self.one_hot(seq)
        else:
            return seq

    def __iter__(self):
        # todo also export the min_bin, max_bin, and tss_bin
        # todo nested parquet
        # todo polars
        interval: Interval
        variant: Variant
        for interval, variant in self.matcher:
            for allele in ['alt', 'ref']:
                for shift in [-43, 0, 43]:
                    yield {
                        "sequence": self._extract_seq(interval, variant, allele, shift),
                        "metadata": {
                            "allele": allele,
                            "shift": shift,
                            "landmark_pos": interval.attrs['landmark'],
                            "variant": {
                                "chrom": variant.chrom,
                                "start": variant.start,
                                "end": variant.end,
                                "ref": variant.ref,
                                "alt": variant.alt,
                                "id": variant.id,
                                "str": str(variant),
                            },
                            "transcript": {
                                "chr": interval.chrom,
                                "start": interval.attrs['transcript_start'],
                                "stop": interval.attrs['transcript_end'],
                                "strand": interval.strand,
                                "transcript_id": interval.attrs['transcript_id'],
                                "gene_id": interval.attrs['gene_id'],
                            },
                            # todo implement this
                            "enformer_input_region": None,
                        }
                    }


def get_tss_from_transcript(transcript_start: int, transcript_end: int, is_on_negative_strand: bool) -> (int, int):
    """
    Get region-of-interest for Enformer in relation to the TSS of a transcript
    :param transcript_start: 0-based start position of the transcript
    :param transcript_end: 1-based end position of the transcript
    :param is_on_negative_strand: is the gene on the negative strand?
    :return: Tuple of (start-0-based, end-1-based) position for the region of interest
    """
    if is_on_negative_strand:
        # convert 1-based to 0-based
        tss = transcript_end - 1
    else:
        tss = transcript_start

    return tss, tss + 1


def get_tss_from_genome_annotation(genome_annotation: pd.DataFrame):
    """
    Get TSS from genome annotation
    :param genome_annotation: Pandas dataframe with the following columns:
        - Chromosome
        - Start
        - End
        - Strand
        - Feature
        - gene_id
        - transcript_id
    :return:
    """
    roi = genome_annotation.query("`Feature` == 'transcript'")
    roi = roi.assign(
        transcript_start=roi["Start"],
        transcript_end=roi["End"],
    )

    def adjust_row(row):
        start, end = get_tss_from_transcript(row.Start, row.End, row.Strand == '-')
        row.Start = start
        row.End = end

        return row

    roi = roi.apply(adjust_row, axis=1)
    roi['tss'] = roi["Start"]
    return roi


class VCF_Enformer_DL(Enformer_DL):
    def __init__(
            self,
            fasta_file,
            gtf_file,
            vcf_file,
            vcf_file_tbi=None,
            vcf_lazy=True,
            upstream_tss: int = 10,
            downstream_tss: int = 10,
            seq_length: int = SEQUENCE_LENGTH,
            is_onehot: bool = True
    ):
        # reads the genome annotation
        # start and end are transformed to 0-based and 1-based respectively
        genome_annotation = pr.read_gtf(gtf_file, as_df=True)
        genome_annotation = get_tss_from_genome_annotation(genome_annotation)
        roi = pr.PyRanges(genome_annotation)
        roi = roi.extend(ext={"5": upstream_tss, "3": downstream_tss})
        roi.landmark = roi.tss

        from kipoiseq.extractors import MultiSampleVCF
        super().__init__(
            roi_regions=roi,
            reference_sequence=FastaStringExtractor(fasta_file, use_strand=True),
            variants=MultiSampleVCF(vcf_file, lazy=vcf_lazy),
            is_onehot=is_onehot,
            seq_length=seq_length
        )
