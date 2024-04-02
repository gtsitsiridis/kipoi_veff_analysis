from kipoi.data import SampleIterator
from kipoiseq import Interval, Variant
from kipoiseq.extractors import VariantSeqExtractor, SingleVariantMatcher, BaseExtractor, FastaStringExtractor
import math
import pandas as pd
import pyranges as pr
import numpy as np
from kipoiseq.extractors import MultiSampleVCF
from kipoi_enformer.logger import logger

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


class Enformer_DL(SampleIterator):
    def __init__(
            self,
            roi_regions: pr.PyRanges,
            reference_sequence: BaseExtractor,
            variants: VariantFetcher,
            seq_length: int,
            shift: int,
            size: int = None,
    ):
        interval_attrs = ['gene_id', 'transcript_id', 'landmark', 'transcript_start', 'transcript_end']
        for attr in interval_attrs:
            assert attr in roi_regions.columns, f"attr must be in {roi_regions.columns}"
        assert shift >= 0, f"shift must be positive or zero but got {shift}"
        assert shift < seq_length, f"shift must be smaller than seq_length but got {shift} >= {seq_length}"

        self.seq_length = seq_length
        self.roi_regions = roi_regions
        self.reference_sequence = reference_sequence
        self.variants = variants
        self.shift = shift
        self.size = size

        if not self.reference_sequence.use_strand:
            raise ValueError(
                "Reference sequence fetcher does not use strand but this is needed to obtain correct sequences!")
        self.variant_seq_extractor = VariantSeqExtractor(reference_sequence=reference_sequence)

        self.matcher = SingleVariantMatcher(
            variant_fetcher=self.variants,
            pranges=self.roi_regions,
            interval_attrs=interval_attrs
        )

    def _extract_seq(self, landmark: int, interval: Interval, variant: Variant):
        assert interval.width() == self.seq_length, f"interval width must be {self.seq_length} but got {interval.width()}"

        # Note: If the landmark is within the variant's interval
        # ====|----------Variant----------|=======
        # ===========|Landmark|===================
        # We take as the new landmark the first base downstream the variant
        # For an explanation on how this works, look at the function
        # VariantSeqExtractor.extract(self, interval, variants, anchor, fixed_len=True, **kwargs)

        ref_seq = self.reference_sequence.extract(interval)
        alt_seq = self.variant_seq_extractor.extract(
            interval,
            [variant],
            anchor=landmark
        )

        return ref_seq, alt_seq

    def __iter__(self):
        """
        Iterate over the dataset.

        :return: Iterator over the dataset. Each item is a dictionary with the following
            keys:
            - sequences: Dictionary with the following keys:
                - ref: List of reference sequences for each shift
                - alt: List of alternative sequences for each shift
            - metadata: Dictionary with the following keys:
                - shift: Shift of the Enformer input sequence
                - enformer_start: 0-based start of the Enformer input sequence
                - enformer_stop: 1-based stop of the Enformer input sequence
                - landmark_pos: 0-based position of the landmark (TSS)
                - chr: Chromosome
                - strand: Strand
                - gene_id: Gene ID
                - transcript_id: Transcript ID
                - transcript_start: 0-based
                - transcript_end: 1-based
                - variant_start: 0-based
                - variant_stop: 1-based
                - ref: Reference allele
                - alt: Alternative allele
        """
        interval: Interval
        variant: Variant
        shifts = (0,) if self.shift == 0 else (-self.shift, 0, self.shift)
        counter = 0
        for interval, variant in self.matcher:
            # check if we reached the end of the dataset
            if self.size is not None and counter == self.size:
                break
            counter += 1

            attrs = interval.attrs
            landmark = attrs['landmark']

            # enformer input interval without shift
            five_end_len = math.floor(self.seq_length / 2)
            three_end_len = math.ceil(self.seq_length / 2)
            enformer_interval = Interval(chrom=interval.chrom,
                                         start=landmark - five_end_len,
                                         end=landmark + three_end_len,
                                         strand=interval.strand)
            assert enformer_interval.width() == self.seq_length, \
                f"enformer_interval width must be {self.seq_length} but got {enformer_interval.width()}"
            assert (landmark - enformer_interval.start) == self.seq_length // 2, \
                f"landmark must be in the middle of the enformer_interval but got {landmark - enformer_interval.start}"

            sequences = {
                "ref": [],
                "alt": []
            }
            # shift intervals and extract sequences
            for shift in shifts:
                shifted_enformer_interval = enformer_interval.shift(shift, use_strand=False)
                assert shifted_enformer_interval.width() == self.seq_length, \
                    f"enformer_interval width must be {self.seq_length} but got {enformer_interval.width()}"

                ref_seq, alt_seq = self._extract_seq(landmark=landmark, interval=shifted_enformer_interval,
                                                     variant=variant)
                for allele in ('ref', 'alt'):
                    yield {
                        "sequence": ref_seq if allele == 'ref' else alt_seq,
                        "metadata": {
                            # Note: To get the landmark bin:
                            # landmark_bin = (landmark - shift - (PRED_SEQUENCE_LENGTH - 1) // 2) // BIN_SIZE
                            "shift": shift,  # shift of the Enformer input sequence,
                            "allele": allele,  # "ref" or "alt
                            "enformer_start": enformer_interval.start,  # 0-based start of the enformer input sequence
                            "enformer_stop": enformer_interval.end,  # 1-based stop of the enformer input sequence
                            "landmark_pos": landmark,  # 0-based position of the landmark (TSS)
                            "chr": interval.chrom,
                            "strand": interval.strand,
                            "gene_id": attrs['gene_id'],
                            "transcript_id": attrs['transcript_id'],
                            "transcript_start": attrs['transcript_start'],  # 0-based
                            "transcript_end": attrs['transcript_end'],  # 1-based
                            "variant_start": variant.start,  # 0-based
                            "variant_stop": variant.end,  # 1-based
                            "ref": variant.ref,
                            "alt": variant.alt,
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
    :return: genome_annotation with additional columns tss (0-based), transcript_start (0-based), transcript_end (1-based)
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
            shift: int = 43,
            is_onehot: bool = True,
            size: int = None
    ):
        assert shift < downstream_tss + upstream_tss + 1, \
            f"shift must be smaller than downstream_tss + upstream_tss + 1 but got {shift} >= {downstream_tss + upstream_tss + 1}"

        # reads the genome annotation
        # start and end are transformed to 0-based and 1-based respectively
        genome_annotation = pr.read_gtf(gtf_file, as_df=True)
        genome_annotation = get_tss_from_genome_annotation(genome_annotation)
        roi = pr.PyRanges(genome_annotation)
        roi = roi.extend(ext={"5": upstream_tss, "3": downstream_tss})
        roi.landmark = roi.tss
        # todo do assert length of roi

        super().__init__(
            roi_regions=roi,
            reference_sequence=FastaStringExtractor(fasta_file, use_strand=True),
            variants=MultiSampleVCF(vcf_file, lazy=vcf_lazy),
            seq_length=seq_length,
            shift=shift,
            size=size
        )
