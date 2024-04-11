from kipoi.data import SampleIterator
from kipoiseq import Interval, Variant
from kipoiseq.extractors import VariantSeqExtractor, SingleVariantMatcher, BaseExtractor, FastaStringExtractor
import math
import pandas as pd
import pyranges as pr
from kipoiseq.extractors import MultiSampleVCF

# length of sequence which enformer gets as input
# ═════┆═════┆════════════════════════┆═════┆═════
SEQUENCE_LENGTH = 393_216


class VCFEnformerDL:
    def __init__(
            self,
            fasta_file,
            gtf_file,
            vcf_file,
            vcf_lazy=True,
            variant_upstream_tss: int = 10,
            variant_downstream_tss: int = 10,
            seq_length: int = SEQUENCE_LENGTH,
            shift: int = 43,
            size: int = None,
            canonical_only: bool = False,
            protein_coding_only: bool = False,
    ):
        """

        :param fasta_file: Fasta file with the reference genome
        :param gtf_file: GTF file with genome annotation
        :param vcf_file: VCF file with variants
        :param vcf_lazy: If True, the VCF file is read lazily
        :param variant_upstream_tss: The number of bases upstream the TSS to look for variants
        :param variant_downstream_tss: The number of bases downstream the TSS to look for variants
        :param seq_length: The length of the sequence to return. This should be the length of the Enformer input sequence.
        :param shift: For each sequence, we have 3 shifts, -shift, 0, shift, in relation to the TSS.
        :param size: The number of samples to return. If None, all samples are returned.
        :param canonical_only: If True, only Ensembl canonical transcripts are extracted from the genome annotation
        :param protein_coding_only: If True, only protein coding transcripts are extracted from the genome annotation
        """

        assert shift < variant_downstream_tss + variant_upstream_tss + 1, \
            f"shift must be smaller than downstream_tss + upstream_tss + 1 but got {shift} >= {variant_downstream_tss + variant_upstream_tss + 1}"
        assert shift >= 0, f"shift must be positive or zero but got {shift}"
        assert shift < seq_length, f"shift must be smaller than seq_length but got {shift} >= {seq_length}"

        self.reference_sequence = FastaStringExtractor(fasta_file, use_strand=True)
        if not self.reference_sequence.use_strand:
            raise ValueError(
                "Reference sequence fetcher does not use strand but this is needed to obtain correct sequences!")
        self.variant_seq_extractor = VariantSeqExtractor(reference_sequence=self.reference_sequence)
        self.canonical_only = canonical_only
        self.protein_coding_only = protein_coding_only
        self.shift = shift
        self.size = size
        self.seq_length = seq_length
        self.gtf_file = gtf_file
        self.vcf_file = vcf_file
        self.variant_upstream_tss = variant_upstream_tss
        self.variant_downstream_tss = variant_downstream_tss
        self.matcher = get_single_variant_matcher(gtf_file, vcf_file, vcf_lazy, variant_upstream_tss,
                                                  variant_downstream_tss, canonical_only, protein_coding_only)

    def _extract_seq(self, landmark: int, interval: Interval, variant: Variant):
        assert interval.width() == self.seq_length, f"interval width must be {self.seq_length} but got {interval.width()}"

        # Note: If the landmark is within the variant's interval
        # ====|----------Variant----------|=======
        # ===========|Landmark|===================
        # We take as the new landmark the first base downstream the variant
        # For an explanation on how this works, look at the function
        # VariantSeqExtractor.extract(self, interval, variants, anchor, fixed_len=True, **kwargs)

        # the landmark/anchor is going to be in the middle of the sequence for both alleles
        ref_seq = self.reference_sequence.extract(interval)
        alt_seq = self.variant_seq_extractor.extract(
            interval,
            [variant],
            anchor=landmark
        )

        return ref_seq, alt_seq

    def __len__(self):
        tmp_matcher = get_single_variant_matcher(self.gtf_file, self.vcf_file, False, self.variant_upstream_tss,
                                                 self.variant_downstream_tss, self.canonical_only,
                                                 self.protein_coding_only)
        total = sum(1 for _, _ in tmp_matcher)
        if self.size:
            return min(self.size, total)
        return total

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
            # the landmark is in the middle of the enformer input sequence
            # if the sequence length is even, the landmark is closer to the end of the sequence by 1 base,
            # because the end is 1 based
            # if the sequence length is odd, the landmark is in the middle of the sequence
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

            sequences = dict()
            # shift intervals and extract sequences
            for shift in shifts:
                shifted_enformer_interval = enformer_interval.shift(shift, use_strand=False)
                assert shifted_enformer_interval.width() == self.seq_length, \
                    f"enformer_interval width must be {self.seq_length} but got {enformer_interval.width()}"

                ref_seq, alt_seq = self._extract_seq(landmark=landmark, interval=shifted_enformer_interval,
                                                     variant=variant)
                sequences[f'ref_{shift}'] = ref_seq
                sequences[f'alt_{shift}'] = alt_seq

            yield {
                "sequences": sequences,
                "metadata": {
                    "enformer_start": enformer_interval.start,  # 0-based start of the enformer input sequence
                    "enformer_end": enformer_interval.end,  # 1-based stop of the enformer input sequence
                    "landmark_pos": landmark,  # 0-based position of the landmark (TSS)
                    "chr": interval.chrom,
                    "strand": interval.strand,
                    "gene_id": attrs['gene_id'],
                    "transcript_id": attrs['transcript_id'],
                    "transcript_start": attrs['transcript_start'],  # 0-based
                    "transcript_end": attrs['transcript_end'],  # 1-based
                    "variant_start": variant.start,  # 0-based
                    "variant_end": variant.end,  # 1-based
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


def get_tss_from_genome_annotation(gtf_file: str, canonical_only: bool, protein_coding_only: bool):
    """
    Get TSS from genome annotation
    :param canonical_only:
    :param protein_coding_only:
    :param gtf_file: str, path to the GTF file
    :return: genome_annotation with additional columns tss (0-based), transcript_start (0-based), transcript_end (1-based)
    """
    genome_annotation = pr.read_gtf(gtf_file, as_df=True, duplicate_attr=True)
    roi = genome_annotation.query("`Feature` == 'transcript'")
    if protein_coding_only:
        roi = roi.query("`gene_type` == 'protein_coding'")
    if canonical_only:
        # check if Ensembl_canonical is in the set of tags
        roi = roi[roi['tag'].apply(lambda x: False if pd.isna(x) else ('Ensembl_canonical' in x.split(',')))]

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


def get_single_variant_matcher(gtf_file: str, vcf_file: str, vcf_lazy: bool, upstream_tss: int, downstream_tss: int,
                               canonical_only: bool, protein_coding_only: bool):
    # reads the genome annotation
    # start and end are transformed to 0-based and 1-based respectively
    genome_annotation = get_tss_from_genome_annotation(gtf_file, canonical_only, protein_coding_only)
    roi = pr.PyRanges(genome_annotation)
    roi = roi.extend(ext={"5": upstream_tss, "3": downstream_tss})
    roi.landmark = roi.tss
    # todo do assert length of roi

    interval_attrs = ['gene_id', 'transcript_id', 'landmark', 'transcript_start', 'transcript_end']
    for attr in interval_attrs:
        assert attr in roi.columns, f"attr must be in {roi.columns}"
    variants = MultiSampleVCF(vcf_file, lazy=vcf_lazy)

    return SingleVariantMatcher(
        variant_fetcher=variants,
        pranges=roi,
        interval_attrs=interval_attrs
    )
