import math
import pandas as pd
import pyranges as pr
import tensorflow as tf
from kipoiseq.extractors import VariantSeqExtractor, FastaStringExtractor
from kipoiseq import Interval, Variant
from kipoiseq.transforms.functional import one_hot_dna
from .vcf_seq import extract_variant


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


def get_tss_from_genome_annotation(gtf: pd.DataFrame | str, chromosome: str | None = None,
                                   protein_coding_only: bool = False, canonical_only: bool = False):
    """
    Get TSS from genome annotation
    :return: genome_annotation with additional columns tss (0-based), transcript_start (0-based), transcript_end (1-based)
    """
    if not isinstance(gtf, pd.DataFrame):
        genome_annotation = pr.read_gtf(gtf, as_df=True, duplicate_attr=True)
    else:
        genome_annotation = gtf.copy()

    if chromosome is not None:
        genome_annotation = genome_annotation.query("`Chromosome` == @chromosome")
    roi = genome_annotation.query("`Feature` == 'transcript'")
    if protein_coding_only:
        roi = roi.query("`gene_type` == 'protein_coding'")
    if canonical_only:
        # check if Ensembl_canonical is in the set of tags
        roi = roi[roi['tag'].apply(lambda x: False if pd.isna(x) else ('Ensembl_canonical' in x.split(',')))]
    if len(roi) == 0:
        return None

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


def construct_enformer_interval(chrom, strand, tss, seq_length):
    # enformer input interval without shift
    # if the sequence length is even, the tss is closer to the end of the sequence
    # if the sequence length is odd, the tss is in the middle of the sequence
    five_end_len = math.floor(seq_length / 2)
    three_end_len = math.ceil(seq_length / 2)

    # WARNING: kipoiseq.Interval has a 0-based end!
    enformer_interval = Interval(chrom=chrom,
                                 start=tss - five_end_len,
                                 end=tss + three_end_len,
                                 strand=strand)

    assert (enformer_interval.width()) == seq_length, \
        f"enformer_interval width must be {seq_length} but got {enformer_interval.width()}"
    assert (tss - enformer_interval.start) == seq_length // 2, \
        f"tss must be in the middle of the enformer_interval but got {tss - enformer_interval.start}"
    return enformer_interval


def extract_sequences_around_tss(shifts, chromosome, strand, tss, seq_length,
                                 ref_seq_extractor: FastaStringExtractor,
                                 variant_extractor: VariantSeqExtractor | None = None,
                                 variant: Variant | None = None):
    assert variant_extractor is None or (variant is not None and variant_extractor is not None), \
        "variant_extractor must be provided if variant is not None"
    chrom_len = len(ref_seq_extractor.fasta.records[chromosome])

    # WARNING: kipoiseq.Interval has a 0-based end!
    enformer_interval = construct_enformer_interval(chromosome, strand, tss, seq_length)
    sequences = []
    # shift intervals and extract sequences
    for shift in shifts:
        shifted_enformer_interval = enformer_interval.shift(shift, use_strand=False)
        five_end_pad = 0
        three_end_pad = 0
        if shifted_enformer_interval.start < 0:
            five_end_pad = abs(shifted_enformer_interval.start)
        if shifted_enformer_interval.end >= chrom_len:
            three_end_pad = shifted_enformer_interval.end - chrom_len + 1
        if five_end_pad > 0 or three_end_pad > 0:
            shifted_enformer_interval = shifted_enformer_interval.truncate(chrom_len)

        if variant is not None:
            seq = extract_variant(variant_extractor,
                                  shifted_enformer_interval,
                                  [variant],
                                  anchor=tss
                                  )
        else:
            seq = ref_seq_extractor.extract(shifted_enformer_interval)

        if five_end_pad > 0:
            seq = 'N' * five_end_pad + seq

        if three_end_pad > 0:
            seq = seq + 'N' * three_end_pad

        assert len(seq) == seq_length, \
            f"enformer_interval width must be {seq_length} but got {len(seq)}"

        sequences.append(one_hot_dna(seq))
    return sequences, enformer_interval


class RandomModel(tf.keras.Model):
    """
    A random model for testing purposes.
    """

    def predict_on_batch(self, input_tensor):
        tf.random.set_seed(42)
        return {'human': tf.abs(tf.random.normal((input_tensor.shape[0], 896, 5313))),
                'mouse': tf.abs(tf.random.normal((input_tensor.shape[0], 896, 1643))),
                }
