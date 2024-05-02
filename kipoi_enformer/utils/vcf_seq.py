from kipoiseq.extractors import VariantSeqExtractor
from pyfaidx import complement, Sequence

__all__ = [
    "extract_variant"
]


# This file contains modified functions from kipoiseq.extractors.VariantSeqExtractor


def extract_variant(variant_extractor: VariantSeqExtractor, interval, variants, anchor):
    """
    Modified extract method from kipoiseq.extractors.VariantSeqExtractor.extract.
    This modification pads the sequence with 'N's if sequence can't extend to the fixed length,
    either upstream or downstream.

    :param variant_extractor:
    :param interval: pybedtools.Interval Region of interest from
            which to query the sequence. 0-based
    :param variants: absolution position w.r.t. the interval start. (0-based).
            E.g. for an interval of `chr1:10-20` the anchor of 10 denotes
            the point chr1:10 in the 0-based coordinate system.
    :param anchor:
    :return: A single sequence (`str`) with all the variants applied.
    """

    # Preprocessing
    anchor = max(min(anchor, interval.end), interval.start)
    variant_pairs = variant_extractor._variant_to_sequence(variants)

    # 1. Split variants overlapping with anchor
    # and interval start end if not fixed_len
    variant_pairs = variant_extractor._split_overlapping(variant_pairs, anchor)

    variant_pairs = list(variant_pairs)

    # 2. split the variants into upstream and downstream
    # and sort the variants in each interval
    upstream_variants = sorted(
        filter(lambda x: x[0].start >= anchor, variant_pairs),
        key=lambda x: x[0].start
    )

    downstream_variants = sorted(
        filter(lambda x: x[0].start < anchor, variant_pairs),
        key=lambda x: x[0].start,
        reverse=True
    )

    # 3. Extend start and end position for deletions
    istart, iend = variant_extractor._updated_interval(
        interval, upstream_variants, downstream_variants)

    # 4. Iterate from the anchor point outwards. At each
    # register the interval from which to take the reference sequence
    # as well as the interval for the variant
    down_sb = variant_extractor._downstream_builder(
        downstream_variants, interval, anchor, istart)

    up_sb = variant_extractor._upstream_builder(
        upstream_variants, interval, anchor, iend)

    # 5. fetch the sequence and restore intervals in builder
    seq = variant_extractor._fetch(interval, istart, iend)
    up_sb.restore(seq)
    down_sb.restore(seq)

    # 6. Concate sequences from the upstream and downstream splits. Concat
    # upstream and downstream sequence. Cut to fix the length.
    down_str = down_sb.concat()
    up_str = up_sb.concat()

    down_str, up_str = variant_extractor._cut_to_fix_len(
        down_str, up_str, interval, anchor)

    seq = down_str + up_str

    if interval.strand == '-':
        seq = complement(seq)[::-1]

    return seq
