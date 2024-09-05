from __future__ import annotations

from abc import ABC, abstractmethod

from kipoi.data import SampleGenerator
import pyarrow as pa
import math
import pandas as pd
from kipoiseq.extractors import VariantSeqExtractor, FastaStringExtractor
from kipoiseq import Interval, Variant
from kipoiseq.transforms.functional import one_hot_dna
from kipoi_aparent2.utils import gtf_to_pandas


class Dataloader(SampleGenerator, ABC):
    def __init__(self, fasta_file, size: int = None, *args, **kwargs):
        """

        :param fasta_file: Fasta file with the reference genome
        :param gtf: GTF file with genome annotation or DataFrame with genome annotation
        :param chromosome: The chromosome to filter for. If None, all chromosomes are used.
        :param seq_length: The length of the sequence to return. This should be the length of the Enformer input sequence.
        :param shift: For each sequence, we have 3 shifts, -shift, 0, shift, in relation to a reference point.
        :param size: The number of samples to return. If None, all samples are returned.
        :param canonical_only: If True, only Ensembl canonical transcripts are extracted from the genome annotation
        :param protein_coding_only: If True, only protein coding transcripts are extracted from the genome annotation
        :param gene_ids: If provided, only the gene with this ID is extracted from the genome annotation
        """

        super().__init__(*args, **kwargs)
        self._reference_sequence = FastaStringExtractor(fasta_file, use_strand=True)
        if not self._reference_sequence.use_strand:
            raise ValueError(
                "Reference sequence fetcher does not use strand but this is needed to obtain correct sequences!")
        self._size = size

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("The length of the dataset is not known.")

    @abstractmethod
    def _sample_gen(self):
        """
        Generate samples for the dataset. The generator should return a tuple of metadata and sequences.
        :return:
        """
        raise NotImplementedError("The sample generator is not implemented.")

    @property
    @abstractmethod
    def pyarrow_metadata_schema(self) -> pa.schema:
        """
        Get the pyarrow schema for the metadata.
        :return: PyArrow schema for the metadata
        """
        raise NotImplementedError("The metadata schema is not implemented.")

    def __iter__(self):
        """
        Iterate over the dataset.

        :return: Iterator over the dataset. Each item is a dictionary with the following
            keys:
            - sequences:
            - metadata:
        """
        counter = 0
        for metadata, sequence in self._sample_gen():
            # check if we reached the end of the dataset
            if self._size is not None and counter == self._size:
                break
            counter += 1

            yield {
                "sequence": sequence,
                "metadata": metadata
            }


def construct_interval(chrom, strand, anchor, seq_length):
    # input interval without shift
    # if the sequence length is even, the tss is closer to the end of the sequence
    # if the sequence length is odd, the tss is in the middle of the sequence
    five_end_len = math.floor(seq_length / 2)
    three_end_len = math.ceil(seq_length / 2)

    # WARNING: kipoiseq.Interval has a 0-based end!
    interval = Interval(chrom=chrom,
                        start=anchor - five_end_len,
                        end=anchor + three_end_len,
                        strand=strand)

    assert (interval.width()) == seq_length, \
        f"interval width must be {seq_length} but got {interval.width()}"
    assert (anchor - interval.start) == seq_length // 2, \
        f"tss must be in the middle of the interval but got {anchor - interval.start}"
    return interval


def extract_sequence_around_anchor(shift, chromosome, strand, anchor, seq_length,
                                   ref_seq_extractor: FastaStringExtractor,
                                   variant_extractor: VariantSeqExtractor | None = None,
                                   variant: Variant | None = None):
    assert variant_extractor is None or (variant is not None and variant_extractor is not None), \
        "variant_extractor must be provided if variant is not None"
    chrom_len = len(ref_seq_extractor.fasta.records[chromosome])

    # WARNING: kipoiseq.Interval has a 0-based end!
    # shift interval and extract sequence
    interval = construct_interval(chromosome, strand, anchor, seq_length).shift(shift, use_strand=True)
    five_end_pad = 0
    three_end_pad = 0
    if interval.start < 0:
        five_end_pad = abs(interval.start)
    if interval.end >= chrom_len:
        three_end_pad = interval.end - chrom_len + 1
    if five_end_pad > 0 or three_end_pad > 0:
        interval = interval.truncate(chrom_len)

    if variant is not None:
        seq = variant_extractor.extract(interval,
                                        [variant],
                                        anchor=anchor,
                                        fixed_length=True,
                                        is_padding=True,
                                        chrom_len=chrom_len,
                                        )
    else:
        seq = ref_seq_extractor.extract(interval)

    if five_end_pad > 0:
        seq = 'N' * five_end_pad + seq

    if three_end_pad > 0:
        seq = seq + 'N' * three_end_pad

    assert len(seq) == seq_length, \
        f"interval width must be {seq_length} but got {len(seq)}"

    return one_hot_dna(seq), interval
