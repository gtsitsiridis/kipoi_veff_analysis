import pathlib
from abc import ABC, abstractmethod

import pandas as pd
from kipoi.data import SampleGenerator
from kipoiseq import Interval, Variant
from kipoiseq.extractors import VariantSeqExtractor, SingleVariantMatcher, FastaStringExtractor
import pyranges as pr
from kipoiseq.extractors import MultiSampleVCF
from kipoiseq.transforms.functional import one_hot_dna
import pyarrow as pa
import numpy as np
from .utils import get_tss_from_genome_annotation, construct_enformer_interval
from .constants import AlleleType
from kipoi_enformer.logger import logger

__all__ = ['TSSDataloader', 'RefTSSDataloader', 'VCFTSSDataloader']

# length of sequence which enformer gets as input
# ═════┆═════┆════════════════════════┆═════┆═════
SEQUENCE_LENGTH = 393_216


class TSSDataloader(SampleGenerator, ABC):
    def __init__(self, allele_type: AlleleType, fasta_file, gtf: pd.DataFrame | str, chromosome: str,
                 seq_length: int = SEQUENCE_LENGTH,
                 shift: int = 43, size: int = None, canonical_only: bool = False, protein_coding_only: bool = False,
                 *args, **kwargs):
        """

        :param fasta_file: Fasta file with the reference genome
        :param gtf: GTF file with genome annotation
        :param seq_length: The length of the sequence to return. This should be the length of the Enformer input sequence.
        :param shift: For each sequence, we have 3 shifts, -shift, 0, shift, in relation to the TSS.
        :param size: The number of samples to return. If None, all samples are returned.
        :param canonical_only: If True, only Ensembl canonical transcripts are extracted from the genome annotation
        :param protein_coding_only: If True, only protein coding transcripts are extracted from the genome annotation
        """

        super().__init__(*args, **kwargs)
        assert shift >= 0, f"shift must be positive or zero but got {shift}"
        assert shift < seq_length, f"shift must be smaller than seq_length but got {shift} >= {seq_length}"

        self._reference_sequence = FastaStringExtractor(fasta_file, use_strand=True)
        if not self._reference_sequence.use_strand:
            raise ValueError(
                "Reference sequence fetcher does not use strand but this is needed to obtain correct sequences!")
        self._variant_seq_extractor = VariantSeqExtractor(reference_sequence=self._reference_sequence)
        self._canonical_only = canonical_only
        self._protein_coding_only = protein_coding_only
        self._size = size
        self._seq_length = seq_length
        self.chromosome = chromosome
        logger.debug(f"Loading genome annotation for chromosome {chromosome}")
        self._genome_annotation = get_tss_from_genome_annotation(gtf, chromosome=self.chromosome,
                                                                 canonical_only=canonical_only,
                                                                 protein_coding_only=protein_coding_only)
        self._shifts = (0,) if shift == 0 else (-shift, 0, shift)
        self.metadata = {'shifts': ';'.join([str(x) for x in self._shifts]), 'allele_type': allele_type.value,
                         'seq_length': str(self._seq_length)}

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
            - sequences: Dictionary of sequences for each TSS shift
            - metadata: Dictionary with metadata for the sample
        """
        counter = 0
        if self._genome_annotation is None or len(self._genome_annotation) == 0:
            return
        for metadata, sequences in self._sample_gen():
            # check if we reached the end of the dataset
            if self._size is not None and counter == self._size:
                break
            counter += 1

            yield {
                "sequences": sequences,
                "metadata": metadata
            }

    @classmethod
    def from_allele_type(cls, allele_type: AlleleType, *args, **kwargs):
        if allele_type == AlleleType.REF:
            return RefTSSDataloader(*args, **kwargs)
        elif allele_type == AlleleType.ALT:
            return VCFTSSDataloader(*args, **kwargs)
        else:
            raise ValueError(f"Unknown allele type: {allele_type}")


class RefTSSDataloader(TSSDataloader):
    def __init__(self, fasta_file, gtf: pd.DataFrame | str, chromosome: str, seq_length: int = SEQUENCE_LENGTH,
                 shift: int = 43,
                 size: int = None, canonical_only: bool = False, protein_coding_only: bool = False, *args, **kwargs):
        """

        :param fasta_file: Fasta file with the reference genome
        :param gtf: GTF file with genome annotation
        :param chromosome: The chromosome to filter for
        :param seq_length: The length of the sequence to return. This should be the length of the Enformer input sequence.
        :param shift: For each sequence, we have 3 shifts, -shift, 0, shift, in relation to the TSS.
        :param size: The number of samples to return. If None, all samples are returned.
        :param canonical_only: If True, only Ensembl canonical transcripts are extracted from the genome annotation
        :param protein_coding_only: If True, only protein coding transcripts are extracted from the genome annotation
        """

        super().__init__(AlleleType.REF, chromosome=chromosome, fasta_file=fasta_file, gtf=gtf,
                         seq_length=seq_length, shift=shift, size=size, canonical_only=canonical_only,
                         protein_coding_only=protein_coding_only, *args, **kwargs)
        logger.debug(f"Dataloader is ready for chromosome {chromosome}")

    def _sample_gen(self):
        for _, row in self._genome_annotation.iterrows():
            chromosome = row['Chromosome']
            strand = row.get('Strand', '.')
            tss = row['tss']
            enformer_interval = construct_enformer_interval(chromosome, strand, tss, self._seq_length)
            sequences = []
            # shift intervals and extract sequences
            for shift in self._shifts:
                shifted_enformer_interval = enformer_interval.shift(shift, use_strand=False)
                assert shifted_enformer_interval.width() == self._seq_length, \
                    f"enformer_interval width must be {self._seq_length} but got {enformer_interval.width()}"
                sequences.append(one_hot_dna(self._reference_sequence.extract(shifted_enformer_interval)))

            metadata = {
                "enformer_start": enformer_interval.start,  # 0-based start of the enformer input sequence
                "enformer_end": enformer_interval.end,  # 1-based stop of the enformer input sequence
                "tss": tss,  # 0-based position of the TSS
                "chr": chromosome,
                "strand": strand,
                "gene_id": row['gene_id'],
                "transcript_id": row['transcript_id'],
                "transcript_start": row['transcript_start'],  # 0-based
                "transcript_end": row['transcript_end'],  # 1-based
            }

            yield metadata, np.stack(sequences)

    def __len__(self):
        if self._genome_annotation is None:
            return 0
        return len(self._genome_annotation) if self._size is None else min(self._size, len(self._genome_annotation))

    @property
    def pyarrow_metadata_schema(self):
        """
        Get the pyarrow schema for the metadata.
        :return: PyArrow schema for the metadata
        """
        return pa.schema(
            [
                ('enformer_start', pa.int64()),
                ('enformer_end', pa.int64()),
                ('tss', pa.int64()),
                ('chr', pa.string()),
                ('strand', pa.string()),
                ('gene_id', pa.string()),
                ('transcript_id', pa.string()),
                ('transcript_start', pa.int64()),
                ('transcript_end', pa.int64()),
            ], metadata=self.metadata)


class VCFTSSDataloader(TSSDataloader):
    def __init__(self, fasta_file, gtf: pd.DataFrame | str, vcf_file, chromosome: str, vcf_lazy=True,
                 variant_upstream_tss: int = 10,
                 variant_downstream_tss: int = 10, seq_length: int = SEQUENCE_LENGTH, shift: int = 43, size: int = None,
                 canonical_only: bool = False, protein_coding_only: bool = False, *args, **kwargs):
        """

        :param fasta_file: Fasta file with the reference genome
        :param gtf: GTF file with genome annotation
        :param chromosome: The chromosome to filter for
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

        super().__init__(AlleleType.ALT, fasta_file=fasta_file, gtf=gtf, chromosome=chromosome,
                         seq_length=seq_length, shift=shift, size=size, canonical_only=canonical_only,
                         protein_coding_only=protein_coding_only, *args,
                         **kwargs)
        assert shift < variant_downstream_tss + variant_upstream_tss + 1, \
            f"shift must be smaller than downstream_tss + upstream_tss + 1 but got {shift} >= {variant_downstream_tss + variant_upstream_tss + 1}"

        self.variant_seq_extractor = VariantSeqExtractor(reference_sequence=self._reference_sequence)
        self.vcf_file = vcf_file
        self.vcf_lazy = vcf_lazy
        self.variant_upstream_tss = variant_upstream_tss
        self.variant_downstream_tss = variant_downstream_tss
        logger.debug(f"Dataloader is ready for chromosome {chromosome}")

    def _sample_gen(self):
        for interval, variant in self._get_single_variant_matcher(self.vcf_lazy):
            attrs = interval.attrs
            tss = attrs['tss']

            enformer_interval = construct_enformer_interval(interval.chrom, interval.strand, tss, self._seq_length)
            sequences = []
            # shift intervals and extract sequences
            for shift in self._shifts:
                shifted_enformer_interval = enformer_interval.shift(shift, use_strand=False)
                assert shifted_enformer_interval.width() == self._seq_length, \
                    f"enformer_interval width must be {self._seq_length} but got {enformer_interval.width()}"

                sequences.append(self._extract_seq(tss=tss, interval=shifted_enformer_interval,
                                                   variant=variant))

            metadata = {
                "enformer_start": enformer_interval.start,  # 0-based start of the enformer input sequence
                "enformer_end": enformer_interval.end,  # 1-based stop of the enformer input sequence
                "tss": tss,  # 0-based position of the TSS
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
            yield metadata, np.stack(sequences),

    def _extract_seq(self, tss: int, interval: Interval, variant: Variant):
        assert interval.width() == self._seq_length, f"interval width must be {self._seq_length} but got {interval.width()}"

        # Note: If the tss is within the variant's interval
        # ====|----------Variant----------|=======
        # ===========|TSS|===================
        # We take as the new tss the first base downstream the variant
        # For an explanation on how this works, look at the function
        # VariantSeqExtractor.extract(self, interval, variants, anchor, fixed_len=True, **kwargs)

        # the tss/anchor is going to be in the middle of the sequence for both alleles
        alt_seq = self.variant_seq_extractor.extract(
            interval,
            [variant],
            anchor=tss
        )

        return one_hot_dna(alt_seq)

    def __len__(self):
        if self._genome_annotation is None:
            return 0
        tmp_matcher = self._get_single_variant_matcher(vcf_lazy=False)
        total = sum(1 for _, _ in tmp_matcher)
        if self._size:
            return min(self._size, total)
        return total

    def _get_single_variant_matcher(self, vcf_lazy=True):
        # reads the genome annotation
        # start and end are transformed to 0-based and 1-based respectively
        roi = pr.PyRanges(self._genome_annotation)
        roi = roi.extend(ext={"5": self.variant_upstream_tss, "3": self.variant_downstream_tss})
        # todo do assert length of roi

        interval_attrs = ['gene_id', 'transcript_id', 'tss', 'transcript_start', 'transcript_end']
        for attr in interval_attrs:
            assert attr in roi.columns, f"attr must be in {roi.columns}"
        variants = MultiSampleVCF(self.vcf_file, lazy=vcf_lazy)

        return SingleVariantMatcher(
            variant_fetcher=variants,
            pranges=roi,
            interval_attrs=interval_attrs
        )

    @property
    def pyarrow_metadata_schema(self):
        """
        Get the pyarrow schema for the metadata.
        :return: PyArrow schema for the metadata
        """
        return pa.schema(
            [
                ('enformer_start', pa.int64()),
                ('enformer_end', pa.int64()),
                ('tss', pa.int64()),
                ('chr', pa.string()),
                ('strand', pa.string()),
                ('gene_id', pa.string()),
                ('transcript_id', pa.string()),
                ('transcript_start', pa.int64()),
                ('transcript_end', pa.int64()),
                ('variant_start', pa.int64()),
                ('variant_end', pa.int64()),
                ('ref', pa.string()),
                ('alt', pa.string()),
            ], metadata=self.metadata)
