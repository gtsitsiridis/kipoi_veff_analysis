from __future__ import annotations

import pandas as pd
from kipoiseq.extractors import VariantSeqExtractor, SingleVariantMatcher
import pyranges as pr
from kipoiseq.extractors import MultiSampleVCF
import pyarrow as pa
import numpy as np
from .dataloader import get_cse_from_genome_annotation, extract_sequence_around_anchor
from kipoi_aparent2.constants import AlleleType
from kipoi_aparent2.logger import logger
from .dataloader import Dataloader

__all__ = ['CSEDataloader', 'RefCSEDataloader', 'VCFCSEDataloader']

# length of sequence which APARENT2 gets as input
# ═════┆═════┆════════════════════════┆═════┆═════
APARENT2_SEQUENCE_LENGTH = 205
CSE_POS_INDEX = 70 # CSE should be roughly around position 70 of the sequence. (zero-based)

class CSEDataloader(Dataloader):
    def __init__(self, allele_type: AlleleType, fasta_file, gtf: pd.DataFrame | str, chromosome: str | None = None,
                 seq_length: int = APARENT2_SEQUENCE_LENGTH, cse_pos_index: int = CSE_POS_INDEX, size: int = None,
                 canonical_only: bool = False, protein_coding_only: bool = False, gene_ids: list | None = None, *args,
                 **kwargs):
        """

        :param fasta_file: Fasta file with the reference genome
        :param gtf: GTF file with genome annotation or DataFrame with genome annotation
        :param chromosome: The chromosome to filter for. If None, all chromosomes are used.
        :param seq_length: The length of the sequence to return.
        :param cse_pos_index: The position of the CSE in the sequence. (0-based)
        :param size: The number of samples to return. If None, all samples are returned.
        :param canonical_only: If True, only Ensembl canonical transcripts are extracted from the genome annotation
        :param protein_coding_only: If True, only protein coding transcripts are extracted from the genome annotation
        :param gene_ids: If provided, only the gene with this ID is extracted from the genome annotation
        """

        super().__init__(fasta_file=fasta_file, size=size, *args, **kwargs)
        assert 0 <= cse_pos_index < seq_length, f"cse_pos_index must be between 0 and seq_length but got {cse_pos_index}"

        self._canonical_only = canonical_only
        self._canonical_only = canonical_only
        self._protein_coding_only = protein_coding_only
        self._seq_length = seq_length
        self.chromosome = chromosome
        logger.debug(f"Loading genome annotation")
        self._genome_annotation = get_cse_from_genome_annotation(gtf, chromosome=self.chromosome,
                                                                 canonical_only=canonical_only,
                                                                 protein_coding_only=protein_coding_only,
                                                                 gene_ids=gene_ids)
        self._shift = seq_length // 2 - cse_pos_index
        self.metadata = {'cse_pos_index': str(cse_pos_index), 'allele_type': allele_type.value,
                         'seq_length': str(self._seq_length)}

    @classmethod
    def from_allele_type(cls, allele_type: AlleleType, *args, **kwargs):
        if allele_type == AlleleType.REF:
            return RefCSEDataloader(*args, **kwargs)
        elif allele_type == AlleleType.ALT:
            return VCFCSEDataloader(*args, **kwargs)
        else:
            raise ValueError(f"Unknown allele type: {allele_type}")


class RefCSEDataloader(CSEDataloader):
    def __init__(self, fasta_file, gtf: pd.DataFrame | str, chromosome: str,
                 seq_length: int = APARENT2_SEQUENCE_LENGTH, cse_pos_index: int = CSE_POS_INDEX, size: int = None,
                 canonical_only: bool = False,
                 protein_coding_only: bool = False, gene_ids: list | None = None, *args, **kwargs):
        """
        :param fasta_file: Fasta file with the reference genome
        :param gtf: GTF file with genome annotation or DataFrame with genome annotation
        :param chromosome: The chromosome to filter for
        :param seq_length: The length of the sequence to return.
        :param cse_pos_index: The position of the CSE in the sequence. (0-based)
        :param size: The number of samples to return. If None, all samples are returned.
        :param canonical_only: If True, only Ensembl canonical transcripts are extracted from the genome annotation
        :param protein_coding_only: If True, only protein coding transcripts are extracted from the genome annotation
        :param gene_id: If provided, only the gene with this ID is extracted from the genome annotation
        """
        assert chromosome is not None, 'A chromosome should be provided'
        super().__init__(AlleleType.REF, chromosome=chromosome, fasta_file=fasta_file, gtf=gtf,
                         seq_length=seq_length, cse_pos_index=cse_pos_index, size=size, canonical_only=canonical_only,
                         protein_coding_only=protein_coding_only, gene_ids=gene_ids, *args, **kwargs)
        logger.debug(f"Dataloader is ready for chromosome {chromosome}")

    def _sample_gen(self):
        for _, row in self._genome_annotation.iterrows():
            try:
                chromosome = row['Chromosome']
                strand = row.get('Strand', '.')
                cse = row['cse']

                sequence, interval = extract_sequence_around_anchor(self._shift, chromosome, strand, cse,
                                                                    self._seq_length,
                                                                    ref_seq_extractor=self._reference_sequence)

                metadata = {
                    "seq_start": interval.start,  # 0-based start of the input sequence
                    "seq_end": interval.end + 1,  # 1-based stop of the input sequence
                    "cse": cse,  # 0-based position of the CSE
                    "strand": strand,
                    "gene_id": row['gene_id'],
                    "transcript_id": row['transcript_id'],
                    "transcript_start": row['transcript_start'],  # 0-based
                    "transcript_end": row['transcript_end'],  # 1-based
                }

                yield metadata, sequence
            except Exception as e:
                logger.error(f"Error processing row: {row}")
                raise e

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
        columns = [
            ('seq_start', pa.int64()),
            ('seq_end', pa.int64()),
            ('cse', pa.int64()),
            ('strand', pa.string()),
            ('gene_id', pa.string()),
            ('transcript_id', pa.string()),
            ('transcript_start', pa.int64()),
            ('transcript_end', pa.int64()), ]

        return pa.schema(columns, metadata=self.metadata)


class VCFCSEDataloader(CSEDataloader):
    def __init__(self, fasta_file, gtf: pd.DataFrame | str, vcf_file, vcf_lazy=True,
                 variant_upstream_cse: int = 10, variant_downstream_cse: int = 10,
                 seq_length: int = APARENT2_SEQUENCE_LENGTH, cse_pos_index: int = CSE_POS_INDEX, size: int = None,
                 canonical_only: bool = False, protein_coding_only: bool = False,
                 gene_ids: list | None = None, *args, **kwargs):
        """

        :param fasta_file: Fasta file with the reference genome
        :param gtf: GTF file with genome annotation or DataFrame with genome annotation
        :param vcf_file: VCF file with variants
        :param vcf_lazy: If True, the VCF file is read lazily
        :param variant_upstream_cse: The number of bases upstream the CSE to look for variants
        :param variant_downstream_cse: The number of bases downstream the CSE to look for variants
        :param seq_length: The length of the sequence to return.
        :param cse_pos_index: The position of the CSE in the sequence. (0-based)
        :param size: The number of samples to return. If None, all samples are returned.
        :param canonical_only: If True, only Ensembl canonical transcripts are extracted from the genome annotation
        :param protein_coding_only: If True, only protein coding transcripts are extracted from the genome annotation
        :param gene_id: If provided, only the gene with this ID is extracted from the genome annotation
        """

        super().__init__(AlleleType.ALT, fasta_file=fasta_file, gtf=gtf, chromosome=None,
                         seq_length=seq_length, cse_pos_index=cse_pos_index, size=size, canonical_only=canonical_only,
                         protein_coding_only=protein_coding_only, gene_ids=gene_ids, *args, **kwargs)
        assert 0 <= variant_upstream_cse <= cse_pos_index, \
            (f"variant_upstream_cse must be between 0 and {cse_pos_index} but got "
             f"{variant_upstream_cse}")
        assert 0 <= variant_downstream_cse <= self._seq_length - cse_pos_index - 1, \
            (f"variant_downstream_cse must be between 0 and {seq_length - cse_pos_index - 1} but got "
                f"{variant_downstream_cse}")

        self._variant_seq_extractor = VariantSeqExtractor(reference_sequence=self._reference_sequence)
        self.vcf_file = vcf_file
        self.vcf_lazy = vcf_lazy
        self.variant_upstream_cse = variant_upstream_cse
        self.variant_downstream_cse = variant_downstream_cse
        logger.debug(f"Dataloader is ready")

    def _sample_gen(self):
        for interval, variant in self._get_single_variant_matcher(self.vcf_lazy):
            try:
                attrs = interval.attrs
                cse = attrs['cse']
                chromosome = interval.chrom
                strand = interval.strand

                sequence, interval = extract_sequence_around_anchor(self._shift, chromosome, strand, cse,
                                                                    self._seq_length,
                                                                    ref_seq_extractor=self._reference_sequence,
                                                                    variant_extractor=self._variant_seq_extractor,
                                                                    variant=variant)
                metadata = {
                    "seq_start": interval.start,  # 0-based start of the input sequence
                    "seq_end": interval.end + 1,  # 1-based stop of the input sequence
                    "cse": cse,  # 0-based position of the CSE
                    "chrom": interval.chrom,
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
                yield metadata, sequence
            except Exception as e:
                logger.error(f"Error processing variant-interval")
                logger.error(f"Interval: {interval}")
                logger.error(f"Variant: {variant}")
                raise e

    def __len__(self):
        if self._genome_annotation is None or len(self._genome_annotation) == 0:
            return 0
        tmp_matcher = self._get_single_variant_matcher(vcf_lazy=False)
        total = sum(1 for _, _ in tmp_matcher)
        if self._size:
            return min(self._size, total)
        return total

    def _get_single_variant_matcher(self, vcf_lazy=True):
        if self._genome_annotation is None or len(self._genome_annotation) == 0:
            return iter([])
        # reads the genome annotation
        # start and end are transformed to 0-based and 1-based respectively
        roi = pr.PyRanges(self._genome_annotation)
        roi = roi.extend(ext={"5": self.variant_upstream_cse, "3": self.variant_downstream_cse})
        # todo do assert length of roi

        interval_attrs = ['gene_id', 'transcript_id', 'cse', 'transcript_start', 'transcript_end']
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
                ('seq_start', pa.int64()),
                ('seq_end', pa.int64()),
                ('cse', pa.int64()),
                ('chrom', pa.string()),
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
