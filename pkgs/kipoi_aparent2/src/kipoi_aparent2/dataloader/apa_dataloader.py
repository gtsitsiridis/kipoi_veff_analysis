from __future__ import annotations

import pandas as pd
from kipoiseq.extractors import VariantSeqExtractor, SingleVariantMatcher
import pyranges as pr
from kipoiseq.extractors import MultiSampleVCF
import pyarrow as pa
from .dataloader import extract_sequence_around_anchor
from .apa_annotation import EnsemblAPAAnnotation, APAAnnotation
from kipoi_aparent2.constants import AlleleType
from kipoi_aparent2.logger import logger
from .dataloader import Dataloader

__all__ = ['ApaDataloader', 'RefApaDataloader', 'VCFApaDataloader']

# length of sequence which APARENT2 gets as input
# ═════┆═════┆════════════════════════┆═════┆═════
APARENT2_SEQUENCE_LENGTH = 205
CSE_POS_INDEX = 70  # CSE should be roughly around position 70 of the sequence. (zero-based)


class ApaDataloader(Dataloader):
    def __init__(self, allele_type: AlleleType, fasta_file, apa_annotation: APAAnnotation,
                 seq_length: int = APARENT2_SEQUENCE_LENGTH, cse_pos_index: int = CSE_POS_INDEX, size: int = None,
                 *args, **kwargs):
        """

        :param fasta_file: Fasta file with the reference genome
        :param apa_annotation: Genome annotation with the APA sites
        :param seq_length: The length of the sequence to return.
        :param cse_pos_index: The position of the CSE in the sequence. (0-based)
        :param size: The number of samples to return. If None, all samples are returned.
        """

        super().__init__(fasta_file=fasta_file, size=size, *args, **kwargs)
        assert 0 <= cse_pos_index < seq_length, f"cse_pos_index must be between 0 and seq_length but got {cse_pos_index}"

        self._seq_length = seq_length
        logger.debug(f"Loading genome annotation")
        self._apa_annotation = apa_annotation.get_annotation().to_pandas()
        self._shift = seq_length // 2 - cse_pos_index
        self.metadata = {'cse_pos_index': str(cse_pos_index), 'allele_type': allele_type.value,
                         'seq_length': str(self._seq_length)}

    @classmethod
    def from_allele_type(cls, allele_type: AlleleType, *args, **kwargs):
        if allele_type == AlleleType.REF:
            return RefApaDataloader(*args, **kwargs)
        elif allele_type == AlleleType.ALT:
            return VCFApaDataloader(*args, **kwargs)
        else:
            raise ValueError(f"Unknown allele type: {allele_type}")


class RefApaDataloader(ApaDataloader):
    def __init__(self, fasta_file, apa_annotation: APAAnnotation, seq_length: int = APARENT2_SEQUENCE_LENGTH,
                 cse_pos_index: int = CSE_POS_INDEX, size: int = None,
                 *args, **kwargs):
        """
        :param fasta_file: Fasta file with the reference genome
        :param apa_annotation: Genome annotation with the APA sites
        :param seq_length: The length of the sequence to return.
        :param cse_pos_index: The position of the CSE in the sequence. (0-based)
        :param size: The number of samples to return. If None, all samples are returned.
        """
        super().__init__(AlleleType.REF, fasta_file=fasta_file, apa_annotation=apa_annotation,
                         seq_length=seq_length, cse_pos_index=cse_pos_index, size=size, *args, **kwargs)

    def _sample_gen(self):
        for _, row in self._apa_annotation.iterrows():
            try:
                chromosome = row['Chromosome']
                strand = row.get('Strand', '.')
                cse = row['cse_pos']

                sequence, interval = extract_sequence_around_anchor(self._shift, chromosome, strand, cse,
                                                                    self._seq_length,
                                                                    ref_seq_extractor=self._reference_sequence)

                metadata = {
                    "seq_start": interval.start,  # 0-based start of the input sequence
                    "seq_end": interval.end + 1,  # 1-based stop of the input sequence
                    "pas_pos": row['pas_pos'],  # 0-based position of the PAS
                    "cse_pos": cse,  # 0-based position of the CSE
                    "strand": strand,
                    "pas_id": row['pas_id'],
                    "gene_id": row['gene_id'],
                    "transcript_id": ';'.join(row['transcript_id']),
                }

                yield metadata, sequence
            except Exception as e:
                logger.error(f"Error processing row: {row}")
                raise e

    def __len__(self):
        if self._apa_annotation is None:
            return 0
        return len(self._apa_annotation) if self._size is None else min(self._size, len(self._apa_annotation))

    @property
    def pyarrow_metadata_schema(self):
        """
        Get the pyarrow schema for the metadata.
        :return: PyArrow schema for the metadata
        """
        columns = [
            ('seq_start', pa.int64()),
            ('seq_end', pa.int64()),
            ('pas_pos', pa.int64()),
            ('cse_pos', pa.int64()),
            ('strand', pa.string()),
            ('pas_id', pa.string()),
            ('gene_id', pa.string()),
            ('transcript_id', pa.string()), ]

        return pa.schema(columns, metadata=self.metadata)


class VCFApaDataloader(ApaDataloader):
    def __init__(self, fasta_file, apa_annotation: APAAnnotation, vcf_file, vcf_lazy=True,
                 variant_upstream_cse: int = 10, variant_downstream_cse: int = 10,
                 seq_length: int = APARENT2_SEQUENCE_LENGTH, cse_pos_index: int = CSE_POS_INDEX, size: int = None,
                 *args, **kwargs):
        """

        :param fasta_file: Fasta file with the reference genome
        :param apa_annotation: Genome annotation with the APA sites
        :param vcf_file: VCF file with variants
        :param vcf_lazy: If True, the VCF file is read lazily
        :param variant_upstream_cse: The number of bases upstream the CSE to look for variants
        :param variant_downstream_cse: The number of bases downstream the CSE to look for variants
        :param seq_length: The length of the sequence to return.
        :param cse_pos_index: The position of the CSE in the sequence. (0-based)
        :param size: The number of samples to return. If None, all samples are returned.
        """

        super().__init__(AlleleType.ALT, fasta_file=fasta_file, apa_annotation=apa_annotation, seq_length=seq_length,
                         cse_pos_index=cse_pos_index, size=size, *args, **kwargs)
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
                cse = attrs['cse_pos']
                chromosome = interval.chrom
                strand = interval.strand

                sequence, interval = extract_sequence_around_anchor(self._shift, chromosome, strand, cse,
                                                                    self._seq_length,
                                                                    ref_seq_extractor=self._reference_sequence,
                                                                    variant_extractor=self._variant_seq_extractor,
                                                                    variant=variant)
                metadata = {
                    "seq_start": interval.start,  # 0-based start of the input sequence
                    # TODO this is wrong
                    # Fast extractor is treating the interval end as 1-based, even though it is 0-based
                    # So the following line should be: "seq_end": interval.end
                    "seq_end": interval.end + 1,  # 1-based stop of the input sequence
                    "pas_pos": attrs['pas_pos'],  # 0-based position of the PAS
                    "cse_pos": cse,  # 0-based position of the CSE
                    "chrom": interval.chrom,
                    "strand": interval.strand,
                    "pas_id": attrs['pas_id'],
                    "gene_id": attrs['gene_id'],
                    "transcript_id": ';'.join(attrs['transcript_id']),
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
        if self._apa_annotation is None or len(self._apa_annotation) == 0:
            return 0
        tmp_matcher = self._get_single_variant_matcher(vcf_lazy=False)
        total = sum(1 for _, _ in tmp_matcher)
        if self._size:
            return min(self._size, total)
        return total

    def _get_single_variant_matcher(self, vcf_lazy=True):
        if self._apa_annotation is None or len(self._apa_annotation) == 0:
            return iter([])
        # reads the genome annotation
        # start and end are transformed to 0-based and 1-based respectively
        roi = pr.PyRanges(self._apa_annotation)
        roi = roi.extend(ext={"5": self.variant_upstream_cse, "3": self.variant_downstream_cse})
        # todo do assert length of roi

        interval_attrs = ['gene_id', 'transcript_id', 'pas_id', 'cse_pos', 'pas_pos']
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
                ('pas_pos', pa.int64()),
                ('cse_pos', pa.int64()),
                ('chrom', pa.string()),
                ('strand', pa.string()),
                ('pas_id', pa.string()),
                ('gene_id', pa.string()),
                ('transcript_id', pa.string()),
                ('variant_start', pa.int64()),
                ('variant_end', pa.int64()),
                ('ref', pa.string()),
                ('alt', pa.string()),
            ], metadata=self.metadata)
