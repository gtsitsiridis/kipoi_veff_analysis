from kipoi.data import SampleIterator
from kipoiseq import Interval, Variant
from kipoiseq.transforms import OneHot
from kipoiseq.extractors import VariantSeqExtractor, SingleVariantMatcher, BaseExtractor, FastaStringExtractor
from kipoi.metadata import GenomicRanges

import pandas as pd
import pyranges as pr

from kipoiseq.variant_source import VariantFetcher


class Enformer_DL(SampleIterator):
    def __init__(
            self,
            regions_of_interest: pr.PyRanges,
            reference_sequence: BaseExtractor,
            variants: VariantFetcher,
            interval_attrs=('gene_id', 'transcript_id'),
            upstream_tss: int = 10, downstream_tss: int = 10
    ):
        self.regions_of_interest = regions_of_interest
        self.reference_sequence = reference_sequence
        self.variants = variants
        self.interval_attrs = interval_attrs
        self.upstream_tss = upstream_tss
        self.downstream_tss = downstream_tss

        if not self.reference_sequence.use_strand:
            raise ValueError(
                "Reference sequence fetcher does not use strand but this is needed to obtain correct sequences!")
        self.variant_seq_extractor = VariantSeqExtractor(reference_sequence=reference_sequence)

        self.matcher = SingleVariantMatcher(
            variant_fetcher=self.variants,
            pranges=self.regions_of_interest,
            interval_attrs=interval_attrs
        )

        self.one_hot = OneHot()

    def __iter__(self):
        # todo onehot encode the sequences
        interval: Interval
        variant: Variant
        for interval, variant in self.matcher:
            yield {
                "inputs": {
                    "ref_seq": (  # self.one_hot(
                        self.reference_sequence.extract(interval)
                    ),
                    "alt_seq": (  # self.one_hot(
                        self.variant_seq_extractor.extract(
                            interval,
                            [variant],
                            anchor=interval.start + self.upstream_tss if not interval.neg_strand
                            else interval.end - self.downstream_tss
                        )),
                },
                "metadata": {
                    "variant": {
                        "chrom": variant.chrom,
                        "start": variant.start,
                        "end": variant.end,
                        "ref": variant.ref,
                        "alt": variant.alt,
                        "id": variant.id,
                        "str": str(variant),
                    },
                    "ranges": GenomicRanges.from_interval(interval),
                    **{k: interval.attrs.get(k, '') for k in self.interval_attrs},
                }
            }


def get_roi_from_transcript(transcript_start: int, transcript_end: int, is_on_negative_strand: bool,
                            upstream_tss: int, downstream_tss: int) -> (int, int):
    """
    Get region-of-interest for Enformer in relation to the TSS of a transcript
    :param upstream_tss: number of base pairs upstream of the TSS
    :param downstream_tss: number of base pairs downstream of the TSS
    :param transcript_start: 0-based start position of the transcript
    :param transcript_end: 1-based end position of the transcript
    :param is_on_negative_strand: is the gene on the negative strand?
    :return: Tuple of (start-0-based, end-1-based) position for the region of interest
    """
    if is_on_negative_strand:
        tss = transcript_end
        # convert 1-based to 0-based
        tss -= 1

        start = tss - downstream_tss
        end = tss + upstream_tss
        # convert 0-based to 1-based
        end += 1
    else:
        tss = transcript_start

        start = tss - upstream_tss
        end = tss + downstream_tss
        # convert 0-based to 1-based
        end += 1

    return start, end


def get_roi_from_genome_annotation(genome_annotation: pd.DataFrame, upstream_tss: int, downstream_tss: int):
    """
    Get region-of-interest for Enformer from some genome annotation
        :param upstream_tss: number of base pairs upstream of the TSS
    :param downstream_tss: number of base pairs downstream of the TSS
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
        start, end = get_roi_from_transcript(row.Start, row.End, row.Strand == '-', upstream_tss, downstream_tss)
        row.Start = start
        row.End = end

        return row

    roi = roi.apply(adjust_row, axis=1)

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
            downstream_tss: int = 10
    ):
        # reads the genome annotation
        # start and end are transformed to 0-based and 1-based respectively
        genome_annotation = pr.read_gtf(gtf_file, as_df=True)
        roi = get_roi_from_genome_annotation(genome_annotation, upstream_tss=upstream_tss,
                                             downstream_tss=downstream_tss)
        roi = pr.PyRanges(roi)

        from kipoiseq.extractors import MultiSampleVCF
        super().__init__(
            regions_of_interest=roi,
            reference_sequence=FastaStringExtractor(fasta_file, use_strand=True),
            variants=MultiSampleVCF(vcf_file, lazy=vcf_lazy),
            upstream_tss=upstream_tss, downstream_tss=downstream_tss
        )
