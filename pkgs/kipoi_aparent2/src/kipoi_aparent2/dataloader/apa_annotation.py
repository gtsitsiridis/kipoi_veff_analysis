from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod

import pandas as pd
from kipoi_aparent2.utils import gtf_to_pandas
import polars as pl

pl.Config.with_columns_kwargs = True


class APAAnnotation(ABC):
    @abstractmethod
    def get_annotation(self):
        """
        For all transcripts in the genome annotation, extract the region around the cleavage site.
        :return: genome_annotation with additional columns cse_pos (0-based), pas_pos (0-based) and pas_id
        """
        raise NotImplementedError("The method is not implemented.")

    @abstractmethod
    def get_isoform_usage(self):
        raise NotImplementedError("The method is not implemented.")


class EnsemblAPAAnnotation(APAAnnotation):
    def __init__(self, gtf: pd.DataFrame | str, chromosome: str | None = None, protein_coding_only: bool = False,
                 canonical_only: bool = False, gene_ids: list | None = None,
                 isoform_usage_path: pathlib.Path | str | None = None):
        roi = get_roi_from_genome_annotation(gtf, chromosome, protein_coding_only, canonical_only, gene_ids)
        roi = pl.from_pandas(roi)
        self._annotation_df = pl.concat([
            roi.filter(pl.col('Strand') == '+').with_columns(pas_pos=pl.col('transcript_end') - 1). \
                groupby(['Chromosome', 'Strand', 'gene_id', 'pas_pos']).agg(
                pl.col('transcript_id')).with_columns(cse_pos=pl.col('pas_pos') - 30),
            roi.filter(pl.col('Strand') == '-').with_columns(pas_pos=pl.col('transcript_start')). \
                groupby(['Chromosome', 'Strand', 'gene_id', 'pas_pos']).agg(
                pl.col('transcript_id')).with_columns(cse_pos=pl.col('pas_pos') + 30)
        ]).with_columns(Start=pl.col('cse_pos'), End=pl.col('cse_pos') + 1,
                        pas_id=pl.col('Chromosome') + ':' + pl.col('pas_pos') + ':' + pl.col('Strand'))

        if isoform_usage_path is not None:
            self._isoform_proportion_df = (pl.read_csv(isoform_usage_path, sep='\t').
                                           select(['gene', 'transcript', 'tissue', 'median_transcript_proportions']).
                                           rename({'median_transcript_proportions': 'isoform_proportion',
                                                   'gene': 'gene_id', 'transcript': 'transcript_id'}).
                                           filter(~pl.col('isoform_proportion').is_null()))

    def get_annotation(self):
        return self._annotation_df

    def get_isoform_usage(self):
        if self._isoform_proportion_df is None:
            raise ValueError("No isoform usage data provided.")
        annot_df = self._annotation_df.explode('transcript_id').with_columns(
            gene_id=pl.col('gene_id').str.replace(r'([^\.]+)\..+$', "${1}"),
            transcript_id=pl.col('transcript_id').str.replace(r'([^\.]+)\..+$', "${1}")
        )
        return self._isoform_proportion_df.join(annot_df, on=['transcript_id', 'gene_id'], how='left'). \
            groupby(['Chromosome', 'Strand', 'gene_id', 'pas_pos', 'pas_id', 'cse_pos', 'tissue']).agg(
            [pl.col('transcript_id'), pl.col('isoform_proportion').sum()])


def get_roi_from_genome_annotation(gtf: pd.DataFrame | str, chromosome: str | None = None,
                                   protein_coding_only: bool = False, canonical_only: bool = False,
                                   gene_ids: list | None = None):
    """
    Get ROI from genome annotation
    :return: filtered genome_annotation
    """
    if not isinstance(gtf, pd.DataFrame):
        genome_annotation = gtf_to_pandas(gtf)
    else:
        genome_annotation = gtf.copy()
    if gene_ids is not None:
        genome_annotation = genome_annotation[genome_annotation['gene_id'].str.contains('|'.join(gene_ids))]
    if chromosome is not None:
        genome_annotation = genome_annotation.query("`Chromosome` == @chromosome")
    roi = genome_annotation.query("`Feature` == 'transcript'")
    if protein_coding_only:
        roi = roi.query("`gene_type` == 'protein_coding'")
    if canonical_only:
        # check if Ensembl_canonical is in the set of tags
        roi = roi[roi['tag'].apply(lambda x: False if pd.isna(x) else ('Ensembl_canonical' in x.split(',')))]
    if len(roi) > 0:
        roi = roi.assign(
            transcript_start=roi["Start"],
            transcript_end=roi["End"],
        )

    return roi[['Chromosome', 'Start', 'End', 'Strand', 'gene_id', 'gene_type', 'gene_name',
                'transcript_id', 'transcript_type', 'transcript_name', 'transcript_start', 'transcript_end']]
