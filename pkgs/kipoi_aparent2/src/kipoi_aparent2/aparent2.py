from __future__ import annotations

import pathlib
from kipoi_aparent2.dataloader import ApaDataloader
from kipoi_aparent2.dataloader.apa_annotation import APAAnnotation
from kipoi_aparent2.logger import logger
from keras.models import load_model
import pyarrow as pa
import math
import pyarrow.parquet as pq
from tqdm.autonotebook import tqdm
import numpy as np
import polars as pl
import re
from scipy.special import expit, logit

pl.Config.with_columns_kwargs = True

__all__ = ['Aparent2']


class Aparent2:

    def __init__(self, model_path: str | pathlib.Path):
        logger.debug(f'Loading model from {model_path}')
        self._model = load_model(str(model_path))

    def predict(self, dataloader: ApaDataloader, batch_size: int, num_cut_sites: int, filepath: str | pathlib.Path):
        """
        Predict on a dataloader and save the results in a parquet file
        :param filepath:
        :param dataloader:
        :param batch_size:
        :param num_cut_sites: The number of cut sites downstream of the CSE to consider for the variant effect calculation.
        :return: filepath to the parquet dataset
        """
        logger.debug('Predicting on dataloader')
        assert batch_size > 0

        schema = dataloader.pyarrow_metadata_schema
        schema = (schema.
                  insert(0, pa.field(f'cleavage_prob_narrow', pa.float32())).
                  insert(0, pa.field(f'cleavage_prob_full', pa.float32())))
        # position of the cse in the sequence
        cse_pos_index = int(schema.metadata[b'cse_pos_index'])

        batch_counter = 0
        total_batches = math.ceil(len(dataloader) / batch_size)
        if total_batches == 0:
            logger.info('The dataloader is empty. No predictions to make.')
            writer = pq.ParquetWriter(filepath, schema)
            writer.close()
            return

        with pq.ParquetWriter(filepath, schema) as writer:
            for batch in tqdm(dataloader.batch_iter(batch_size=batch_size), total=total_batches):
                batch_counter += 1
                batch = self._to_pyarrow(self._process_batch(batch, num_cut_sites, cse_pos_index))
                writer.write_batch(batch)

    def _process_batch(self, batch, num_cut_sites, cse_pos_index):
        """
        Process a batch of data. Run the model and prepare the results dict.
        :param batch: list of data dicts
        :param num_cut_sites: The number of cut sites downstream of the CSE to consider for the variant effect calculation.
        :return: Results dict. Structure: {'metadata': {field: [values]}, 'predictions': {sequence_key: [values]}}
        """
        batch_size = batch['sequence'].shape[0]
        sequences = batch['sequence'][:, None, :, :]

        # Always set this one-hot variable to 11 (training sub-library bias)
        lib = np.zeros((len(sequences), 13))
        lib[:, 11] = 1.

        assert sequences.shape == (batch_size, 1, 205, 4)
        assert lib.shape == (batch_size, 13)

        # run model
        predictions = self._model.predict_on_batch([sequences, lib])[1]
        assert predictions.shape == (batch_size, 206)

        # sum the cut site probabilities
        # after core-hexamer (e.g. AATAAA)
        cleavage_probability_narrow = np.sum(predictions[:, (cse_pos_index + 7):(cse_pos_index + 7 + num_cut_sites)],
                                             axis=1)
        cleavage_probability_full = np.sum(predictions[:, 0:205], axis=1)

        results = {
            'metadata': batch['metadata'],
            'cleavage_prob_narrow': cleavage_probability_narrow,
            'cleavage_prob_full': cleavage_probability_full
        }
        return results

    @staticmethod
    def _to_pyarrow(results: dict):
        """
        Convert the results dict from the _process_batch method to a pyarrow table and write it in a parquet file.
        :param results: pyarrow.RecordBatch object
        """
        logger.debug('Converting results to pyarrow')

        # format predictions
        metadata = {}
        for k, v in results['metadata'].items():
            v = pa.array(v.tolist())
            if isinstance(v, np.ndarray):
                v = v.tolist()
            metadata[k] = v

        formatted_results = {
            'cleavage_prob_full': pa.array(results['cleavage_prob_full'].tolist(), type=pa.float32()),
            'cleavage_prob_narrow': pa.array(results['cleavage_prob_narrow'].tolist(), type=pa.float32()),
            **metadata
        }

        logger.debug('Constructing pyarrow record batch')
        # construct RecordBatch
        return pa.RecordBatch.from_arrays(list(formatted_results.values()), names=list(formatted_results.keys()))


class Aparent2Veff:

    def __init__(self, apa_annotation: APAAnnotation | None = None):
        """
        Initialize the Aparent2Veff object.
        :param apa_annotation: An object that contains the genome annotation and isoform proportions.
        """
        if apa_annotation is not None:
            self.annotation_df = apa_annotation.get_annotation()
            self.isoform_usage_df = apa_annotation.get_isoform_usage().with_columns(
                transcript_id=pl.col('transcript_id').arr.join(';'))

    def run(self, ref_paths: list[str] | list[pathlib.Path], alt_path: str | pathlib.Path,
            output_path: str | pathlib.Path, aggregation_mode: str, upstream_cse: int | None = None,
            downstream_cse: int | None = None, use_narrow_score: bool = True):
        """
        Given a file containing enformer scores for alternative alleles, calculate the variant effect.
        Then aggregate the scores by gene, variant and tissue. Save the results in a parquet file.

        :param ref_paths: The parquet files that contains the reference scores
        :param alt_path: The parquet file that contains the alternate scores
        :param output_path: The parquet file that will contain the variant effect scores
        :param aggregation_mode: One of ['lor', 'delta_pdui'].
        :param upstream_cse: Variant effects outside of the interval [-upstream_cse, downstream_cse] will be set to 0.
        :param downstream_cse: Variant effects outside of the interval [-upstream_cse, downstream_cse] will be set to 0.
        :param use_narrow_score: If True, use the narrow score for the variant effect calculation.
        :return:
        """

        if aggregation_mode not in ['lor', 'delta_pdui']:
            raise ValueError(f'Unknown mode: {aggregation_mode}')
        elif aggregation_mode in ['delta_pdui']:
            assert self.annotation_df is not None, 'Genome annotation is required.'
            assert len(self.annotation_df) > 0, 'Genome annotation is empty.'
            assert self.isoform_usage_df is not None, 'Isoform proportions are required.'

        logger.debug(f'Calculating the variant effect for {alt_path}')

        if use_narrow_score:
            keep_score = 'cleavage_prob_narrow'
            exclude_score = 'cleavage_prob_full'
        else:
            keep_score = 'cleavage_prob_full'
            exclude_score = 'cleavage_prob_narrow'

        ref_ldf = pl.concat(
            [pl.scan_parquet(path).select(pl.exclude(exclude_score)).rename({keep_score: 'ref_score'}).
             with_columns(chrom=pl.lit(_extract_chromosome_from_parquet(path))) for path in ref_paths]
        )
        alt_ldf = pl.scan_parquet(alt_path).select(pl.exclude(exclude_score)).rename({keep_score: 'alt_score'})

        # check if alt_ldf is empty and write empty file if that's the case
        if alt_ldf.select(pl.count()).collect()[0, 0] == 0:
            logger.warning('The alternate scores are empty. No variant effect to calculate.')
            veff_ldf = (alt_ldf.select(['chrom', 'strand', 'gene_id', 'variant_start',
                                        'variant_end', 'ref', 'alt', ]).
                        with_columns(pl.Series('veff_score', [], dtype=pl.Float32)).
                        collect())
            veff_ldf.write_parquet(output_path)
            return

        on = ['chrom', 'strand', 'gene_id', 'transcript_id', 'pas_id', 'cse_pos', 'pas_pos', 'seq_start', 'seq_end']

        veff_ldf = alt_ldf.join(ref_ldf, how='left', on=on)
        veff_ldf = veff_ldf.select(['seq_start', 'seq_end', 'chrom', 'strand',
                                    'gene_id', 'transcript_id', 'pas_id', 'cse_pos', 'pas_pos',
                                    'variant_start', 'variant_end', 'ref', 'alt', 'ref_score', 'alt_score'])

        # remove gene and transcript versions
        veff_ldf = veff_ldf.with_columns(
            [pl.col('gene_id').str.replace(r'([^\.]+)\..+$', "${1}").alias('gene_id'),
             pl.col('transcript_id').str.replace(r'([^\.]+)\..+$', "${1}").alias('transcript_id')]
        )

        # calculate variant position relative to the cse
        if downstream_cse is not None or upstream_cse is not None:
            veff_ldf = veff_ldf.with_columns(
                (pl.when(pl.col('strand') == '+').then(
                    pl.col('variant_start') - pl.col('cse_pos')
                ).otherwise(
                    pl.col('cse_pos') - pl.col('variant_start'))
                ).alias('relative_pos'))

            if downstream_cse is not None:
                veff_ldf = veff_ldf.filter(pl.col('relative_pos') <= downstream_cse)
            if upstream_cse is not None:
                veff_ldf = veff_ldf.filter(pl.col('relative_pos') >= -upstream_cse)

        # calculate the log odds ratio (LOR)
        veff_ldf = veff_ldf.with_columns(lor=((pl.col('alt_score') / (1 - pl.col('alt_score'))) /
                                              (pl.col('ref_score') / (1 - pl.col('ref_score')))).log())
        veff_df = self._aggregate(veff_ldf, aggregation_mode)
        logger.debug(f'Writing the variant effect to {output_path}')
        veff_df.write_parquet(output_path)

    def _aggregate(self, veff_ldf, aggregation_mode: str):
        """
        Given a dataframe containing variant effect scores, aggregate the scores by gene, variant and tissue.

        :param veff_ldf: A polars dataframe containing the variant effect scores.
        :param aggregation_mode: One of ['lor', 'delta_pdui'].
        :return: A pandas dataframe containing the aggregated scores.
        """

        if aggregation_mode == 'lor':
            veff_ldf = veff_ldf.with_columns(abs_lor=pl.col('lor').abs()). \
                with_columns(max_abs_lor=pl.col('abs_lor').max().over(['chrom', 'strand', 'gene_id', 'variant_start',
                                                                       'variant_end', 'ref', 'alt', ]))
            # aggregate by gene and variant and keep the maximum absolute LOR
            veff_ldf = veff_ldf.groupby(['chrom', 'strand', 'gene_id', 'variant_start',
                                         'variant_end', 'ref', 'alt', ]).agg(
                pl.col(['transcript_id', 'pas_id', 'cse_pos', 'pas_pos', 'ref_score', 'alt_score', 'lor']).filter(
                    pl.col('abs_lor') == pl.col('max_abs_lor')).first()
            ).rename({'lor': 'veff_score'}).drop(['abs_lor', 'max_abs_lor'])
            veff_df = veff_ldf.collect()

        elif aggregation_mode == 'delta_pdui':
            pdui_ldf = veff_ldf.drop('transcript_id').join(self.isoform_usage_df.lazy(), on=['gene_id', 'pas_id'],
                                                           how='inner'). \
                rename({'isoform_proportion': 'pdui_ref'}). \
                with_columns(pdui_alt=expit(pl.col('lor') + logit(pl.col('pdui_ref')))). \
                with_columns(delta_pdui=pl.col('pdui_alt') - pl.col('pdui_ref'))
            veff_df = pdui_ldf.drop(['pdui_alt', 'lor']). \
                with_columns(abs_delta_pdui=pl.col('delta_pdui').abs()). \
                with_columns(max_abs_delta_pdui=pl.col('abs_delta_pdui').max().
                             over(['chrom', 'strand', 'gene_id', 'tissue', 'variant_start',
                                   'variant_end', 'ref', 'alt'])). \
                groupby(['chrom', 'strand', 'gene_id', 'tissue', 'variant_start',
                         'variant_end', 'ref', 'alt', ]). \
                agg(pl.col(['transcript_id', 'pas_id', 'cse_pos', 'pas_pos', 'ref_score',
                            'alt_score', 'delta_pdui', 'pdui_ref']). \
                    filter(pl.col('abs_delta_pdui') == pl.col('max_abs_delta_pdui')).first()). \
                rename({'delta_pdui': 'veff_score'}).drop(['abs_delta_pdui', 'max_abs_delta_pdui']).collect()
        else:
            # todo look for the effect on the distal APA site
            # distal_ldf = self.isoform_proportion_ldf.join(self.gtf_ldf, on=['gene_id', 'transcript_id'], how='left'). \
            #     rename({
            #     'transcript_id': 'distal_transcript_id', 'isoform_proportion': 'ref_pdui', 'cse': 'distal_cse'
            # }). \
            #     join(veff_ldf.rename({'transcript_id': 'proximal_transcript_id', 'cse': 'proximal_cse'}),
            #          on=['chrom', 'strand', 'gene_id']). \
            #     with_columns(
            #     distal_transcript_distance=pl.min([
            #         (pl.col('variant_start') - pl.col('distal_cse')).abs(),
            #         (pl.col('variant_end') - pl.col('distal_cse')).abs()
            #     ])
            # ).with_columns(
            #     lambda_=pl.when(pl.col('distal_transcript_distance') > 150).then(pl.lit(-1)).otherwise(pl.lit(1)))
            # distal_ldf.collect().sort(
            #     ['gene_id', 'variant_end', 'tissue', 'proximal_transcript_id', 'distal_transcript_distance',
            #      'distal_transcript_id']).select(
            #     ['gene_id', 'variant_end', 'tissue', 'proximal_transcript_id', 'distal_transcript_distance',
            #      'distal_transcript_id', 'lor', 'ref_pdui'])
            raise ValueError(f'Unknown mode: {aggregation_mode}')

        logger.debug(f'Aggregated table size: {len(veff_df)}')
        return veff_df


# this polars version does not support hive partitioning
# we extract the chromosome manually
def _extract_chromosome_from_parquet(file_path: str | pathlib.Path) -> str | None:
    # Use regular expression to extract the chromosome value
    match = re.search(r'/chrom=([^/]+)/data.parquet', str(file_path))
    if match:
        return match.group(1)
    else:
        return None
