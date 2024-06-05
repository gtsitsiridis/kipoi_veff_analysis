import pathlib
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from .dataloader import TSSDataloader
from .utils import RandomModel
from kipoi_enformer.logger import logger
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.autonotebook import tqdm
from collections import defaultdict
import math
import yaml
import pickle
import polars as pl
from scipy.special import logsumexp

__all__ = ['Enformer', 'EnformerTissueMapper', 'calculate_veff', 'aggregate_veff']

# Enformer model URI
MODEL_PATH = 'https://tfhub.dev/deepmind/enformer/1'


class Enformer:
    # length of the bins in the enformer model
    BIN_SIZE = 128
    NUM_PREDICTION_BINS = 896
    NUM_SEEN_BINS = 1536
    # length of central sequence for which enformer gives predictions (896 bins)
    # ─────┆─────┆════════════════════════┆─────┆─────
    PRED_SEQUENCE_LENGTH = NUM_PREDICTION_BINS * BIN_SIZE
    # length of central sequence which enformer actually sees (1536 bins)
    # ─────┆═════┆════════════════════════┆═════┆─────
    SEEN_SEQUENCE_LENGTH = NUM_SEEN_BINS * BIN_SIZE

    def __init__(self, is_random: bool = False):
        """
        :param is_random: If True, load a random model for testing purposes.
        """
        if not is_random:
            logger.debug(f'Loading model from {MODEL_PATH}')
            self._model = hub.load(MODEL_PATH).model
        else:
            self._model = RandomModel()

    def predict(self, dataloader: TSSDataloader, batch_size: int, filepath: str | pathlib.Path,
                num_output_bins=NUM_PREDICTION_BINS):
        """
        Predict on a dataloader and save the results in a parquet file
        :param num_output_bins: The number of bins to extract from enformer's output
        :param filepath:
        :param dataloader:
        :param batch_size:
        :return: filepath to the parquet dataset
        """
        logger.debug('Predicting on dataloader')
        assert batch_size > 0

        # Hint: order matters
        schema = dataloader.pyarrow_metadata_schema
        schema = schema.insert(0, pa.field(f'tracks', pa.list_(pa.list_(pa.list_(pa.float32())))))

        shifts = [int(x) for x in schema.metadata[b'shifts'].split(b';')]
        max_abs_shift = max([abs(shift) for shift in shifts])
        assert math.ceil(max_abs_shift / self.BIN_SIZE) < num_output_bins <= self.NUM_PREDICTION_BINS, \
            f'num_output_bins must be fit the maximum shift and be at most {self.NUM_PREDICTION_BINS}'

        batch_counter = 0
        total_batches = math.ceil(len(dataloader) / batch_size)
        if total_batches == 0:
            logger.info('The dataloader is empty. No predictions to make.')
            return

        with pq.ParquetWriter(filepath, schema) as writer:
            for batch in tqdm(dataloader.batch_iter(batch_size=batch_size), total=total_batches):
                batch_counter += 1
                batch = self._to_pyarrow(self._process_batch(batch, num_output_bins=num_output_bins))
                writer.write_batch(batch)

        # sanity check for the dataloader
        assert batch_counter == total_batches

    def _process_batch(self, batch, num_output_bins=11):
        """
        Process a batch of data. Run the model and prepare the results dict.
        :param batch: list of data dicts
        :return: Results dict. Structure: {'metadata': {field: [values]}, 'predictions': {sequence_key: [values]}}
        """
        batch_size = batch['sequences'].shape[0]
        seqs_per_record = batch['sequences'].shape[1]
        sequences = np.reshape(batch['sequences'], (batch_size * seqs_per_record, 393_216, 4))

        # create input tensor
        input_tensor = tf.convert_to_tensor(sequences)
        assert input_tensor.shape == (batch_size * seqs_per_record, 393_216, 4)

        # run model
        predictions = self._model.predict_on_batch(input_tensor)['human'].numpy()
        assert predictions.shape == (batch_size * seqs_per_record, 896, 5313)
        predictions = predictions.reshape(batch_size, seqs_per_record, 896, 5313)

        # extract central bins if the number of output bins is different from the number of prediction bins
        if num_output_bins != self.NUM_PREDICTION_BINS:
            # calculate TSS bin
            tss_bin = (self.PRED_SEQUENCE_LENGTH // 2 + 1) // self.BIN_SIZE
            bins = [tss_bin + i for i in range(-math.floor(num_output_bins / 2), math.ceil(num_output_bins / 2))]
            assert len(bins) == num_output_bins
            predictions = predictions[:, :, bins, :]

        results = {
            'metadata': batch['metadata'],
            'tracks': predictions
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
            'tracks': pa.array(results['tracks'].tolist(), type=pa.list_(pa.list_(pa.list_(pa.float32())))),
            **metadata
        }

        logger.debug('Constructing pyarrow record batch')
        # construct RecordBatch
        return pa.RecordBatch.from_arrays(list(formatted_results.values()), names=list(formatted_results.keys()))


class EnformerTissueMapper:
    def __init__(self, tracks_path: str | pathlib.Path, tissue_matcher_path: str | pathlib.Path | None = None):
        """
        :param tracks_path: A yaml file mapping the name of the tracks to the index in the predictions.
        Only the tracks in the file are considered for the mapping.
        :param tissue_matcher_path: A pickle containing a dictionary of linear models for each GTEx tissue.
        """
        self.tissue_matcher_lm_dict = None
        # If tissue_matcher_path is not None, load the linear models
        if tissue_matcher_path is not None:
            with open(tissue_matcher_path, 'rb') as f:
                self.tissue_matcher_lm_dict = {k: v['ingenome'] for k, v in pickle.load(f).items()}

        with open(tracks_path, 'rb') as f:
            self.tracks_dict = yaml.safe_load(f)

    def predict(self, prediction_path: str | pathlib.Path, output_path: str | pathlib.Path, num_bins: int = 3):
        """
        Load the predictions from the parquet file lazily.
        For each record, calculate the average predictions over the bins centered at the tss bin.
        For each tissue in the tissue_matcher_lm_dict, predict a tissue-specific expression score.
        Finally, save the expression scores in a new parquet file.

        :param prediction_path: The parquet file that contains the enformer predictions.
        :param output_path: The parquet file that will contain the tissue-specific expression scores.
        :param num_bins: number of bins to average over for each record.

        The average predictions will be calculated at the tss bin of each record.
        """
        if self.tissue_matcher_lm_dict is None:
            raise ValueError('The tissue_matcher_lm_dict is not provided. Please train the linear models first.')

        prediction_file = pq.ParquetFile(prediction_path)
        prediction_schema = prediction_file.schema.to_arrow_schema()
        output_schema = prediction_schema.remove(prediction_schema.get_field_index('tracks'))
        output_schema = pa.unify_schemas([output_schema, pa.schema([
            pa.field('tissue', pa.string()),
            pa.field('score', pa.float64()),
        ])])
        metadata = output_schema.metadata
        metadata['nbins'] = str(num_bins)
        output_schema = output_schema.with_metadata(metadata)
        shifts = [int(x) for x in metadata[b'shifts'].split(b';')]
        tracks = list(self.tracks_dict.values())

        logger.debug(f'Iterating over the parquet files in {prediction_path}')
        with pq.ParquetWriter(output_path, output_schema) as writer:
            for i in tqdm(range(prediction_file.num_row_groups)):
                batch_agg_pred, batch_meta = self._aggregate_batch(pl.from_arrow(prediction_file.read_row_group(i)),
                                                                   Enformer.BIN_SIZE, num_bins, shifts, tracks)
                logger.debug('Running model on the batch...')
                record_batch = self._predict_batch(batch_agg_pred, batch_meta)
                logger.debug('Writing to file')
                writer.write(record_batch)

    @staticmethod
    def _aggregate_batch(frame, bin_size, num_bins, shifts, tracks):
        """
        Aggregate the predictions for a record.
        :param frame:
        :param bin_size:
        :param num_bins:
        :param tracks:
        :return:
        """
        logger.debug('Aggregating the predictions for a batch')

        pred = np.stack(frame['tracks'].to_list())
        pred_seq_length = bin_size * pred.shape[2]
        agg_pred = []
        for shift_i, shift in enumerate(shifts):
            # estimate the tss bin
            # todo verify this calculation
            tss_bin = (pred_seq_length // 2 + 1 - shift) // bin_size
            # get num_bins - 1 neighboring bins centered at tss bin
            bins = [tss_bin + i for i in range(-math.floor(num_bins / 2), math.ceil(num_bins / 2))]
            assert len(bins) == num_bins
            agg_pred.append(pred[:, shift_i, bins, :][:, :, tracks])

        agg_pred = np.stack(agg_pred).swapaxes(0, 1)

        assert agg_pred.shape == (len(frame), len(shifts), num_bins, len(tracks))
        # average over shifts and bins
        agg_pred = agg_pred.mean(axis=(1, 2))

        assert agg_pred.shape == (len(frame), len(tracks),)
        metadata = frame.select([k for k in frame.columns if k != 'tracks']).to_dict(as_series=False)
        return agg_pred, metadata

    def _predict_batch(self, batch_agg_pred, batch_meta):
        """
        Map the enformer predictions to the GTEx tissues using the linear models.
        :param batch_agg_pred: The averaged predictions.
        :param batch_meta: The metadata for each record.
        :return:
        """
        batch_agg_pred = np.log10(batch_agg_pred + 1)
        tissue_scores = defaultdict()
        for tissue, lm in self.tissue_matcher_lm_dict.items():
            tissue_scores[tissue] = lm.predict(batch_agg_pred)

        scores = []
        tissues = []
        meta = defaultdict(list)
        for tissue in self.tissue_matcher_lm_dict.keys():
            tissues.extend([tissue] * len(batch_agg_pred))
            scores.extend(tissue_scores[tissue].tolist())
            for k, v in batch_meta.items():
                meta[k].extend(v)

        logger.debug('Constructing pyarrow record batch')
        values = [pa.array(v) for v in meta.values()] + [pa.array(tissues),
                                                         pa.array(scores), ]
        names = [k for k in meta.keys()] + ['tissue', 'score']

        # write to parquet
        return pa.RecordBatch.from_arrays(values, names=names)


def calculate_veff(ref_path, alt_path, output_path):
    """
    Given a file containing enformer scores for alternative alleles,
     1) load the corresponding reference chromosomes into memory,
     2) iterate over the alternative allele in batches,
     3) join the reference and alternative scores on the chromosome and position,
     and 4) calculate the variant effect by subtracting the reference scores from the alternate scores.

    :param ref_path: The parquet files that contains the reference scores
    :param alt_path: The parquet file that contains the alternate scores
    :param output_path: Whether the model scores are already log transformed
    :return:
    """

    logger.debug(f'Calculating the variant effect for {alt_path}')

    ref_df = pl.scan_parquet(ref_path).rename({'score': 'ref_score'})
    alt_df = pl.scan_parquet(alt_path).rename({'score': 'alt_score'})
    alt_metadata = pq.ParquetFile(alt_path).schema_arrow.metadata
    del alt_metadata[b'allele_type']

    on = ['tss', 'chrom', 'strand', 'gene_id', 'transcript_id', 'transcript_start', 'transcript_end',
          'enformer_start', 'enformer_end', 'tissue']

    joined_df = alt_df.join(ref_df, how='left', on=on)
    joined_df = joined_df.with_columns(((pl.col("alt_score") - pl.col("ref_score")) / np.log10(2)).alias('log2fc'))
    joined_df = joined_df.select(['enformer_start', 'enformer_end', 'tss', 'chrom', 'strand',
                                  'gene_id', 'transcript_id', 'transcript_start', 'transcript_end',
                                  'variant_start', 'variant_end', 'ref', 'alt', 'tissue',
                                  'ref_score', 'alt_score', 'log2fc'])
    joined_df = joined_df.collect().to_arrow()
    joined_df = joined_df.replace_schema_metadata(alt_metadata)
    logger.debug(f'Writing the variant effect to {output_path}')
    pq.write_table(joined_df, output_path)


def aggregate_veff(veff_path: str | pathlib.Path, output_path: str | pathlib.Path,
                   isoforms_path: str | pathlib.Path | None = None, mode='logsumexp'):
    """
    Given a file containing variant effect scores, aggregate the scores by gene, variant and tissue.
    If the isoform proportions per tissue are given, then calculate the weighted sum. Otherwise, calculate the average.

    :param veff_path: The parquet file that contains the variant effect scores
    :param isoforms_path: The file containing the isoform proportions per tissue
    :param output_path: The parquet file that will contain the aggregated scores
    :param mode: One of ['logsum', 'mean']. If 'logsum', calculate the logsumexp of the scores.
    If 'mean', calculate the mean.
    """

    veff_metadata = pq.ParquetFile(veff_path).schema_arrow.metadata

    veff_ldf = pl.scan_parquet(veff_path).filter(pl.col('log2fc').is_not_null())
    veff_ldf = veff_ldf.with_columns(pl.col('gene_id').str.replace(r'([^\.]+)\..+$', "${1}").alias('gene_id'))
    veff_ldf = veff_ldf.with_columns(
        pl.col('transcript_id').str.replace(r'([^\.]+)\..+$', "${1}").alias('transcript_id'))
    if isoforms_path is not None:
        isoform_proportion_ldf = (pl.scan_csv(isoforms_path, separator='\t').
                                  select(['gene', 'transcript', 'tissue', 'median_transcript_proportions']).
                                  rename({'median_transcript_proportions': 'isoform_proportion',
                                          'gene': 'gene_id', 'transcript': 'transcript_id'}).
                                  filter(~pl.col('isoform_proportion').is_null()))
        joined_df = (veff_ldf.join(isoform_proportion_ldf, on=['gene_id', 'tissue', 'transcript_id'], how='left').
                     with_columns('isoform_proportion').fill_null(0))
    else:
        # assign equal weight to each transcript, if no isoform proportions are given
        joined_df = veff_ldf.with_columns(
            (1 / pl.len()).over(['chrom', 'strand', 'gene_id', 'variant_start', 'variant_end', 'ref', 'alt', 'tissue']).
            alias('isoform_proportion'))

    def logsumexp_udf(score, weight):
        return logsumexp(score / math.log10(math.e), b=weight) / math.log(10)

    if mode == 'logsumexp':
        joined_df = joined_df. \
            group_by(['chrom', 'strand', 'gene_id', 'variant_start', 'variant_end', 'ref', 'alt', 'tissue']). \
            agg(
            pl.struct(['ref_score', 'isoform_proportion']).
            map_elements(lambda x: logsumexp_udf(x.struct.field('ref_score'), x.struct.field('isoform_proportion')),
                         return_dtype=pl.Float64()).
            alias('ref_score'),
            pl.struct(['alt_score', 'isoform_proportion']).
            map_elements(lambda x: logsumexp_udf(x.struct.field('alt_score'), x.struct.field('isoform_proportion')),
                         return_dtype=pl.Float64()).
            alias('alt_score'),
        )
        joined_df = joined_df.with_columns(((pl.col("alt_score") - pl.col("ref_score")) / np.log10(2)).alias('log2fc')). \
            fill_nan(0)
    elif mode == 'mean':
        joined_df = (joined_df.
        group_by(['chrom', 'strand', 'gene_id', 'variant_start', 'variant_end', 'ref', 'alt', 'tissue']).
        agg(
            (pl.col('isoform_proportion') * (pl.col('alt_score') - pl.col('ref_score'))).sum().alias('log2fc'),
        ))
    else:
        raise ValueError(f'Unknown mode: {mode}')

    joined_df = joined_df.collect().to_arrow()
    logger.debug(f'Aggregated table size: {len(joined_df)}')
    joined_df = joined_df.replace_schema_metadata(veff_metadata)
    logger.debug(f'Writing the variant effect to {output_path}')
    pq.write_table(joined_df, output_path)
