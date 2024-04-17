import pathlib
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from .dataloader import TSSDataloader
from kipoi_enformer.logger import logger
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.autonotebook import tqdm
from collections import defaultdict
import math
import yaml
import pickle
import polars as pl

__all__ = ['Enformer', 'EnformerTissueMapper']

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

    def __init__(self, model: str | tf.keras.Model = MODEL_PATH):
        """
        :param model: path to model or tf.keras.Model
        """
        if isinstance(model, str):
            self._model = hub.load(model).model
        else:
            self._model = model

    def predict(self, dataloader: TSSDataloader, batch_size: int, filepath: str | pathlib.Path):
        """
        Predict on a dataloader and save the results in a parquet file
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

        batch_counter = 0
        total_batches = math.ceil(len(dataloader) / batch_size)
        with pq.ParquetWriter(filepath, schema) as writer:
            for batch in tqdm(dataloader.batch_iter(batch_size=batch_size), total=total_batches):
                batch_counter += 1
                batch = self._to_pyarrow(self._process_batch(batch))
                writer.write_batch(batch)

        # sanity check for the dataloader
        assert batch_counter == total_batches

    def _process_batch(self, batch):
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
        tissue_matcher_lm_dict = None
        # If tissue_matcher_path is not None, load the linear models
        if tissue_matcher_path is not None:
            with open(tissue_matcher_path, 'rb') as f:
                self.tissue_matcher_lm_dict = {k: v['ingenome'] for k, v in pickle.load(f).items()}

        with open(tracks_path, 'rb') as f:
            self.tracks_dict = yaml.safe_load(f)

    def predict(self, prediction_path: str | pathlib.Path, output_path: str | pathlib.Path, num_bins: int = 3,
                batch_size: int = 1):
        """
        Load the predictions from the parquet file lazily.
        For each record, calculate the average predictions over the bins centered at the tss bin.
        For each tissue in the tissue_matcher_lm_dict, predict a tissue-specific expression score.
        Finally, save the expression scores in a new parquet file.

        :param prediction_path: The parquet file that contains the enformer predictions.
        :param batch_size: The number of records to read and write at once.
        :param output_path: The parquet file that will contain the tissue-specific expression scores.
        :param num_bins: number of bins to average over for each record
        The average predictions will be calculated at the tss bin of each record.
        """
        if self.tissue_matcher_lm_dict is None:
            raise ValueError('The tissue_matcher_lm_dict is not provided. Please train the linear models first.')

        prediction_schema = pq.read_metadata(prediction_path).schema.to_arrow_schema()
        output_schema = prediction_schema.remove(prediction_schema.get_field_index('tracks'))
        output_schema = pa.unify_schemas([output_schema, pa.schema([
            pa.field('tissue', pa.string()),
            pa.field('score', pa.float64())
        ])])

        with pq.ParquetWriter(output_path, output_schema) as writer:
            num_rows = pq.read_metadata(prediction_path).num_rows
            total_batches = math.ceil(num_rows / batch_size)
            batch_counter = 0
            logger.debug(f'Iterating over the batches in the parquet file {prediction_path}')
            logger.debug(f'Writing the veff values to the parquet file {output_path}')
            for batch_agg_pred, batch_meta in tqdm(
                    self._batch_iterator(prediction_path, num_bins, batch_size=batch_size), total=total_batches):
                batch_counter += 1
                logger.debug('Running model on the batch...')
                record_batch = self._predict_batch(batch_agg_pred, batch_meta)
                logger.debug('Writing to file')
                writer.write(record_batch)

            # sanity check
            assert batch_counter == total_batches

    def _batch_iterator(self, prediction_path: str | pathlib.Path, num_bins: int = 3, batch_size: int = 1):
        """
        Load the predictions from the parquet file lazily.
        :param prediction_path: The parquet file that contains the enformer predictions.
        :param num_bins: The number of bins to average over for each record.
        :param batch_size: The number of records to load at once.
        :return:
        """
        bin_size = Enformer.BIN_SIZE
        pred_seq_length = Enformer.PRED_SEQUENCE_LENGTH
        tracks = list(self.tracks_dict.values())

        df = pl.read_parquet(prediction_path)
        for frame in df.iter_slices(n_rows=batch_size):
            yield self._aggregate_batch(frame, pred_seq_length,
                                        bin_size, num_bins, tracks)

    @staticmethod
    def _aggregate_batch(frame, pred_seq_length, bin_size, num_bins, tracks):
        """
        Aggregate the predictions for a record.
        :param frame:
        :param pred_seq_length:
        :param bin_size:
        :param num_bins:
        :param tracks:
        :return:
        """
        logger.debug('Aggregating the predictions for a batch')

        pred = np.stack(frame['tracks'].to_list())
        agg_pred = []
        # we assume that all records have the same shifts
        shifts = frame[0, 'shifts']
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
        # calculate veff
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
