import pathlib
import pickle

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from .dataloader import VCFEnformerDL
from kipoi_enformer.logger import logger
import pyarrow as pa
import pyarrow.parquet as pq
from kipoiseq.transforms.functional import one_hot_dna
from tqdm.autonotebook import tqdm
from collections import defaultdict
import math
import yaml
import pickle
import polars as pl

__all__ = ['Enformer', 'EnformerVeff']

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

    def predict(self, dataloader: VCFEnformerDL, batch_size: int, filepath: str | pathlib.Path):
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
        schema = pa.schema(
            [
                (f'tracks_{allele}_{shift}', pa.list_(pa.list_(pa.float32())))
                for shift in
                ([-dataloader.shift, 0, dataloader.shift] if dataloader.shift else [0])
                for allele in ['ref', 'alt']
            ] + [
                ('enformer_start', pa.int64()),
                ('enformer_end', pa.int64()),
                ('landmark_pos', pa.int64()),
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
            ])

        batch_iterator = self._batch_iterator(dataloader, batch_size)
        batch_counter = 0
        total_batches = math.ceil(len(dataloader) / batch_size)
        with pq.ParquetWriter(filepath, schema) as writer:
            for batch in tqdm(batch_iterator, total=total_batches):
                batch_counter += 1
                writer.write_batch(batch)

        # sanity check for the dataloader
        assert batch_counter == total_batches

    def _batch_iterator(self, dataloader: VCFEnformerDL, batch_size: int):
        batch = []
        counter = 0
        for data in dataloader:
            batch.append(data)
            if len(batch) == batch_size:
                counter += 1
                logger.debug(f'Processing batch {counter}')
                # process batch and save results in a parquet file
                yield self._to_pyarrow(self._process_batch(batch))
                batch = []

        # process remaining sequences if any
        if len(batch) > 0:
            counter += 1
            # process batch and save results in a parquet file
            yield self._to_pyarrow(self._process_batch(batch))

    def _process_batch(self, batch):
        """
        Process a batch of data. Run the model and prepare the results dict.
        :param batch: list of data dicts
        :return: Results dict. Structure: {'metadata': {field: [values]}, 'predictions': {sequence_key: [values]}}
        """
        batch_size = len(batch)

        # presumably all records will have the same number of sequences
        seqs_per_record = len(batch[0]['sequences'])

        sequence_keys = [k for data in batch for k in data['sequences'].keys()]
        sequences = [one_hot_dna(seq).astype(np.float32) for data in batch for seq in data['sequences'].values()]

        # create input tensor
        input_tensor = tf.convert_to_tensor(sequences)
        assert input_tensor.shape == (batch_size * seqs_per_record, 393_216, 4)

        # run model
        predictions = self._model.predict_on_batch(input_tensor)['human'].numpy()
        assert predictions.shape == (batch_size * seqs_per_record, 896, 5313)
        predictions = predictions.tolist()

        # prepare results dict
        # metadata fields are the same for all records
        # prediction fields are the same for all records
        metadata_field_names = list(batch[0]['metadata'].keys())
        results = {
            'metadata': defaultdict(list),
            'predictions': defaultdict(list)
        }
        for i in range(batch_size):
            metadata = batch[i]['metadata']
            for f in metadata_field_names:
                results['metadata'][f].append(metadata[f])

            for j in range(seqs_per_record):
                pred_idx = i * seqs_per_record + j
                seq_key = sequence_keys[pred_idx]
                results['predictions'][f'tracks_{seq_key}'].append(predictions[pred_idx])

        return results

    @staticmethod
    def _to_pyarrow(results: dict):
        """
        Convert the results dict from the _process_batch method to a pyarrow table and write it in a parquet file.
        :param results: pyarrow.RecordBatch object
        """
        logger.debug('Converting results to pyarrow')

        # format predictions
        predictions = {k: pa.array(v, type=pa.list_(pa.list_(pa.float32())))
                       for k, v in results['predictions'].items()}
        metadata = {k: pa.array(v) for k, v in results['metadata'].items()}

        logger.debug('Constructing pyarrow record batch')
        # construct RecordBatch
        return pa.RecordBatch.from_arrays((list(predictions.values()) + list(metadata.values())),
                                          names=(list(predictions.keys()) + list(metadata.keys())))


class EnformerVeff:
    def __init__(self, tissue_matcher_path: str | pathlib.Path,
                 enformer_tracks_path: str | pathlib.Path):
        """
        :param tissue_matcher_path: A pickle containing dictionary of linear models for each GTEx tissue.
        :param enformer_tracks_path: A yaml file mapping the name of the tracks to the index in the predictions.
        """
        with open(tissue_matcher_path, 'rb') as f:
            self.tissue_matcher_lm_dict = {k: v['ingenome'] for k, v in pickle.load(f).items()}

        with open(enformer_tracks_path, 'rb') as f:
            self.enformer_tracks_dict = yaml.safe_load(f)

    def estimate_veff(self, prediction_path: str | pathlib.Path, output_path: str | pathlib.Path, shift: int = 43,
                      num_bins: int = 3, batch_size: int = 1):
        """
        Load the predictions from the parquet file lazily.
        Then, calculate the variant effect size (veff) for each variant-transcript pair.
        Finally, save the veff values in a new parquet file.

        :param prediction_path: The parquet file that contains the enformer predictions.
        :param batch_size: The number of records to read and write at once.
        :param output_path: The parquet file that will contain the veff values.
        :param num_bins: number of bins to average over for each variant-transcript pair
        :param shift: The shift applied to the central sequence.
        The average predictions will be calculated at the landmark bin of each variant-transcript pair.
        """

        schema = pa.schema(
            [('enformer_start', pa.int64()),
             ('enformer_end', pa.int64()),
             ('landmark_pos', pa.int64()),
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
             ('tissue', pa.string()),
             ('ref_score', pa.float64()),
             ('alt_score', pa.float64()),
             ('delta_score', pa.float64()),
             ])
        with pq.ParquetWriter(output_path, schema) as writer:
            num_rows = pq.read_metadata(prediction_path).num_rows
            total_batches = math.ceil(num_rows / batch_size)
            batch_counter = 0
            logger.debug(f'Iterating over the batches in the parquet file {prediction_path}')
            logger.debug(f'Writing the veff values to the parquet file {output_path}')
            for batch_ref, batch_alt, batch_meta in tqdm(
                    self._batch_iterator(prediction_path, shift, num_bins, batch_size=batch_size), total=total_batches):
                batch_counter += 1
                # todo: implement this
                record_batch = self._map_to_tissue(batch_ref, batch_alt, batch_meta)
                writer.write(record_batch)

            # sanity check
            assert batch_counter == total_batches

    def _batch_iterator(self, prediction_path: str | pathlib.Path, shift: int = 43, num_bins: int = 3,
                        batch_size: int = 1):
        """
        Load the predictions from the parquet file lazily.
        :param prediction_path: The parquet file that contains the enformer predictions.
        :param shift: The shift applied to the central sequence.
        :param num_bins: The number of bins to average over for each variant-transcript pair.
        :param batch_size: The number of records to load at once.
        :return:
        """
        bin_size = Enformer.BIN_SIZE
        pred_seq_length = Enformer.PRED_SEQUENCE_LENGTH
        shifts = [0] if shift == 0 else [-shift, 0, shift]
        universal_samples = ['Clontech Human Universal Reference Total RNA',
                             'SABiosciences XpressRef Human Universal Total RNA',
                             'CAGE:Universal RNA - Human Normal Tissues Biochain']
        tracks = [v for k, v in self.enformer_tracks_dict.items()
                  if "CAGE" in k and not any(s in k for s in universal_samples)]

        df = pl.read_parquet(prediction_path)
        for frame in df.iter_slices(n_rows=batch_size):
            yield self._aggregate_batch(frame, shifts, pred_seq_length,
                                        bin_size, num_bins, tracks)

    @staticmethod
    def _aggregate_batch(frame, shifts, pred_seq_length, bin_size, num_bins, tracks):
        """
        Aggregate the predictions for a variant-transcript pair.
        :param frame:
        :param shifts:
        :param pred_seq_length:
        :param bin_size:
        :param num_bins:
        :param tracks:
        :return:
        """

        ref_preds = []
        alt_preds = []
        for shift in shifts:
            # estimate the landmark bin
            # todo verify this calculation
            landmark_bin = (pred_seq_length // 2 + 1 - shift) // bin_size
            # get num_bins - 1 neighboring bins centered at landmark bin
            bins = [landmark_bin + i for i in range(-math.floor(num_bins / 2), math.ceil(num_bins / 2))]
            assert len(bins) == num_bins
            ref = np.stack(frame[f'tracks_ref_{shift}'].to_list())[:, bins, :][:, :, tracks]
            alt = np.stack(frame[f'tracks_alt_{shift}'].to_list())[:, bins, :][:, :, tracks]
            ref_preds.append(ref)
            alt_preds.append(alt)

        ref = np.stack(ref_preds).swapaxes(0, 1)
        alt = np.stack(alt_preds).swapaxes(0, 1)

        assert ref.shape == alt.shape == (len(frame), len(shifts), num_bins, len(tracks))
        # average over shifts and bins
        ref = ref.mean(axis=(1, 2))
        alt = alt.mean(axis=(1, 2))

        assert ref.shape == alt.shape == (len(frame), len(tracks),)
        metadata = frame.select([k for k in frame.columns if not ('tracks_' in k)]).to_dict()
        return ref, alt, metadata

    def _map_to_tissue(self, batch_ref, batch_alt, batch_meta):
        """
        Map the enformer predictions to the GTEx tissues using the linear models.
        :param batch_ref: The reference predictions.
        :param batch_alt: The alternative predictions.
        :param batch_meta: The metadata for each variant-transcript pair.
        :return:
        """
        # calculate veff
        ref = np.log10(batch_ref + 1)
        alt = np.log10(batch_alt + 1)
        ref_tissue_veff = defaultdict()
        alt_tissue_veff = defaultdict()
        for tissue, lm in self.tissue_matcher_lm_dict.items():
            ref_tissue_veff[tissue] = lm.predict(ref)
            alt_tissue_veff[tissue] = lm.predict(alt)

        ref_scores = []
        alt_scores = []
        delta_scores = []
        tissues = []
        meta = defaultdict(list)
        for tissue in self.tissue_matcher_lm_dict.keys():
            tissues.extend([tissue] * len(batch_ref))
            ref_scores.extend(ref_tissue_veff[tissue].tolist())
            alt_scores.extend(alt_tissue_veff[tissue].tolist())
            delta_scores.extend((ref_tissue_veff[tissue] - alt_tissue_veff[tissue]).tolist())
            for k, v in batch_meta.items():
                meta[k].extend(batch_meta[k])

        values = [pa.array(meta[k]) for k in meta] + [pa.array(tissues),
                                                      pa.array(ref_scores),
                                                      pa.array(alt_scores),
                                                      pa.array(delta_scores)]
        names = [k for k in meta] + ['tissue', 'ref_score', 'alt_score', 'delta_score']

        # write to parquet
        return pa.RecordBatch.from_arrays(values, names=names)
