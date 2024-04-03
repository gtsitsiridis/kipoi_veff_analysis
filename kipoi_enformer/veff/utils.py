import pathlib

import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from .dataloader import Enformer_DL
from kipoi_enformer.logger import logger
import pyarrow as pa
import pyarrow.parquet as pq
from kipoiseq.transforms.functional import one_hot_dna
import pyarrow.dataset as ds

__all__ = ['Enformer']

MODEL_PATH = 'https://tfhub.dev/deepmind/enformer/1'


class Enformer:

    def __init__(self, model: str | tf.keras.Model = MODEL_PATH):
        """
        :param model: path to model or tf.keras.Model
        """
        if isinstance(model, str):
            self._model = hub.load(model).model
        else:
            self._model = model

    def predict(self, dataloader: Enformer_DL, batch_size: int, output_dir: str | pathlib.Path):
        """
        Predict on a dataloader and save the results in a parquet file
        :param output_dir:
        :param dataloader:
        :param batch_size:
        :return: filepath to the parquet dataset
        """

        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(exist_ok=False, parents=False)

        logger.debug('Predicting on dataloader')
        assert batch_size > 0
        # todo fetch schema from data
        schema = pa.schema([
            ('ref_-43', pa.list_(pa.list_(pa.float32()))),
            ('ref_0', pa.list_(pa.list_(pa.float32()))),
            ('ref_43', pa.list_(pa.list_(pa.float32()))),
            ('alt_-43', pa.list_(pa.list_(pa.float32()))),
            ('alt_0', pa.list_(pa.list_(pa.float32()))),
            ('alt_43', pa.list_(pa.list_(pa.float32()))),
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

        batch_iterator = self.batch_iterator(dataloader, batch_size)
        ds.write_dataset(batch_iterator, output_dir, format='parquet', schema=schema)

    def batch_iterator(self, dataloader: Enformer_DL, batch_size: int):
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
            logger.debug(f'Processing batch {counter}')
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
        pred_field_names = (batch[0]['sequences'].keys())
        results = {
            'metadata': {f: [] for f in metadata_field_names},
            'predictions': {f: [] for f in pred_field_names}
        }
        for i in range(batch_size):
            metadata = batch[i]['metadata']
            for f in metadata_field_names:
                results['metadata'][f].append(metadata[f])

            for j in range(seqs_per_record):
                pred_idx = i * seqs_per_record + j
                seq_key = sequence_keys[pred_idx]
                results['predictions'][seq_key].append(predictions[pred_idx])

        return results

    @staticmethod
    def _to_pyarrow(results: dict):
        """
        Convert the results dict from the _process_batch method to a pyarrow table and write it in a parquet file.
        :param path: path to write the parquet file
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
        #
        # logger.debug(f'Writing pyarrow table to {path}')
        # pq.write_table(pa_table, path)
