from __future__ import annotations

import pathlib
import tensorflow as tf
from kipoi_aparent2.dataloader import CSEDataloader
from kipoi_aparent2.logger import logger
from keras.models import load_model
import pyarrow as pa
import math
import pyarrow.parquet as pq
from tqdm.autonotebook import tqdm
import numpy as np

__all__ = ['Aparent2']


class Aparent2:

    def __init__(self, model_path: str | pathlib.Path):
        logger.debug(f'Loading model from {model_path}')
        self._model = load_model(str(model_path))

    def predict(self, dataloader: CSEDataloader, batch_size: int, filepath: str | pathlib.Path):
        """
        Predict on a dataloader and save the results in a parquet file
        :param filepath:
        :param dataloader:
        :param batch_size:
        :return: filepath to the parquet dataset
        """
        logger.debug('Predicting on dataloader')
        assert batch_size > 0

        schema = dataloader.pyarrow_metadata_schema
        schema = schema.insert(0, pa.field(f'tracks', pa.list_(pa.list_(pa.float32()))))

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
                batch = self._to_pyarrow(self._process_batch(batch))
                writer.write_batch(batch)

    def _process_batch(self, batch):
        """
        Process a batch of data. Run the model and prepare the results dict.
        :param batch: list of data dicts
        :return: Results dict. Structure: {'metadata': {field: [values]}, 'predictions': {sequence_key: [values]}}
        """
        batch_size = batch['sequences'].shape[0]
        seqs_per_record = batch['sequences'].shape[1]
        sequences = np.reshape(batch['sequences'], (batch_size * seqs_per_record, 205, 4))[:, None, :, :]

        # Always set this one-hot variable to 11 (training sub-library bias)
        lib = np.zeros((len(sequences), 13))
        lib[:, 11] = 1.

        assert sequences.shape == (batch_size * seqs_per_record, 1, 205, 4)
        assert lib.shape == (batch_size * seqs_per_record, 13)

        # run model
        predictions = self._model.predict_on_batch([sequences, lib])[1]
        assert predictions.shape == (batch_size * seqs_per_record, 206)
        predictions = predictions.reshape(batch_size, seqs_per_record, 206)

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
            'tracks': pa.array(results['tracks'].tolist(), type=pa.list_(pa.list_(pa.float32()))),
            **metadata
        }

        logger.debug('Constructing pyarrow record batch')
        # construct RecordBatch
        return pa.RecordBatch.from_arrays(list(formatted_results.values()), names=list(formatted_results.keys()))
