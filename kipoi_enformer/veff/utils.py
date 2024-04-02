import pathlib

import tensorflow_hub as hub
import tensorflow as tf
from .dataloader import Enformer_DL
from kipoi_enformer.logger import logger
import pyarrow as pa
import pyarrow.parquet as pq

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

    def predict(self, dataloader: Enformer_DL, batch_size: int):
        """
        Predict on a dataloader and return the results
        :param dataloader:
        :param batch_size:
        :return: dict of results in the following format
        {
            <Transcript-variant ID>: {
                'predictions': {
                    '<allele>:<shift>': <prediction>
                },
                'metadata': {...}
            }
        }
        """
        logger.debug('Predicting on dataloader')
        assert batch_size > 0

        # the structure of the results dict:
        # {
        #   <Transcript-variant ID>: {
        #       'predictions': {
        #           '<allele>:<shift>': <prediction>
        #       },
        #       'metadata': {...}
        #   }
        # }
        results = dict()

        def save_result(batch_result):
            """
            Save the results of a batch to the results dict
            :param batch_result:
            :return:
            """
            for res in batch_result:
                pred = res[0]
                metadata = res[1]
                id_ = (f'{metadata["transcript_id"]}:{metadata["variant_start"] + 1}:'
                       f'{metadata["ref"]}>{metadata["alt"]}:')
                res_obj: dict | None = results.get(id_, None)
                if res_obj is None:
                    results[id_] = {
                        'predictions': {
                            f'{metadata["allele"]}_{metadata["shift"]}': pred
                        },
                        'metadata': {k: v for k, v in metadata.items() if k not in ['allele', 'shift']}
                    }
                else:
                    res_obj['predictions'][f'{metadata["allele"]}_{metadata["shift"]}'] = pred

        batch = []
        counter = 0
        for data in dataloader:
            batch.append(data)
            if len(batch) == batch_size:
                counter += 1
                logger.debug(f'Processing batch {counter}')
                # process batch and save results in the results dict
                save_result(self._process_batch(batch))
                batch = []

        # process remaining sequences if any
        if len(batch) > 0:
            counter += 1
            logger.debug(f'Processing batch {counter}')
            # process batch and save results in the results dict
            save_result(self._process_batch(batch))

        return results

    def _process_batch(self, batch):
        batch_size = len(batch)
        sequences = [data['sequence'] for data in batch]
        metadata = [data['metadata'] for data in batch]

        # create input tensor
        input_tensor = tf.convert_to_tensor(sequences)
        assert input_tensor.shape == (batch_size, 393_216, 4)

        # run model
        predictions = self._model.predict_on_batch(input_tensor)['human'].numpy()
        assert predictions.shape == (batch_size, 896, 5313)

        # return predictions and metadata
        return list(zip(predictions, metadata))

    @staticmethod
    def to_parquet(results: dict, path: str | pathlib.Path):
        """
        Convert the results dict from the predict method to a pyarrow table and write it in a parquet file.
        :param path: path to write the parquet file
        :param results: dict from the predict method
        """
        logger.debug('Converting results to pyarrow table')

        predictions = [x['predictions'] for x in results.values()]
        metadata = [x['metadata'] for x in results.values()]

        pred_names = predictions[0].keys()
        metadata_names = metadata[0].keys()

        # format predictions
        predictions = {pred_name: pa.array([pred[pred_name].tolist() for pred in predictions],
                                           type=pa.list_(pa.list_(pa.float64())))
                       for pred_name in pred_names}

        # format metadata
        metadata = {metadata_name: pa.array([meta[metadata_name] for meta in metadata])
                    for metadata_name in metadata_names}

        logger.debug('Constructing pyarrow table')
        # construct table
        pa_table = pa.Table.from_arrays(arrays=(list(predictions.values()) + list(metadata.values())),
                                        names=(list(pred_names) + list(metadata_names)))

        logger.debug(f'Writing pyarrow table to {path}')
        pq.write_table(pa_table, path)
