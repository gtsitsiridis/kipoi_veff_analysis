import pathlib

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

__all__ = ['Enformer']

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
                (f'{allele}_{shift}', pa.list_(pa.list_(pa.float32())))
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
        with pq.ParquetWriter(filepath, schema) as writer:
            for batch in tqdm(batch_iterator, total=len(dataloader) // batch_size + 1):
                writer.write_batch(batch)

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
                results['predictions'][seq_key].append(predictions[pred_idx])

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


def estimate_veff(prediction_path: str | pathlib.Path, tissue_matcher_lm_dict: dict, enformer_tracks_dict: dict,
                  output_path: str | pathlib.Path, shift: int = 43, num_bins: int = 3):
    """
    Load the predictions from the parquet file lazily.
    Then, calculate the variant effect size (veff) for each variant-transcript pair.
    Finally, save the veff values in a new parquet file.

    :param enformer_tracks_dict: A dictionary of mapping the name of the tracks to the index in the predictions.
    :param prediction_path: The parquet file that contains the enformer predictions.
    :param tissue_matcher_lm_dict: A dictionary of linear models for each GTEx tissue.
    :param output_path: The parquet file that will contain the veff values.
    :param num_bins: number of bins to average over for each variant-transcript pair
    :param shift: The shift applied to the central sequence.
    The average predictions will be calculated at the landmark bin of each variant-transcript pair.
    """
    bin_size = Enformer.BIN_SIZE
    pred_seq_length = Enformer.PRED_SEQUENCE_LENGTH
    shifts = [0] if shift == 0 else [-shift, 0, shift]
    universal_samples = ['Clontech Human Universal Reference Total RNA',
                         'SABiosciences XpressRef Human Universal Total RNA',
                         'CAGE:Universal RNA - Human Normal Tissues Biochain']
    tracks = [v for k, v in enformer_tracks_dict.items()
              if "CAGE" in k and not any(s in k for s in universal_samples)]
    assert len(tracks) == 635

    # Load the predictions lazily
    average_refs = []
    average_alts = []
    veff_tbl = None
    with pq.ParquetFile(prediction_path) as pf:
        metadata_cols = [x for x in pf.schema.names if x != 'element']
        veff_tbl = pf.read(columns=metadata_cols)
        for batch in pf.iter_batches(batch_size=1):
            ref_preds = []
            alt_preds = []
            for shift in shifts:
                # estimate the landmark bin
                # todo verify this calculation
                landmark_bin = (pred_seq_length // 2 + 1 - shift) // bin_size
                # get num_bins - 1 neighboring bins centered at landmark bin
                bins = [landmark_bin + i for i in range(-math.floor(num_bins / 2), math.ceil(num_bins / 2))]
                assert len(bins) == num_bins
                ref = batch[f'ref_{shift}'][0].as_py()
                alt = batch[f'alt_{shift}'][0].as_py()
                ref_preds.append([ref[bin_] for bin_ in bins])
                alt_preds.append([alt[bin_] for bin_ in bins])
            ref = np.array(ref_preds)
            alt = np.array(alt_preds)

            assert ref.shape == (len(shifts), num_bins, 5313)
            assert alt.shape == (len(shifts), num_bins, 5313)

            ref = ref[:, :, tracks]
            alt = alt[:, :, tracks]

            # average over shifts and bins
            ref = ref.mean(axis=(0, 1))
            alt = alt.mean(axis=(0, 1))

            assert ref.shape == alt.shape
            assert ref.shape == (635,)
            assert alt.shape == (635,)

            average_refs.append(ref)
            average_alts.append(alt)

    # calculate veff
    ref = np.stack(average_refs)
    alt = np.stack(average_alts)
    ref = np.log10(ref + 1)
    alt = np.log10(alt + 1)
    ref_tissue_veff = defaultdict()
    alt_tissue_veff = defaultdict()
    for tissue, lm in tissue_matcher_lm_dict.items():
        ref_tissue_veff[tissue] = lm.predict(ref).tolist()
        alt_tissue_veff[tissue] = lm.predict(alt).tolist()

    for tissue in tissue_matcher_lm_dict.keys():
        veff_tbl = veff_tbl.append_column(f'ref_{tissue}', [ref_tissue_veff[tissue]])
        veff_tbl = veff_tbl.append_column(f'alt_{tissue}', [alt_tissue_veff[tissue]])

    # save the veff values in a new parquet file
    pq.write_table(veff_tbl, output_path)
