#!/usr/bin/env python
"""
Convert TIMIT audio files into spectral coefficients.
"""

import numpy as np
import librosa
import sklearn.preprocessing
import h5py
import logging
from tqdm import tqdm # progress bar

import os
import fnmatch

from phoneme import timit_phonemes, phoneme2index_list, phoneme2index_dict


# global config: load from previous saved dataset if True, else recompute
load_features = False

# TODO: configurable path to /data/speech/timit/
timit_dir = '/data/speech/timit/TIMIT/'
timit_hdf5_filepath = '/data/speech/timit/timit.hdf5'

train_name, test_name = 'train', 'test'
train_dir = os.path.join(timit_dir, train_name.upper())
test_dir = os.path.join(timit_dir, test_name.upper())


# simple logging
logger = logging.getLogger('TIMIT')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
ch.setFormatter(formatter)
logger.addHandler(ch)


def recursive_glob_ext(dirpath, ext):
  """Recursively find files with an extension in a TIMIT directory."""
  return [os.path.splitext(os.path.join(dirpath, filename))[0] # remove extension
    for dirpath, _, files in os.walk(dirpath)
    for filename in fnmatch.filter(files, '*.{}'.format(ext))]


def mfcc_features(filename):
  """Preprocessing per CTC paper.

  (These are not the simpler linear spectrogram features alone as in Deep
  Speech).

  Properties:
  - 10ms frames with 5ms overlap
  - 12 MFCCs with 26 filter banks
  - replace first MFCC with energy (TODO: log-energy)
  - add first-order derivatives for all of the above
  - total: 26 coefficients
  """
  d, sr = librosa.load(filename)

  frame_length_seconds = 0.010
  frame_overlap_seconds = 0.005

  mfccs = librosa.feature.mfcc(d, sr, n_mfcc=1+12, n_fft=int(frame_overlap_seconds*sr), hop_length=int(frame_overlap_seconds*sr))

  # energy (TODO: log?)
  energy = librosa.feature.rmse(d, n_fft=int(frame_overlap_seconds*sr), hop_length=int(frame_overlap_seconds*sr))

  mfccs[0] = energy # replace first MFCC with energy, per convention

  deltas = librosa.feature.delta(mfccs, order=1)
  mfccs_plus_deltas = np.vstack([mfccs, deltas])

  coeffs = sklearn.preprocessing.scale(mfccs_plus_deltas, axis=1)

  return coeffs


def dirpath2dataset(dirpath):
  """Convert a TIMIT dirpath to a dataset.

  The filename alone is not unique.

  e.g., TIMIT/TRAIN/DR8/MMPM0/SX251.WAV => MMPM0/SX251.WAV
  """
  if not '/' in dirpath:
    raise Exception, "not a valid TIMIT dirpath"

  dataset_name = '/'.join(dirpath.split('/')[-2:])
  return dataset_name


def phoneme_transcription(phoneme_filename):
  phoneme_column = -1
  # we can discard the first two columns, which provide the time alignment
  transcription = [line.split()[phoneme_column].strip() for line in open(phoneme_filename)]
  return transcription


def verify_phonemes(timit_phoneme_set, transcription_phoneme_set):
  """Make sure every pre-specified phoneme was seen in data, and the converse."""
  for phoneme in transcription_phoneme_set:
    if phoneme not in timit_phoneme_set:
      logger.error(phoneme + ' not in TIMIT phonemes')

  for phoneme in timit_phoneme_set:
    if phoneme not in transcription_phoneme_set:
      logger.error(phoneme + ' not in transcribed phonemes')


def compute_spectrograms(audio_filenames):
  """Extract spectrogram features from each audio file."""
  features_list = []
  audio_ext = ".WAV"

  for audio_basename in tqdm(audio_filenames):
    # recompute spectrogram features
    # FIXME: on interrupt, kill the thread which librosa launches via audioread
    feature_vector = mfcc_features(audio_basename + audio_ext)
    features_list.append(feature_vector)

  return features_list


def load_precomputed_spectrograms(filepath):
  """Load precomputed spectrogram features to save time."""
  features_list = []
  # TODO: this HDF5 group structure is outdated, recompute and save a new one
  with h5py.File(filepath, 'r') as hf:
    for g in hf['utterances']:
      for dataset in hf['utterances'][g]:
        data = np.array(hf['utterances'][g][dataset])
        features_list.append(data)

  return features_list


def load_timit(filepath, train=True):
  # TODO: load test also
  with h5py.File(filepath, 'r') as hf:
    train_spectrograms = np.array(hf['timit']['train']['spectrograms'])
    train_labels = np.array(hf['timit']['train']['labels'])
    train_seq_lens = np.array(hf['timit']['train']['seq_lens'])

    return train_spectrograms, train_labels, train_seq_lens


def save_feature_dataset(audio_filenames, spectrograms, seq_lens, phoneme2index_list, labels, filepath, overwrite=False):
  """Save computed features for TIMIT.

  Args:
  - maps from subset kinds 'train' and 'test' to corresponding data:
    - audio_filenames: list of basepaths to TIMIT examples
    - spectrograms: np.array((n_examples, max_frames, n_coeffs))
      - n_examples: number of TIMIT examples (e.g., train=4206)
      - max_frames: the most frames in any example
      - n_coeffs: number of spectrogram features (e.g., 26 with 12 MFCCs, one
        energy, and their 13 deltas)
    - seq_lens: number of labels in each target sequence (<= max_labels)
    - labels: np.array((n_examples, max_labels))
      - max_labels: the most labels in any example (e.g., train=75)
  - phoneme2index_list: a map from phoneme strings (e.g., 'sh') to indices,
    ordered as in TIMIT PHONCODE.DOC
  """
  if overwrite:
    file_mode = 'w'
  else:
    file_mode = 'w-' # fail if file exists

  with h5py.File(filepath, file_mode) as hf:
    timit = hf.create_group('timit')

    train_name = 'train'
    test_name = 'test'

    train = timit.create_group(train_name)
    test = timit.create_group(test_name)

    for subset_kind, subset_dataset in [(train_name, train), (test_name, test)]:
      # (n_examples,)
      subset_dataset.create_dataset('example_paths', dtype="S100", data=np.array(audio_filenames[subset_kind]))

      # (n_examples, max_frames, n_coeffs)
      subset_dataset.create_dataset('spectrograms', data=spectrograms[subset_kind])

      # (n_examples,)
      subset_dataset.create_dataset('seq_lens', data=seq_lens[subset_kind])

      # (n_examples, max_labels)
      label_dataset = subset_dataset.create_dataset('labels', data=labels[subset_kind])

      # store phoneme <-> index mapping in HDF5 attributes to avoid numpy structured arrays
      # indices are per order in TIMIT phoncode.doc
      for phoneme, index in phoneme2index_list:
        label_dataset.attrs[phoneme] = index

        # NOTE: because we don't use '1' and '2' as TIMIT phonemes, there
        # shouldn't be any collisions with the indices '1' and '2' when we put
        # both into the same dict as strings
        label_dataset.attrs[str(index)] = phoneme


def index_labels(phoneme2index_dict, timit_transcriptions, max_labels):
  """Convert TIMIT transcriptions to integer np.array of indices."""
  labels = np.empty((n_examples, max_labels))
  seq_lens = np.empty((n_examples,))
  for i, transcription in enumerate(timit_transcriptions):
    index_transcription = [phoneme2index_dict[phoneme] for phoneme in transcription]
    labels[i,:len(transcription)] = index_transcription
    seq_lens[i] = len(index_transcription)

  return labels, seq_lens


def build_spectrogram_array(features_list, n_examples, max_frames, n_coeffs):
  """Convert list of ragged spectrograms to np.array with list of lens."""
  spectrograms = np.empty((n_examples, max_frames, n_coeffs))

  for i, feature_vector in enumerate(features_list):
    example_frames = feature_vector.shape[1]
    spectrograms[i,:example_frames,:] = feature_vector.T

  return spectrograms


def load_transcriptions(audio_filenames):
  """Load list of phoneme transcriptions.

  Each phoneme transcription is a list of phonemes without time alignments.
  """
  phoneme_ext = ".PHN"
  transcriptions = []
  for audio_basename in tqdm(audio_filenames):
    # obtain list of phonemes, discarding time-alignment
    tr = phoneme_transcription(audio_basename + phoneme_ext)
    transcriptions.append(tr)

  return transcriptions


def phoneme_set(transcriptions):
  """Reduce list of lists of phonemes to a set of phonemes."""
  transcription_phonemes = set()
  for transcription in transcriptions:
    for phoneme in transcription:
      transcription_phonemes.add(phoneme)

  return transcription_phonemes


if __name__ == "__main__":
  logger.info("Starting to preprocess TIMIT audio data.")
  logger.info("Walking TIMIT data directory...")

  audio_filenames = {}
  spectrograms = {}
  seq_lens = {}
  labels = {}

  for subset_kind, subset_dir in [(train_name, train_dir), (test_name, test_dir)]:
    subset_audio_filenames = recursive_glob_ext(subset_dir, ext="WAV")

    logger.info("Loading phoneme transcriptions for {}...".format(subset_kind))
    subset_transcriptions = load_transcriptions(subset_audio_filenames)

    # sanity check
    verify_phonemes(set(timit_phonemes), phoneme_set(subset_transcriptions))

    subset_features_list = []
    if load_features:
      logger.info("Loading precomputed spectrograms for {}...".format(subset_kind))
      features_list = load_precomputed_spectrograms(filepath='/data/speech/timit/mfcc-timit.hdf5')
    else:
      logger.info("Computing spectrograms for {}...".format(subset_kind))
      subset_features_list = compute_spectrograms(subset_audio_filenames)

    # compute sizes for np.arrays
    n_examples = len(subset_features_list)
    max_frames = max(feature_vector.shape[1] for feature_vector in subset_features_list)
    n_coeffs = subset_features_list[0].shape[0] # same for all
    max_labels = max(len(transcription) for transcription in subset_transcriptions)

    logger.info("Building label array by indexing labels from transcriptions for {}...".format(subset_kind))
    subset_labels, subset_seq_lens = index_labels(phoneme2index_dict, subset_transcriptions, max_labels)

    logger.info("Building spectrogram array for {}...".format(subset_kind))
    subset_spectrograms = build_spectrogram_array(subset_features_list, n_examples, max_frames, n_coeffs)

    # store for later saving
    audio_filenames[subset_kind] = subset_audio_filenames
    spectrograms[subset_kind] = subset_spectrograms
    labels[subset_kind] = subset_labels
    seq_lens[subset_kind] = subset_seq_lens

    logger.info("Finished preprocessing {}.".format(subset_kind))

  logger.info("Saving HDF5 train/test dataset...")
  save_feature_dataset(audio_filenames, spectrograms, seq_lens, phoneme2index_list, labels, filepath=timit_hdf5_filepath)

  logger.info("Finished.")

