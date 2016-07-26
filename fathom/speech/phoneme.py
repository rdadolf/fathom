#!/usr/bin/env python

"""
Python representation of 61 TIMIT phonemes from DOC/PHONCODE.DOC.
"""

# list of 61 TIMIT phonemes from phoncode.doc
timit_phonemes = [
  # stop (and corresponding closures)
  'b', 'bcl',
  'd', 'dcl',
  'g', 'gcl',
  'p', 'pcl',
  't', 'tcl', # NOTE: typo of "tck" in TIMIT docs
  'k', 'kcl',
  'dx',
  'q',

  # affricates
  'jh',
  'ch',

  # fricatives
  's',
  'sh',
  'z',
  'zh',
  'f',
  'th',
  'v',
  'dh',

  # nasals
  'm',
  'n',
  'ng',
  'em',
  'en',
  'eng',
  'nx',

  # semivowels and glides
  'l',
  'r',
  'w',
  'y',
  'hh',
  'hv',
  'el',

  # vowels
  'iy',
  'ih',
  'eh',
  'ey',
  'ae',
  'aa',
  'aw',
  'ay',
  'ah',
  'ao',
  'oy',
  'ow',
  'uh',
  'uw',
  'ux',
  'er',
  'ax',
  'ix',
  'axr',
  'ax-h',

  # others
  'pau',
  'epi',
  'h#',

  # lexicon-only (thus omitted from transcriptions)
  #'1',
  #'2',
]

# map phoneme to index
phoneme2index_list = [(phoneme, index) for index, phoneme in enumerate(timit_phonemes)]
phoneme2index_dict = dict(phoneme2index_list)

index2phoneme_list = [(index, phoneme) for index, phoneme in enumerate(timit_phonemes)]
index2phoneme_dict = dict(index2phoneme_list)

