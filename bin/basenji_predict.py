#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import print_function
from optparse import OptionParser
import os

import numpy as np
import pandas as pd
import pyBigWig
import tensorflow as tf

from basenji import dataset
from basenji import params
from basenji import seqnn

"""
basenji_predict.py

Predict sequences from TFRecords.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <data_dir>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='track_bed',
      help='BED file describing regions so we can output BigWig tracks')
  parser.add_option('-d', dest='sample_down',
      default=1.0, type='float',
      help='Sample sequence positions down. [Default: %default]')
  parser.add_option('-g', dest='genome_file',
      default='%s/tutorials/data/human.hg19.genome' % os.environ['BASENJIDIR'],
      help='Chromosome length information [Default: %default]')
  parser.add_option('--mc', dest='mc_n',
      default=0, type='int',
      help='Monte carlo test iterations [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='test_out',
      help='Output directory for test statistics [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average the fwd and rc predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--ti', dest='track_indexes',
      help='Comma-separated list of target indexes to output BigWig tracks')
  parser.add_option('--tfr', dest='tfr_pattern',
      default='test-*.tfr',
      help='TFR pattern string appended to data_dir [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters, model, and test data HDF5')
  else:
    params_file = args[0]
    model_file = args[1]
    data_dir = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  # parse shifts to integers
  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  # read targets
  if options.targets_file is None:
    options.targets_file = '%s/targets.txt' % data_dir
    targets_df = pd.read_table(options.targets_file, index_col=0)
    target_subset = None
  else:
    targets_df = pd.read_table(options.targets_file, index_col=0)
    target_subset = targets_df.index

  # read model parameters
  job = params.read_job_params(params_file)

  # construct data ops
  tfr_pattern_path = '%s/tfrecords/%s' % (data_dir, options.tfr_pattern)
  data_ops, test_init_op = make_data_ops(job, tfr_pattern_path)

  # initialize model
  model = seqnn.SeqNN()
  model.build_from_data_ops(job, data_ops, ensemble_rc=options.rc,
                            ensemble_shifts=options.shifts,
                            target_subset=target_subset)

  # initialize saver
  saver = tf.train.Saver()

  with tf.Session() as sess:
    # start queue runners
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    # load variables into session
    saver.restore(sess, model_file)

    # test
    sess.run(test_init_op)
    test_preds = model.predict_tfr(sess, sample=options.sample_down)

    np.save('%s/preds.npy' % options.out_dir, test_preds)

    # print normalization factors
    target_means = test_preds.mean(axis=(0,1), dtype='float64')
    target_means_median = np.median(target_means)
    target_means /= target_means_median
    norm_out = open('%s/normalization.txt' % options.out_dir, 'w')
    print('\n'.join([str(tu) for tu in target_means]), file=norm_out)
    norm_out.close()


  #######################################################
  # BigWig tracks

  # NOTE: THESE ASSUME THERE WAS NO DOWN-SAMPLING ABOVE

  # print bigwig tracks for visualization
  if options.track_bed:
    if options.genome_file is None:
      parser.error('Must provide genome file in order to print valid BigWigs')

    if not os.path.isdir('%s/tracks' % options.out_dir):
      os.mkdir('%s/tracks' % options.out_dir)

    track_indexes = range(test_preds.shape[2])
    if options.track_indexes:
      track_indexes = [int(ti) for ti in options.track_indexes.split(',')]

    bed_set = 'test'
    if options.valid:
      bed_set = 'valid'

    for ti in track_indexes:
      test_targets_ti = test_targets[:, :, ti]

      # make true targets bigwig
      bw_file = '%s/tracks/t%d_true.bw' % (options.out_dir, ti)
      bigwig_write(
          bw_file,
          test_targets_ti,
          options.track_bed,
          options.genome_file,
          bed_set=bed_set)

      # make predictions bigwig
      bw_file = '%s/tracks/t%d_preds.bw' % (options.out_dir, ti)
      bigwig_write(
          bw_file,
          test_preds[:, :, ti],
          options.track_bed,
          options.genome_file,
          model.hp.batch_buffer,
          bed_set=bed_set)


def bigwig_open(bw_file, genome_file):
  """ Open the bigwig file for writing and write the header. """

  bw_out = pyBigWig.open(bw_file, 'w')

  chrom_sizes = []
  for line in open(genome_file):
    a = line.split()
    chrom_sizes.append((a[0], int(a[1])))

  bw_out.addHeader(chrom_sizes)

  return bw_out


def bigwig_write(bw_file,
                 signal_ti,
                 track_bed,
                 genome_file,
                 buffer=0,
                 bed_set='test'):
  """ Write a signal track to a BigWig file over the regions
         specified by track_bed.

    Args
     bw_file:     BigWig filename
     signal_ti:   Sequences X Length array for some target
     track_bed:   BED file specifying sequence coordinates
     genome_file: Chromosome lengths file
     buffer:      Length skipped on each side of the region.
    """

  bw_out = bigwig_open(bw_file, genome_file)

  si = 0
  bw_hash = {}

  # set entries
  for line in open(track_bed):
    a = line.split()
    if a[3] == bed_set:
      chrom = a[0]
      start = int(a[1])
      end = int(a[2])

      preds_pool = (end - start - 2 * buffer) // signal_ti.shape[1]

      bw_start = start + buffer
      for li in range(signal_ti.shape[1]):
        bw_end = bw_start + preds_pool
        bw_hash.setdefault((chrom,bw_start,bw_end),[]).append(signal_ti[si,li])
        bw_start = bw_end

      si += 1

  # average duplicates
  bw_entries = []
  for bw_key in bw_hash:
    bw_signal = np.mean(bw_hash[bw_key])
    bwe = tuple(list(bw_key)+[bw_signal])
    bw_entries.append(bwe)

  # sort entries
  bw_entries.sort()

  # add entries
  for line in open(genome_file):
    chrom = line.split()[0]

    bw_entries_chroms = [be[0] for be in bw_entries if be[0] == chrom]
    bw_entries_starts = [be[1] for be in bw_entries if be[0] == chrom]
    bw_entries_ends = [be[2] for be in bw_entries if be[0] == chrom]
    bw_entries_values = [float(be[3]) for be in bw_entries if be[0] == chrom]

    if len(bw_entries_chroms) > 0:
      bw_out.addEntries(
          bw_entries_chroms,
          bw_entries_starts,
          ends=bw_entries_ends,
          values=bw_entries_values)

  bw_out.close()


def make_data_ops(job, tfr_pattern):
  def make_dataset(pat, mode):
    return dataset.DatasetSeq(
      tfr_pattern,
      job['batch_size'],
      job['seq_length'],
      job['target_length'],
      mode=mode)

  test_dataseq = make_dataset(tfr_pattern, mode=tf.estimator.ModeKeys.EVAL)
  test_dataseq.make_iterator_structure()
  data_ops = test_dataseq.iterator.get_next()
  test_init_op = test_dataseq.make_initializer()

  return data_ops, test_init_op


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
