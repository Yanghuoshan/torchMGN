import numpy as np
from absl import app
from absl import flags
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import json
import functools


""" 
The data format is as follows:

simulation_data = {
    "simulation_0": {
        "mesh_pos": (ntimestep, nnodes, ndims),
        "world_pos": (ntimestep, nnodes, ndims),
        "node_type": (ntimestep, nnodes, 1),
        "cells": (ntimestep, ncells, nnodes_per_cell),
        ...
    },
    "simulation_1": {
        ...
    },
    "simulation_n": {
        ...
    }
}
"""

tf._logging

FLAGS = flags.FLAGS
flags.DEFINE_string('in_dir', '../tmp/datasets/', 'Input datasets directory')
flags.DEFINE_string('out_dir', '../tmp/datasets_npdict/','Output numpy datasets directory')
flags.DEFINE_string('dataset_name', 'flag_simple', '')
flags.DEFINE_string('split', 'train', '')
flags.DEFINE_boolean('debug', False, 'Enable debugging output')

def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string)
                    for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
        out[key] = data
    return out


def load_dataset(dataset_name, split, meta):
    
    ds = tf.data.TFRecordDataset(os.path.join(dataset_name, split+'.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds


def convert_to_npdict(example) -> dict:
    result = {}
    # example.features.feature is the dictionary
    for key, feature in example.features.feature.items():
        # The values are the Feature objects which contain a `kind` which contains:
        # one of three fields: bytes_list, float_list, int64_list

        kind = feature.WhichOneof('kind')
        result[key] = np.array(getattr(feature, kind).value)

    return result

def main(argv):
    in_dir = FLAGS.in_dir
    out_dir = FLAGS.out_dir
    dataset_name = FLAGS.dataset_name
    split = FLAGS.split

    in_path = os.path.join(in_dir, dataset_name)
    out_path = os.path.join(out_dir, dataset_name)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"in_path {in_path} doesn't exixt")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print("in_path:", in_path)
    print("out_path:", out_path)

    with open(os.path.join(in_path, 'metadata.json'), 'r') as fp:
        meta = json.loads(fp.read())

    ds = load_dataset(in_path, split, meta)
    # first = True

    # outfile = os.path.join(out_path, 'metadata.json')
    numpyDict = dict()
    
    
    for i, record in enumerate(ds):
        print(f'==== Record {i} ====')
        record_np = dict()
        for k, v in record.items():
            print(k, v.shape)
            record_np[k]=v.numpy()
        numpyDict[f"simulation_{i}"] = record_np

    np.savez_compressed(os.path.join(out_path, f'{split}.npz'), **numpyDict)


    with open(os.path.join(out_path, 'metadata.json'), 'w') as fp:
        fp.write(json.dumps(meta))

if __name__ == "__main__":
    app.run(main)



