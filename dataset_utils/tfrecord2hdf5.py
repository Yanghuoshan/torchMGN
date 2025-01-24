#!/usr/bin/env python3
from absl import app
from absl import flags
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import json
import functools
import h5py

""" 
The data format is as follows:

material_dataset/
├── metadata.json -> dataset features
├── train/
│   ├── metadata.json
│   └── dataset.h5
├── valid/
│   ├── metadata.json -> timesteps per ex
│   └── dataset.h5
├── test/
│   ├── metadata.json
│   └── dataset.h5
└── ...

train.h5 = {ex0:dict, ex1:dict, ...}

"""

FLAGS = flags.FLAGS
flags.DEFINE_string('in_dir', '../tmp/datasets/', 'Input datasets directory')
flags.DEFINE_string('out_dir', '../tmp/datasets_hdf5/','Output numpy datasets directory')
flags.DEFINE_string('dataset_name', 'deforming_plate', '')
flags.DEFINE_string('split', 'valid', '')
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


def load_dataset(dataset_name, split):
    with open(os.path.join(dataset_name, 'metadata.json'), 'r') as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(dataset_name, f'{split}.tfrecord'))
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
    out_path = os.path.join(out_dir, dataset_name, split)

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"in_path {in_path} doesn't exixt")

    if not os.path.exists(out_path):
        os.makedirs(out_path)


    print("in_path:", in_path)
    print("out_path:", out_path)

    ds = load_dataset(in_path, split)
    first = True

    outfile = os.path.join(out_path, 'metadata.json')
    outh5 = os.path.join(out_path, f'dataset.h5')
    with h5py.File(outh5, 'w') as h5f:
        with open(outfile, 'w') as meta_file:
            print('{', file=meta_file)
            print('    "files": {', file=meta_file)
            for i, record in enumerate(ds):
                print(f'==== Record {i} ====')
                for k, v in record.items():
                    print(k, v.shape)
                    h5f.create_dataset(f"ex{i}/{k}", data=v)

                ns = record['cells'].shape[0]

                if first: first = False
                else: print(',', file=meta_file)
                
                # np.savez_compressed(os.path.join(out_path, f'ex{i}.npz'), **record)
                print(f'        "ex{i}": {ns}', file=meta_file, end='')

            print('', file=meta_file)
            print('    }', file=meta_file)
            print('}', file=meta_file)

if __name__ == "__main__":
    app.run(main)



