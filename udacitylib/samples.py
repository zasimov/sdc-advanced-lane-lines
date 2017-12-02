"""samples is a module =)
"""

import collections
import pickle

import numpy as np
from sklearn.utils import shuffle

try:
    import h5py
except ImportError:
    h5py = None

try:
    import scipy.io as scipy_io
except ImportError:
    scipy_io = None


class Samples(collections.namedtuple('Samples', ['features', 'targets'])):
    """Samples struct describes a set of feature-target pairs

         - features - train, valid or test features (numpy.ndarray)
         - targets- train, valid or test targets (numpy.ndarray)
    """

    def __len__(self):
        return len(self.features)

    @property
    def targets_set(self):
        """targets_set returns a set of targets
        """
        return frozenset(self.targets)

    def shuffle(self):
        features, targets = shuffle(self.features, self.targets)
        return Samples(features, targets)

    def append(self, samples):
        return Samples(np.append(self.features, samples.features, axis=0),
                       np.append(self.targets, samples.targets, axis=0))

    def batches(self, batch_size):
        for start in range(0, len(self), batch_size):
            features = self.features[start:start + batch_size]
            targets = self.targets[start:start + batch_size]

            yield Samples(features, targets)

    def raw_batches(self, batch_size):
        for batch in self.batches(batch_size):
            yield batch.features, batch.targets

    def map(self, fn):
        features = []
        for feature in self.features:
            features.append(fn(feature))
        features = np.array(features)
        return Samples(features, self.targets)

    def save_pickle(self, save_path):
        with open(save_path, 'wb') as f:
            dataset = {
                'features': self.features,
                'targets': self.targets,
            }
            pickle.dump(dataset, f)

    def save_hdf5(self, save_path):
        if not h5py:
            raise RuntimeError('h5py is not installed')
        out = h5py.File(save_path)
        try:
            out.create_dataset('features', data=np.array(self.features))
            out.create_dataset('targets', data=np.array(self.features))
        finally:
            out.close()

    def save_mat(self, save_path):
        if not scipy_io:
            raise RuntimeError('scipy.io is not installed')
        mat = dict(
            features=self.features,
            targets=self.targets,
        )
        scipy_io.savemat(save_path, mat)


def load_pickle(file_name):
    """load_pickle loads the data set from pickle file

    Pickle file should contain dictionary with following keys

      - features - list of features
      - targets - list of targets

    load_pickle returns Samples struct
    """
    with open(file_name, 'rb') as infile:
        dataset = pickle.load(infile)
    return Samples(dataset['features'], dataset['targets'])


def load_hdf5(file_name, group=None):
    """load_hdf5 loads the data set from HDF5 file

    HDF5 file should contain 

      - features dataset
      - targets dataset

    load_hdf5 returns Samples struct
    """
    if not h5py:
        raise RuntimeError('h5py is not installed')

    h5 = h5py.File(file_name, 'r')
    try:
        ds = h5[group] if group else h5

        features = np.array(ds['features'])
        targets = np.array(ds['targets'])

        return Samples(features, targets)
    finally:
        h5.close()


class HDF5Samples:

    FEATURES = 'features'
    TARGETS = 'targets'

    def __init__(self, file_name, do_not_open=False):
        self.file_name = file_name
        if not do_not_open:
            self._h5 = h5py.File(file_name, 'r')
        else:
            self._h5 = None

    def open(self):
        if self._h5:
            return
        self._h5 = h5py.File(self.file_name, 'r')

    def close(self):
        self._h5.close()
        self._h5 = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        return self.close()

    def group(self, group_name, features_fn=None, targets_fn=None):
        """group returns Samples from specific group"""
        features = self._h5[group_name][self.FEATURES]
        targets = self._h5[group_name][self.TARGETS]

        if features_fn:
            features = features_fn(features)

        if targets_fn:
            targets = targets_fn(targets)

        return Samples(features, targets)

    @property
    def features(self):
        return self.group[self.FEATURES]

    @property
    def targets(self):
        return self.group[self.TARGETS]

