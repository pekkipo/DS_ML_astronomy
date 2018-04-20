"""
Represents a dataset of HDF5 file.
"""

import h5py
from pathlib import Path
from dal.dataset import Dataset
chunk_size=32
class Hdf5Dataset(Dataset):
    """
    Dataset of HDF5 file
    """
    def __init__(self,path,shape=None,chunks=chunk_size):
        path = Path(path)
        file_exists = path.exists()
        self.file = h5py.File(path,mode='a')
        entries=0
        if file_exists:
            self.ds = self.file[path.stem]
            entries = self.ds.shape[0]
            self.size=entries
        else:
            max_shape = (None,)+shape
            chunk_shape = (chunk_size,) + shape
            self.ds = self.file.create_dataset(path.stem,shape=chunk_shape,maxshape=max_shape,chunks=chunk_shape)
            self.size = chunk_size
        super(Hdf5Dataset, self).__init__(entries,chunk_size)

    def insert(self,entry):
        if self.size<=self.entries:
            self.ds.resize(self.entries + chunk_size, axis=0)
            self.size = self.entries+chunk_size
        self.ds[self.entries] = entry
        self.entries += 1

    def close(self):
        self.ds.resize(self.entries,axis=0)
        self.ds.flush()
        self.file.close()

    @property
    def shape(self):
        return self.ds.shape

    def __len__(self):
        return self.entries

    def __iter__(self):
        for entry in self.ds:
            yield entry

    def __getitem__(self, item):
        return self.ds[item]

    def __setitem__(self, key, value):
        self.ds[key] = value