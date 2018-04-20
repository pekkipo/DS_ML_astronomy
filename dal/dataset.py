"""
Interface for dataset abstraction
"""

from abc import ABC,abstractclassmethod

class Dataset(ABC):
    def __init__(self,num_entries,chunk_size):
        self.entries=num_entries
        self.chunk_size=chunk_size

    @property
    @abstractclassmethod
    def shape(self):
        """
        Shape of the dataset (number of entries, data shape)
        :return: tuple representing shape
        """
        pass

    @abstractclassmethod
    def __len__(self):
        """
        Number of samples in the dataset
        :return:
        """
        pass
    @abstractclassmethod
    def insert(self,entry):
        """
        Inserts a new entry to the dataset
        :param entry: the data to insert (Should match the shape of the dataset)
        :return:
        """
        pass

    @abstractclassmethod
    def close(self):
        """
        Used to save all data and safely close connections.
        :return:
        """
        pass

    @abstractclassmethod
    def __getitem__(self, item):
        pass

    @abstractclassmethod
    def __setitem__(self, key, value):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()