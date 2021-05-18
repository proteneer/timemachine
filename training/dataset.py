from typing import Tuple, Any, List

import numpy as np

class Dataset():
    """Dataset is a utility for training. It provides some utilities to split
    data and create batches of data.
    """

    def __init__(self, data: List[Any]):
        self.data = data

    def num_batches(self, batch_size: int):
        return np.math.ceil(len(self.data) / batch_size)

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        np.random.shuffle(self.data)

    def iterbatches(self, batch_size: int) -> List[Any]:
        for batch in range(self.num_batches(batch_size)):
            start = batch*batch_size
            end = min((batch+1)*batch_size, len(self.data))
            yield self.data[start:end]

    def split(self, frac: float) -> Tuple["Dataset", "Dataset"]:
        """Split dataset into two, using a fraction.


        Parameters
        ----------

        frac: float
            The proportion of the dataset to have to split on.

        Returns
        -------
        tuple(Dataset, Dataset)
            The split datasets, the first dataset containing the fractional percentage
            and the second with the leftover.

        """
        if frac > 1.0 or frac < 0.0:
            raise ValueError("frac must be between 0 and 1")
        split_idx = int(len(self.data)*frac)
        train_dataset = self.data[:split_idx]
        test_dataset = self.data[split_idx:]
        return Dataset(train_dataset), Dataset(test_dataset)

    def random_split(self, frac: float) -> Tuple["Dataset", "Dataset"]:
        """Shuffle data and split into two datasets, using a fraction.


        Parameters
        ----------

        frac: float
            The proportion of the dataset to have to split on.

        Returns
        -------
        tuple(Dataset, Dataset)
            The split datasets, the first dataset containing the fractional percentage
            and the second with the leftover

        """
        inds = np.arange(len(self))
        np.random.shuffle(inds)
        split_idx = int(len(self.data)*frac)
        return self.indices_split(inds[:split_idx], inds[split_idx:])

    def indices_split(self, left: List[int], right: List[int]) -> Tuple["Dataset", "Dataset"]:
        """Split dataset into two, providing the indices. Used when the data needs to
        be split in more specific ways that the Dataset class does not provide out of the box.


        Parameters
        ----------

        left: array of integers
            Indices of dataset to construct first dataset from.

        right: array of integers
            Indices of dataset to construct second dataset from.

        Returns
        -------
        tuple(Dataset, Dataset)
            The split datasets, the first dataset containing the objects at the indices
            provided by left and the second with the right indices.

        """
        left = set(left)
        right = set(right)
        indices = set(range(len(self)))
        if len(left.intersection(right)) > 0:
            raise ValueError("Left and right indices contain overlap")
        if len(left.union(right)) != len(indices) or len(left.union(right).difference(indices)) > 0:
            raise ValueError("Indices provided don't match dataset indices")
        left_split = [self.data[i] for i in left]
        right_split = [self.data[i] for i in right]
        return Dataset(left_split), Dataset(right_split)
