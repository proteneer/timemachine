import numpy as np

class Dataset():

    def __init__(self, data):
        self.data = data

    def num_batches(self, batch_size):
        return np.math.ceil(len(self.data) / batch_size)

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        np.random.shuffle(self.data)

    def iterbatches(self, batch_size):
        batch = 0
        for _ in range(self.num_batches(batch_size)):
            start = batch*batch_size
            end = min((batch+1)*batch_size, len(self.data))
            yield self.data[start:end]
            batch += 1

    def split(self, frac):
        split_idx = int(len(self.data)*frac)
        train_dataset = self.data[:split_idx]
        test_dataset = self.data[split_idx:]
        return Dataset(train_dataset), Dataset(test_dataset)
