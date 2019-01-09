import random

class ReservoirSampler():

    def __init__(self, generator, k):
        self.generator = generator # generator
        self.k = k # number of samples we want to keep
        self.R = []
        self.count = 0

    def sample(self):
        for item in self.generator:
            if self.count < self.k:
                self.R.append(item)
            else:
                j = random.randint(0, self.count)
                if j < self.k:
                    self.R[j] = item

            self.count += 1
            yield item
