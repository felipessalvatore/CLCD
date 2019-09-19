from torchtext import data


class DataHolder():
    """
    Class to store all data using the data.BucketIterator class.

    """
    def __init__(self,
                 config,
                 train,
                 valid,
                 test):
        self.train_iter = data.BucketIterator(train,
                                              batch_size=config.batch_size,
                                              repeat=False)
        self.valid_iter = data.BucketIterator(valid,
                                              batch_size=config.batch_size,
                                              repeat=False)
        self.test_iter = data.BucketIterator(test,
                                             batch_size=len(test),
                                             repeat=False)

