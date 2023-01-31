import data_lib


class DataSetup:
    def __init__(self, data_name) -> None:
        print('-' * 100)
        print('[INFO] SETTING UP DATASET ...')
        
        self.dataset_name = data_name

        (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)

    def setup(self):
        self.preprocessing_data()

    def preprocessing_data(self):
        if self.dataset_name == 'mnist':
            print('Using mnist dataset ...')
            (self.x_train, self.y_train), (self.x_test, self.y_test) = data_lib.mnist_load_data()

    def get_data(self):
        return (self.x_train, self.y_train), (self.x_test, self.y_test)