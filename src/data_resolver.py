import load_data


class Data_Resolver:

    def __init__(self, standardize=True):
        """
        Instantiates Data_Resolver instance and loads image labeled dataset for the supervised learning purposes

        """
        train_img_data, train_l_data, test_img_data, test_l_data, classes = self.flatten_data(self.__load_data())
        if standardize is True:
            train_img_data, test_img_data = self.standardize_data(train_img_data, test_img_data)
        self.train_image_data, self.train_label_data = train_img_data, train_l_data
        self.test_image_data, self.test_label_data, self.classes = test_img_data, test_l_data, classes

        self.training_examples = train_img_data.shape[0]
        self.image_size = train_img_data.shape[1] # containing image resolution and color channels
        self.test_examples = test_img_data[0]

    def __load_data(self):
        train_img_data, train_label_data, test_img_data, test_label_data, classes = load_data.load_data()
        return train_img_data, train_label_data, test_img_data, test_label_data, classes

    def flatten_data(self, data):
        train_img_data, train_label_data, test_img_data, test_label_data, classes = data
        train_img_data_flatten = train_img_data.reshape(train_img_data.shape[0], -1).T
        test_img_data_flatten = test_img_data.reshape(test_img_data.shape[0], -1).T
        return train_img_data_flatten, train_label_data, test_img_data_flatten, test_label_data, classes

    def standardize_data(self, train_img_data, test_img_data):
        train_img_std = train_img_data / 255
        test_img_std = test_img_data / 255
        return train_img_std, test_img_std