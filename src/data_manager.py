import numpy as np
import train_images_data as X
import train_labels_data as Y


def load_data():
    train_set_images = np.array(X.train_img_data[:])
    train_set_labels = np.array(Y.train_img_data)