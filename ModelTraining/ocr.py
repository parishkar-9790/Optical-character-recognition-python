DEBUG = False
if DEBUG:
    from PIL import Image
    import numpy as np

    def read_image(path):
        return np.asarray(Image.open(path).convert('L'))

    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)
# change the data_dir accordingly
DATA_DIR = 'X:\\Machine_Learning\\OCR\ModelTraining\\data\\'
TEST_DIR = 'X:\\Machine_Learning\\OCR\ModelTraining\\test\\'
TEST_DATA_FILENAME = DATA_DIR+'t10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR+'t10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR+'train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR+'train-labels.idx1-ubyte'


def bytes_to_ints(byte_data):
    return int.from_bytes(byte_data, 'big')


def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb')as f:
        _ = f.read(4)  # magic number
        n_images = bytes_to_ints(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_ints(f.read(4))
        n_columns = bytes_to_ints(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb')as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_ints(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_ints(f.read(1))
            labels.append(label)
    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return[flatten_list(sample) for sample in X]


def dist(x, y):
    # return the list [x,y]
    return sum([
        (bytes_to_ints(X_i)-bytes_to_ints(Y_i))**2
        for X_i, Y_i in zip(x, y)
    ]
    )**(.5)


def get_training_distances_for_test_sample(X_train, test_sample):
    return[dist(train_sample, test_sample)for train_sample in X_train]


def get_most_frequent_element(l):
    return max(l, key=l.count)


def knn(X_train, Y_train, X_test, k):
    Y_pred = []  # predicted lables
    for test_sample_idx, test_sample in enumerate(X_test):
        training_distance = get_training_distances_for_test_sample(
            X_train, test_sample)
        # Y_sample = whatever()
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(enumerate(training_distance), key=lambda x:x[1])
        ]
        candidates = [
            Y_train[idx]
            for idx in sorted_distance_indices[:k]
        ]
        top_candidate = get_most_frequent_element(candidates)
        Y_pred.append(top_candidate)
    return Y_pred


def main():
    X_train = read_images(TRAIN_DATA_FILENAME, 30000)
    Y_train = read_labels(TRAIN_LABELS_FILENAME, 30000)
    X_test = read_images(TEST_DATA_FILENAME, 300)
    Y_test = read_labels(TEST_LABELS_FILENAME, 300)
    if DEBUG:
        for idx, test_sample in enumerate(X_test):
            write_image(test_sample, f'{TEST_DIR}{idx}.png')
        X_test = [read_image(f'{TEST_DIR}parishkar.png')]
        Y_test = [1]

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)
    Y_pred = knn(X_train, Y_train, X_test, 7)
    accuracy = sum([
        int(Y_pred_I) == (Y_test_I) for Y_pred_I, Y_test_I in zip(Y_pred, Y_test)])/len(Y_test)
    print(Y_pred)
    # print(Y_test)
    print(f'Accuracy is {accuracy*100}%')


if __name__ == '__main__':
    main()
