import os
import random
from glob import glob

import cv2
import joblib
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def extract_hog_features(img, target_img_size=(100, 100)):
    """
    Extract HOG features from an image for classification.

    Parameters:
    - img: Input image.
    - target_img_size: Size to which the image will be resized.

    Returns:
    - h: Flattened HOG feature vector.
    """
    img = cv2.resize(img, target_img_size)
    win_size = (100, 100)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)

    block_size = (
        block_size_in_cells[1] * cell_size[1],
        block_size_in_cells[0] * cell_size[0],
    )
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()


def read_data(data_path):
    """
    Read images and their corresponding labels from the specified directory.

    Parameters:
    - data_path: Path to the directory containing images.

    Returns:
    - imgs: List of loaded images.
    - labels: List of corresponding labels.
    """
    imgs = []
    labels = []

    # Iterate through each subdirectory in the base directory
    for subdir in os.listdir(data_path):
        subdir_path = os.path.join(data_path, subdir)
        # Check if the path is a directory
        if os.path.isdir(subdir_path):
            # Extract the label from the subdirectory name (xxx)
            label = "_".join(subdir.split("_")[:-1])
            # print(label)
            # Get all PNG images in the subdirectory
            image_files = glob(os.path.join(subdir_path, "*.png"))

            # Append image paths and labels to the lists
            for image_file in image_files:
                img = cv2.imread(image_file)
                if img is not None:
                    imgs.append(img)
                    labels.append(label)
                else:
                    print(f"Warning: Could not read image {image_file}")

    if not imgs:
        raise ValueError(f"No valid images found in directory {data_path}")

    print(f"Total images loaded: {len(imgs)}")
    return imgs, labels


def load_classifiers():
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    classifiers = {
        "SVM": svm.LinearSVC(random_state=random_seed),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "NN": MLPClassifier(
            activation="relu",
            hidden_layer_sizes=(200,),
            max_iter=10000,
            alpha=1e-4,
            solver="adam",
            verbose=20,
            tol=1e-8,
            random_state=1,
            learning_rate_init=0.0001,
            learning_rate="adaptive",
        ),
    }
    return classifiers, random_seed


def train_model(labels, imgs, classifier):

    features = []
    for img in imgs:
        features.append(extract_hog_features(img))

    classifiers, random_seed = load_classifiers()
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=random_seed
    )
    model = classifiers[classifier]
    print("############## Training", classifier, "##############")
    model.fit(train_features, train_labels)

    joblib.dump(model, f"{classifier}_model.pkl")

    accuracy = model.score(test_features, test_labels)
    print(classifier, "accuracy:", accuracy * 100, "%")


def load_model(classifier):
    return joblib.load(f"{classifier}_model.pkl")


def main():
    imgs, labels = read_data("../data/data_")
    train_model(labels, imgs, "NN")

    # model = load_model("NN")
    # You can now use `model` to make predictions on new data


if __name__ == "__main__":
    main()
