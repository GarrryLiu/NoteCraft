import joblib

from recognition.train import extract_hog_features


def preprocess_image(img):
    features = extract_hog_features(img)
    return features.reshape(1, -1)


def load_model(path):
    return joblib.load(path)


def predict(img, model):
    features = preprocess_image(img)
    prediction = model.predict(features)
    return prediction[0]
