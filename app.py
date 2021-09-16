import os
import pickle

import numpy as np
import tensorflow
from numpy.linalg import norm
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Dataset
# https://www.kaggle.com/paramaggarwal/fashion-product-images-small

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

print(model.summary())


def extract_features(img_path, model):
    # load img
    img = image.load_img(img_path, target_size=(224, 224))

    # convert img to array
    img_array = image.img_to_array(img)

    # expand dimension to to 4d
    expanded_img_array = np.expand_dims(img_array, axis=0)

    # preprocess img to make suitable for ResNet50 input type
    preprocessed_img = preprocess_input(expanded_img_array)

    # predict
    result = model.predict(preprocessed_img).flatten()

    # normalize img
    normalized_result = result / norm(result)

    return normalized_result


filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
