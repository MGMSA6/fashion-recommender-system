import os
import pickle

import numpy as np
import streamlit as st
import tensorflow
from PIL import Image
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image

# Dataset
# https://www.kaggle.com/paramaggarwal/fashion-product-images-small

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

st.title('Fashion recommender system')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path, model):
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


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')

    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices


# Steps
# 1. file upload -> save
uploaded_file = st.file_uploader("Choose an image")

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        # 2. load file -> feature extract
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # 3. recommendation
        indices = recommend(features, feature_list)

        # 4. show
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occurred in file upload")

# Optimize
# 44k to 10m
# Use Annoy (Spotify)

# Deploy
# storing 44k images
# Use AWS (S3)
