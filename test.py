import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
import tensorflow
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))

filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPool2D()
])

# load img
img = image.load_img('sample/watch.jpeg', target_size=(224, 224))

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

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')

neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])


for file in indices[0][1:6]:
    print(filenames[file])
    tem_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(tem_img, (512, 512)))
    cv2.wait(0)
