from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

import os
import pickle

path = 'E:\\Scrinysmieci2.0'
folder_content = os.listdir(path)
img_arr = np.empty([1,224,224,3])
for file in folder_content:
    if file.endswith(".jpg") or file.endswith(".JPG"):
        img = image.load_img(path + '\\' + file, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_arr = np.append(img_arr, img_data, axis = 0)
          
img_arr = preprocess_input(img_arr)

model = VGG16(weights='E:\\Nauka\\DL\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
vgg16_feature = model.predict(img_arr)

vgg16_feature.dump('embeddings.pickle')


