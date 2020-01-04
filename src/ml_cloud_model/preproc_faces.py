import numpy as np
import os
from sklearn.model_selection import train_test_split
import re
import pickle
from tqdm import tqdm
import cv2

FACES_PATH = '../../face-dataset/'
RESOURCES = '../resources/'
model = cv2.dnn.readNetFromCaffe(RESOURCES + 'deploy.prototxt', RESOURCES + 'weights.caffemodel')

all_faces_paths = os.listdir(FACES_PATH)


def get_face(image_path):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0, (224, 224), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, show box around face
        if (confidence > 0.5):
            only_face = image[startY:endY, startX:endX]
            only_face = cv2.resize(only_face, (224, 224))

            return only_face


if __name__ == "__main__":
    input_images = []
    races = []
    for face in tqdm(all_faces_paths):
        path = FACES_PATH + face
        try:
            img = get_face(path)
            pre = re.search(r'[0-9]*_[0-9]*_', face).group()
            post = re.search(r'_[0-9]*\..*', face).group()
        except:
            continue
        race = face.replace(pre, '').replace(post, '')
        input_images.append(img)
        races.append(race)

    X_train, X_test, y_train, y_test = train_test_split(input_images, races, test_size=0.3, random_state=42)

    with open(RESOURCES + 'train_faces.pickle', 'wb') as handle:
        pickle.dump(X_train, handle)
    with open(RESOURCES + 'test_faces.pickle', 'wb') as handle:
        pickle.dump(X_test, handle)
    with open(RESOURCES + 'train_races.pickle', 'wb') as handle:
        pickle.dump(y_train, handle)
    with open(RESOURCES + 'test_races.pickle', 'wb') as handle:
        pickle.dump(y_test, handle)
