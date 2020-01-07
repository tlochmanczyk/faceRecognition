import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
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
        if confidence > 0.5:
            only_face = image[startY:endY, startX:endX]
            only_face = cv2.resize(only_face, (224, 224))
            if only_face.shape == (224, 224, 3):
                return only_face
    return False


def train_test_split_races(face_list, race_list):
    split_dict = {}
    for n_face, n_race in zip(face_list, race_list):
        if n_race not in split_dict.keys():
            split_dict[n_race] = [n_face]
        else:
            split_dict[n_race].append(n_face)
    all_train_faces = []
    all_train_races = []
    all_test_faces = []
    all_test_races = []
    for n_race, faces in split_dict.items():
        x_train, x_test = faces[:int(len(faces)*0.7)], faces[int(len(faces)*0.7):]
        y_train, y_test = [n_race] * len(x_train), [n_race] * len(x_test)
        all_train_faces += x_train
        all_test_faces += x_test
        all_train_races += y_train
        all_test_races += y_test

    all_train_faces, all_train_races = shuffle(all_train_faces, all_train_races)
    all_test_faces, all_test_races = shuffle(all_test_faces, all_test_races)
    return all_train_faces, all_train_races, all_test_faces, all_test_races


if __name__ == "__main__":
    input_images = []
    races = []
    for face in tqdm(all_faces_paths):
        path = FACES_PATH + face
        try:
            img = get_face(path)
            if type(img) == bool:
                continue
            pre = re.search(r'[0-9]*_[0-9]*_', face).group()
            post = re.search(r'_[0-9]*\..*', face).group()
        except:
            continue
        race = face.replace(pre, '').replace(post, '')
        if len(race) > 1 or race == '4':
            continue
        input_images.append(img)
        races.append(race)
        if img.shape != (224, 224, 3):
            print("dupa")

    X_train, y_train, X_test, y_test = train_test_split_races(input_images, races)

    with open(RESOURCES + 'train_faces.pickle', 'wb') as handle:
        pickle.dump(X_train, handle)
    with open(RESOURCES + 'test_faces.pickle', 'wb') as handle:
        pickle.dump(X_test, handle)
    with open(RESOURCES + 'train_races.pickle', 'wb') as handle:
        pickle.dump(y_train, handle)
    with open(RESOURCES + 'test_races.pickle', 'wb') as handle:
        pickle.dump(y_test, handle)
