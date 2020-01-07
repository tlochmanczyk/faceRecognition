from keras.applications.vgg16 import VGG16
from keras_vggface.vggface import VGGFace
from keras.models import Model
import pickle
import numpy as np
from tqdm import tqdm

RESOURCES = "../resources/"
TRAIN_PATH = RESOURCES + "train_faces.pickle"
TEST_PATH = RESOURCES + "test_faces.pickle"
shape_img = (1, 224, 224, 3)


def generate_embs(pickle_file, vgg):
    with open(pickle_file, "rb") as handle:
        faces = pickle.load(handle)

    face_embs = []
    for face in tqdm(faces):
        face = np.reshape(face, shape_img)
        face_embs.append(vgg.predict(face))

    return face_embs


if __name__ == "__main__":
    model = VGGFace(model="resnet50")
    layer_name = "flatten_1"
    model = Model(model.input, model.get_layer(layer_name).output)

    print(model.summary())

    with open(RESOURCES + "train_embeddings.pickle", "wb") as handle:
        pickle.dump(generate_embs(TRAIN_PATH, model), handle)
    with open(RESOURCES + "test_embeddings.pickle", "wb") as handle:
        pickle.dump(generate_embs(TEST_PATH, model), handle)
