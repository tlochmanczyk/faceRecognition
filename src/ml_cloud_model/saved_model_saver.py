from keras_vggface.vggface import VGGFace
from keras.models import Model, load_model, Sequential
from keras.layers import Input
import tensorflow as tf
RESOURCES = "../resources/"

layer_name = "flatten_1"
model = VGGFace(model="resnet50")
model = Model(model.input, model.get_layer(layer_name).output)
model.save(RESOURCES + "temp_embs.h5")

emb_model = tf.keras.models.load_model(RESOURCES + "temp_embs.h5")
clf = tf.keras.models.load_model("../resources/race_model.h5")

tf.saved_model.save(clf, RESOURCES + "race_classifier/")
tf.saved_model.save(emb_model, RESOURCES + "race_embeddings_generator/")




