from googleapiclient import discovery, errors
import pickle
import numpy as np
import time
with open("../resources/test_faces.pickle", "rb") as handle:
    x_test = pickle.load(handle)
ar = np.reshape(x_test[0], (1, 224, 224, 3)).tolist()

service = discovery.build('ml', 'v1')
name = 'projects/{}/models/{}'.format("infra-optics-247216", "race_model")
emb_model = "full_model"
if emb_model is not None:
    name += '/versions/{}'.format(emb_model)
start = time.time()
response = service.projects().predict(
    name=name,
    body={'instances': ar}
).execute()
stop = time.time()
print(stop - start)
if 'error' in response:
    raise RuntimeError(response['error'])

print(response['predictions'])