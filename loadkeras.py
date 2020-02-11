from keras.models import load_model

model=load_model("model_latest.h5")


import numpy as np

test_mat=np.load("test_matrix.npy")

test_mat=np.transpose(test_mat,(0,2,1))


print(model.predict_prob(test_mat))
