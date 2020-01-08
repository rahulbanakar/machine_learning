import os,glob as gb
import numpy as np
from keras.models import load_model
import time,cv2
import pickle

classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()

#Define Path
model_path = 'airprobe/model/model.h5'
model_weights_path = 'airprobe/model/weights.h5'
test_path = 'test_model'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

start = time.time()
#Prediction Function

path = 'C:\Rahul\machine learning\Hachathon\AirprobeAssignment\data\model'
labels = os.listdir(path)
result = []
for images in labels:
    image = cv2.imread(path+'//'+images)
    image = cv2.resize(image, (150, 150))
    image=np.array([image])
    image = image.astype('float32')
    image = image / 255.0
    prediction=model.predict(image)
    result.append(prediction)
    print("Output is : ",int_to_word_out[np.argmax(prediction)])


#Calculate execution time
end = time.time()
dur = end-start

print("Execution Time: ",dur," seconds")
