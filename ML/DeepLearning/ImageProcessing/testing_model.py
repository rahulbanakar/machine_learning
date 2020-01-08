import os,glob as gb
import numpy as np
from keras.models import load_model
import time,cv2
import pickle

classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()

#Define Path
model_path = 'models_with_augmentation/model.h5'
model_weights_path = 'models_with_augmentation/weights.h5'
test_path = 'test_model'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

start = time.time()
#Prediction Function

Image_Folder=r"test_model"
img=gb.glob(Image_Folder+"/*.png")+gb.glob(Image_Folder+"/*.jpg")
for k in img:
    print(k)
    image=cv2.imread(k)
    image = cv2.resize(image, (150, 150))
    image=np.array([image])
    image = image.astype('float32')
    image = image / 255.0
    prediction=model.predict(image)
    print("Output is : ",int_to_word_out[np.argmax(prediction)])
    print()

#Calculate execution time
end = time.time()
dur = end-start

print("Execution Time:",dur,"seconds")
