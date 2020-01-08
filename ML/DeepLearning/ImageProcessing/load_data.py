import pickle
from sklearn.model_selection import train_test_split
from scipy import misc
import imutils
import numpy as np
import os,cv2

path=r"E:\Machine Learning\examples\Modaka_Tech_AI_assignment\data\train"
label = os.listdir(path)
print(label)
dataset=[]
for image_label in label:
    images = os.listdir(path+"/"+image_label)
    for image in images:
        # print(image_label)
        img = cv2.imread(path+"/"+image_label+"/"+image)
        for k in np.arange(0,360,30):
            img=imutils.rotate_bound(img,k)
            img = cv2.resize(img, (150, 150))
            dataset.append((img,image_label))

X=[]
Y=[]

for input,image_label in dataset:
    X.append(input)
    Y.append(label.index(image_label))


X=np.array(X)
Y=np.array(Y)
# print(Y)

X_train,y_train,  = X,Y
print(y_train.shape)

data_set=(X_train,y_train)

path=r"E:\Machine Learning\examples\Modaka_Tech_AI_assignment\data\test"
label = os.listdir(path)
print(label)
dataset=[]
for image_label in label:
    images = os.listdir(path+"//"+image_label)
    for image in images:
        img = cv2.imread(path+"//"+image_label+"//"+image)
        for k in np.arange(0,360,30):
            img=imutils.rotate_bound(img,k)
            img = cv2.resize(img, (150, 150))
            dataset.append((img,image_label))

X=[]
Y=[]

for input,image_label in dataset:
    X.append(input)
    Y.append(label.index(image_label))

X=np.array(X)
Y=np.array(Y)

print(y_train.shape)
X_train,y_train,  = X,Y


test_data_set=(X_train,y_train)



save_label = open("int_to_word_out.pickle","wb")
pickle.dump(label, save_label)
save_label.close()
