import os,glob
import numpy as np
import sklearn
from sklearn import model_selection
from PIL import Image


classes=["nako","sakura","hitomi"]
num_classes=len(classes)
image_resize=50
X=[]
Y=[]
for index,classlabel in enumerate(classes):
    photo_dir="./"+classlabel
    files=glob.glob(photo_dir+"/*.jpg")
    for i,file in enumerate(files):
        image=Image.open(file)
        image=image.convert("RGB")
        image=image.resize((image_resize,image_resize))
        data=np.asarray(image)
        X.append(data)
        Y.append(index)

X=np.array(X)
Y=np.array(Y)
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,Y)
xy=(X_train,X_test,y_train,y_test)
np.save("./izone.npy",xy)
