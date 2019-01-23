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
    files=glob.glob(photo_dir+"/*.jpg") #globでディレクトリの中の一致したものをfilesに入れる
    for i,file in enumerate(files):
        image=Image.open(file)
        image=image.convert("RGB")#3色に変換
        image=image.resize((image_resize,image_resize))
        data=np.asarray(image)
        X.append(data)
        Y.append(index)#Yはラベル

#TensorFlowが扱いやすいデータ型にする
X=np.array(X)
Y=np.array(Y)

#トレーニングとテストに分割
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,Y)
xy=(X_train,X_test,y_train,y_test)

#データ保存
np.save("./izone.npy",xy)
