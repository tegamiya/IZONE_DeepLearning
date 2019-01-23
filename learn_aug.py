import keras
from keras import layers,models
from keras import optimizers
from keras.utils import np_utils
import numpy as np


classes=["nako","sakura","hitomi"]
num_classes=len(classes)
image_size=50
def main():
    X_train,X_test,y_train,y_test=np.load("./izone_aug.npy")
    X_train=X_train.astype("float")/255
    X_test=X_test.astype("float")/255
    y_train=np_utils.to_categorical(y_train,num_classes)
    y_test=np_utils.to_categorical(y_test,num_classes)

    model=model_train(X_train,y_train)
    model_eval(model,X_test,y_test)

def model_train(X,y):
    #モデルの作成
    model=models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(50,50,3)))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(128,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation="relu"))
    model.add(layers.Dense(3,activation="softmax"))
    #コンパイル
    model.compile(loss="binary_crossentropy",optimizer=optimizers.RMSprop(lr=1e-4),metrics=["acc"])
    #モデルの学習
    model.fit(X,y,batch_size=20,epochs=100)
    #モデルの保存
    model.save("./izone_cnn_aug.h5")
    return model

def model_eval(model,X,y):
    scores=model.evaluate(X,y,verbose=1)
    print("Test acc",scores[1])

if __name__=="__main__":
    main()
