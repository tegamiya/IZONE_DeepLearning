import keras
from keras import layers,models
from keras import optimizers
from keras.utils import np_utils
import numpy as np


classes=["nako","sakura","hitomi"]
num_classes=len(classes)
image_size=50

#メイン関数定義
def main():
    # ファイルからデータを配列に読み込む
    X_train,X_test,y_train,y_test=np.load("./izone_aug.npy")
    # 正規化　最大値で割って0-1に収束
    X_train=X_train.astype("float")/255
    X_test=X_test.astype("float")/255
    # one-hot-Vector :　正解１　他は０
    y_train=np_utils.to_categorical(y_train,num_classes)
    y_test=np_utils.to_categorical(y_test,num_classes)
    
    model=model_train(X_train,y_train)
    # モデルの評価
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
    #コンパイル 最適化手法の宣言 lr=ラーニングレート　loss:損失関数 正解と推定値の誤差
    model.compile(loss="binary_crossentropy",optimizer=optimizers.RMSprop(lr=1e-4),metrics=["acc"])
    #モデルの学習
    model.fit(X,y,batch_size=20,epochs=100)
    #モデルの保存
    model.save("./izone_cnn_aug.h5")
    return model

# XYにはX_test,Y_testが入っている
def model_eval(model,X,y):
    scores=model.evaluate(X,y,verbose=1)
    print("Test acc",scores[1])

#直接このプログラムが呼ばれていたらmain()を実行
if __name__=="__main__":
    main()
