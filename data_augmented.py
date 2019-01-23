from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes=["nako","sakura","hitomi"]
num_classes = len(classes)
image_size = 50
num_testdata = 50

#画像の読み込み
X_train = []
X_test = []
Y_train = []
Y_test = []

#反転と5度ずつ回転させて画像の水増しをする
for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)

        if i < num_testdata:
            X_test.append(data)
            Y_test.append(index)
        else:
            for angle in range(-20, 20, 5):
                #回転
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                #反転
                img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)



X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

xy = (X_train, X_test, Y_train, Y_test)
np.save("./izone_aug.npy", xy)




