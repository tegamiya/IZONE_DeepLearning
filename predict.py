import os
import io
import time
import numpy as np
import cv2
from flask import Flask,render_template,request,redirect,url_for,send_from_directory,session
from werkzeug import secure_filename
from keras.models import Sequential,load_model
import keras,sys
from PIL import Image

#Flaskでアプリ化

classes=["nako","sakura","hitomi"]
num_classes=len(classes)
image_resize=50

UPLOAD_FOLDER="./uploads"
ALLOWED_EXTENSIONS=set(["png","jpg","gif"])

app=Flask(__name__)
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER
IMAGE_WIDTH=640


# ---------
import tensorflow as tf
from keras.models import load_model

model = load_model('./izone_cnn_aug.h5')
graph = tf.get_default_graph()


#ファイルのアップロード可否判定
#許可したファイルの形式なのかを判定する
def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/send",methods=["GET","POST"])
def send():
    if request.method=="POST":
        img_file=request.files["img_file"]
        if img_file and allowed_file(img_file.filename):
            filename=secure_filename(img_file.filename)
        else:
            return '''<h2>許可されていない拡張子です</h2>'''
        img_file.save(os.path.join(app.config["UPLOAD_FOLDER"],filename))
        #パスの結合
        filepath=os.path.join(app.config["UPLOAD_FOLDER"],filename)
        classes=["nako","sakura","hitomi"]
        img=cv2.imread(filepath)
        image=img
        HAAR_FILE="./haarcascade_frontalface_default.xml"#カスケードファイル
        cascade=cv2.CascadeClassifier(HAAR_FILE)
        face=cascade.detectMultiScale(image)#顔認識
        size=(50,50)
        if 0<len(face):
            for x,y,w,h in face:
                image=image[y:y+h,x:x+w]
                image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image=cv2.resize(image,size)
                data=np.asarray(image)

                global graph
                with graph.as_default():

                    X=[]
                    X.append(data)
                    X=np.array(X)
                    result=model.predict([X])[0]
                    predicted=result.argmax()
                    percentage=int(result[0]*100)
                    percentage_1=int(result[1]*100)
                    percentage_2=int(result[2]*100)
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)#顔を四角で囲む
                    cv2.putText(img,classes[predicted],(x+30,y+20+h),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0),2)#文字を入力
                    add_img_url=os.path.join(app.config["UPLOAD_FOLDER"],"add_"+filename).replace('/', os.sep) #replaceは\と/の混同防止
                    #リサイズ
                    img=cv2.resize(img,(IMAGE_WIDTH,int(IMAGE_WIDTH*img.shape[0]/img.shape[1])))
                    cv2.imwrite(add_img_url,img)
                    return render_template("index.html",add_img_url=add_img_url,percentage=percentage,percentage_1=percentage_1,percentage_2=percentage_2)
        else:
            return '''<h2>顔を検出できませんでした</h2>'''
    return render_template("index.html")


#/uploads/<filename>でディレクトリ内の静的ファイルにアクセスできるようにする
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
