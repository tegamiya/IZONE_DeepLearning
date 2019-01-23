# IZONE_DeepLearning

## 概要
IZ*ONEというグループの日本人3名の顔認識をしようとして作られたもの  
https://www.universal-music.co.jp/izone/  
認識できるのは、宮脇咲良、矢吹奈子、本田仁美

## 各ファイル概要

detection_save.py　→　画像フォルダから顔だけ切り出して保存  
split.py　→　画像データを読み込んでテスト用と検証用配列データにする  
data_augmented.py　→　画像データを反転＆回転させてデータ水増し  
learn_aug.py　→　DeepLearningで学習  
predict.py　→　データを元にWEBアプリ化　ネット上で画像判別できるようにする  

