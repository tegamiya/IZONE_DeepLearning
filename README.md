# IZONE_DeepLearning

## 概要
IZ*ONEという日韓合同アイドルグループの日本人3名の顔認識をしようとして作られたもの  
https://www.universal-music.co.jp/izone/  
認識できるのは、宮脇咲良、矢吹奈子、本田仁美

## 各ファイル概要

img_collector.py　→　画像スクレイピング  
detection_save.py　→　画像フォルダから顔だけ切り出して保存  
split.py　→　画像データを読み込んでテスト用と検証用配列データにする  
data_augmented.py　→　画像データを反転＆回転させてデータ水増し  
learn_aug.py　→　DeepLearningで学習  
predict.py　→　データを元にWEBアプリ化　ネット上で画像判別できるようにする  
index.html → predict.pyから呼ばれて表示されるhtml

## 参考にした資料
https://www.udemy.com/tensorflow-advanced/learn/v4/overview


## 結果
<img src="https://user-images.githubusercontent.com/40752235/51887079-cca6cb00-23d5-11e9-8c54-8482ead1230b.png">
<img src="https://user-images.githubusercontent.com/40752235/51887082-d0d2e880-23d5-11e9-97ad-0b4b75203cbd.png">

### うまくいかないこともあります
<img src="https://user-images.githubusercontent.com/40752235/51887083-d3cdd900-23d5-11e9-9ad7-36f31b53e7de.png">
