##基础数据：
#recall
python recall.py recall.csv

#特征
python feature.py recall.csv feature.csv

## 训练数据、测试数据生成
python train_data.py feature.csv train.csv test.csv

# train
python train.py train.csv model_name

# pred
python pred.py test.csv out_put.csv