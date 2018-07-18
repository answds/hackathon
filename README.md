# hackathon
hackathon比赛，图片识别代码

## Directory
- ./data/ 训练文件存放目录

- data_helpers.py 数据预处理：包括图片加载, 切分训练集测试集, 做成batch

- vgg_model.py 模型代码：vgg16

- vgg_train.py 训练代码, 可复用，换掉模型即可


## Requirements
- python 3.5
- numpy
- sklearn
- tensorflow


## Quick start 
```bash
git clone https://github.com/answds/hackathon.git
cd hackathon
python vgg_train.py 
```
