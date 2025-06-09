# CatVTON-Virtual-Fitting

## Group Members
group leader:  欧阳雨琪(Github:@WendyOu126)  
group members:  盛夏(@summerok03)  、尹雅萱(@ddoulibra2y)  、施承熙(@AlphaXnevermind)  、黄俊彬(@mgmsk0923)

## structure
```
main/
├── app.py                   # 主应用脚本
├── app_flux.py              # CatVTON-FLUX 应用脚本
├── app_p2p.py               # Person-to-Person 应用脚本
├── densepose/               # DensePose 模块
├── detectron2/              # Detectron2 模块
├── eval.py                  # 模型评估脚本
├── inference.py             # 推理脚本
├── index.html               # Web 界面
├── model/                   # 模型定义
├── preprocess_agnostic_mask.py # 预处理脚本
├── resource/                # 资源文件
├── utils.py                 # 工具函数
├── requirements.txt         # 依赖项
├── LICENSE                  # 许可证
└── README.md                # 项目说明文档
```
## Updates 
### 2025.5.22 
我们从各种电商网站收集已经收集了春夏部分的各800对服装-模特数据。数据保存在百度网盘中：
```shell
通过网盘分享的文件：CatCTON-dataset
链接: https://pan.baidu.com/s/1oI1H0N2W8B70kAISj5I8_Q?pwd=j6jf 提取码: j6jf 
```
### 2025.6.9
我们对收集的数据进行了分类和生成匹配对操作，方便后续模型输入，并进行了模型需要的有掩码制作。
## Installation

Create a conda environment & Install requirments
```shell
conda create -n catvton python==3.9.0
conda activate catvton
cd CatVTON-main  # or your path to CatVTON project dir
pip install -r requirements.txt
```

## model
We reproduce the mask-based CatVTON model, and calculate trained and the indicators are calculated.

## Inference
### 1. Data Preparation
We will collect about 1000 pairs of clothing and model data for spring, summer, autumn and winter from e-commerce websites.

### 2. training

### 3. test

### 4. log

## Citation

