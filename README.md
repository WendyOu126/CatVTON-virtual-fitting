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
├── eval_metric.py           # 我们自己的评估代码
├── image_conposite.py       # 制作mask代码
├── pad.py                   # 制作mask代码
├── train_new.py             # 训练基于掩码模型代码
├── train_new_virtual.py     # 训练无需掩码模型代码
├── virtual_tryon_mask.py    # 基于掩码模型测试输出
├── virtual_tryon_nomask.py  # 无需掩码模型测试输出
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
我们从各大电商网站收集了春、夏、秋、冬四个季节各1000对数据，共计4000对数据。
并将收集的数据进行了处理，先分类后制作mask。
上传到百度网盘上可以自行使用。

### 2. mask masking

通过我们的pad.py可以生成0-1掩码图，之后再通过image_conposite.py可以合成模特图和0-1掩码图得到我们需要的输入掩码。

### 2. training

我们的训练分为两部分：基于掩码模型的训练和无需掩码的模型训练。
需要从https://huggingface.co/stabilityai/sd-vae-ft-mse中下载sd-vae-ft-mse模型权重。下载目录如下：
```
main/
├── config.json        
├── diffusion_pytorch_model.bin              
├── diffusion_pytorch_model.safetensors               
├── gitattributes               
├── README.md             
```
从https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/tree/main中下载stable-diffusion-inpainting模型权重。下载目录如下：
```
main/
├── feature_extractor/     
├── safety_checker/
├── scheduler/
├── text_encoder/
├── tokenizer/
├── unet
├── vae
├── config.json
├── gitattributes
├── gitattributes%20copy
├── model_index.json
├── README.md
```
先训练基于掩码的模型，代码参考train_new.py，训练输出新的模型权重（重点更新UNet部分），通过新的模型权重可以评估指标，通过我们的评估代码eval_metric.py。

之后用训练好的模型生成“模特-衣服-换衣图”三元数据组，再次进行训练，通过我们的无需掩码的训练代码train_new_virtual.py，代码中重点修改了输入通道和输入信息。我们也提供了自己生成的三元组数据集，已上传到百度网盘。

### 3. test

通过我们的virtual_tryon_mask.py可以进行输入衣服图、模特图和掩码图，输出换衣后的图片。
通过我们的virtual_tryon_nomask.py可以进行输入衣服图、模特图，不输入掩码图，成功输出换衣后的图片。

