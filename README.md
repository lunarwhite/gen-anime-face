# gen-anime-face

![GitHub Repo stars](https://img.shields.io/github/stars/lunarwhite/gen-anime-face?color=orange)
![GitHub watchers](https://img.shields.io/github/watchers/lunarwhite/gen-anime-face?color=yellow)
![GitHub forks](https://img.shields.io/github/forks/lunarwhite/gen-anime-face?color=green)
![GitHub top language](https://img.shields.io/github/languages/top/lunarwhite/gen-anime-face)
![GitHub License](https://img.shields.io/github/license/lunarwhite/gen-anime-face?color=white)

GAN-based image generation of animated/ACGN faces. 基于李宏毅老师的二次元人脸/头像数据集，应用训练GAN，实现二次元人脸生成

```
.
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
├── res
│   └── image # 数据集
└── tmp # 保存 model 图片
```

## 1 Overview

- 数据初始来源：[MakeGirlsMoe](https://make.girls.moe/#/)
- 数据集下载：[Kaggle](https://www.kaggle.com/lunarwhite/anime-face-dataset-ntumlds)
- 选择 GAN 搭建模型，借助 Keras 搭建训练
- 主要工具包版本为 TensorFlow `2.2.0`、和 Python `3.7.10`

## 2 Setup

- clone repo：`git clone https://github.com/lunarwhite/gen-anime-face.git`
- 更新 pip：`pip3 install --upgrade pip`
- 为项目创建虚拟环境：`conda create --name <env_name> python=3.7`
- 激活 env：`conda activate <env_name>`
- 安装 Python 库依赖：`pip3 install -r requirements.txt`

## 3 Train

- 运行：`python main.py`
- 调参：在 `main.py` 文件中修改常用参数
  ```python
  def main():
      gan = GAN(
          optimizer = Adam(0.0002, 0.5), # 优化方法，初始学习率
          loss="binary_crossentropy" # 损失函数
      )
      gan.train(
          epochs=300, # 训练轮数
          batch_size=32, # 一次训练所抓取的数据样本数量
          sample_interval=50 # 多少个 epoch之 后 save 一次实验结果
      )
  ```

## 4 Workflow

- 观察数据
  - 数据集大小
  - 数据集样本
  - 图像分辨率
- 数据预处理
  - 归一化
- 搭建模型
  - Update discriminator # do not update the parameters of generator
    - real_images = `sample_batch_data(training_data, batch_size)`
    - noise = `sample_batch_noise(batch_size, noise_dim)`
    - fake_images = `generator(noise)`
    - real_predicts = `discriminator(real_images)`
    - fake_predicts = `discriminator(fake_images)`
    - d_loss = `loss_d_fn(real_predicts, real_labels, fake_predicts, fake_labels)`
    - d_grad = `gradients(d_loss, d_params)`
    - d_params = `updates(d_params, d_grad)`
  - Update Generator # do not update the parameters of discriminator
    - noise = `sample_batch_noise(noise_dim, batch_size)`
    - fake_images = `generator(noise)`
    - fake_predicts = `discriminator(fake_images)`
    - g_loss = `loss_g_fn( fake_predicts, real_labels)`
    - g_grad = `gradients(g_loss, g_params)`
    - g_params = `updates(g_params, g_grad)`
- 可视化分析
  - 模型可视化
- 改进模型
  - 1: Normalize the inputs
  - 2: A modified loss function
  - 4: BatchNorm
  - 5: Avoid Sparse Gradients: ReLU, MaxPool
  - 6: Use Soft and Noisy Labels
  - 9: Use the ADAM Optimizer
  - 10: Track failures early
  - 13: Add noise to inputs, decay over time
  - 17: Use Dropouts in G in both train and test phase

## 5 Reference

- [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
