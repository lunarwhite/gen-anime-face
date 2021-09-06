# anime-face-generation

![anime-face-generation](https://socialify.git.ci/lunarwhite/anime-face-generation/image?description=1&descriptionEditable=GAN-based%20generation%20of%20animated%2FACGN%20faces.&font=Raleway&forks=1&issues=1&language=1&logo=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2Fa%2Fae%2FKeras_logo.svg%2F180px-Keras_logo.svg.png&owner=1&pattern=Charlie%20Brown&pulls=1&stargazers=1&theme=Light)

GAN-based generation of animated/ACGN faces. || 基于李宏毅老师的二次元人脸/头像数据集，应用训练GAN

```
.
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
├── res
│   └── image # 数据集
└── tmp # 保存model图片

3 directories, 3 files
```

## 1 概览

- Image Generation
- 基于李宏毅老师的二次元人脸/头像数据集，实现二次元人脸生成
- 数据集来源：[moe](https://make.girls.moe/#/)
- 数据集下载：[kaggle](https://www.kaggle.com/lunarwhite/anime-face-dataset-ntumlds) || [google-drive](https://drive.google.com/file/d/1tpW7ZVNosXsIAWu8-f5EpwtF3ls3pb79/view)
- 选择GAN作为模型，借助Keras搭建训练
- 主要工具包版本为TensorFlow 2.2.0、和Python 3.7.10

## 2 部署

- 克隆repo：`git clone https://github.com/lunarwhite/anime-face-generation.git`
- 更新pip：`pip3 install --upgrade pip`
- 为项目创建虚拟环境：`conda create --name <env_name> python=3.7`
- 激活env：`conda activate <env_name>`
- 安装python库依赖：`pip3 install -r requirements.txt`

## 3 运行

- 运行：`python main.py`

- 调参：在`main.py`文件中修改常用参数，如下
  ```python
  def main():
      gan = GAN(
          optimizer = Adam(0.0002, 0.5), # 优化方法，初始学习率
          loss="binary_crossentropy" # 损失函数
      )
      gan.train(
          epochs=300, # 训练轮数
          batch_size=32, # 一次训练所抓取的数据样本数量
          sample_interval=50 # 多少个epoch之后save一次实验结果
      )
  ```

## 4 流程

- 观察数据
  - 数据集大小
  - 数据集样本
  - 图像分辨率
- 数据预处理
  - 归一化
- 搭建模型

  - Update discriminator//do not update the parameters of generator
    - real_images = sample_batch_data(training_data, batch_size)
    - noise = sample_batch_noise(batch_size, noise_dim)
    - fake_images = generator(noise)
    - real_predicts = discriminator(real_images)
    - fake_predicts = discriminator(fake_images)
    - d_loss = loss_d_fn(real_predicts, real_labels, fake_predicts, fake_labels)
    - d_grad = gradients(d_loss, d_params)
    - d_params = updates(d_params, d_grad)
  - Update Generator //do not update the parameters of discriminator
    - noise = sample_batch_noise(noise_dim, batch_size)
    - fake_images = generator(noise)
    - fake_predicts = discriminator(fake_images)
    - g_loss = loss_g_fn( fake_predicts, real_labels )
    - g_grad = gradients(g_loss, g_params)
    - g_params = updates(g_params, g_grad)
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

## 5 参考

- [How to Train a GAN? Tips and tricks to make GANs work](https://github.com/soumith/ganhacks)
