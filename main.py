import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import plot_model

def main(): # 调参
    gan = GAN(
        optimizer = Adam(0.0002, 0.5), # 优化方法
        loss="binary_crossentropy" # 损失函数
    )
    gan.train(
        epochs=300, # 训练轮数
        batch_size=32, # 一次训练所抓取的数据样本数量
        sample_interval=50 # 多少个epoch之后save一次实验结果
    )

# 观察图像数据
def preprocess_data():
    res_path = 'res/images/'
    images = os.listdir(res_path)
    
    plt.figure(figsize=(15, 8))
    plt.title('First 120 image in training')

    row, col = 15, 8
    for i in range(row*col):
        plt.subplot(col, row, i+1)
        img = Image.open(res_path + images[i])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.show()

    for i in range(len(images)):
        myimg = Image.open(res_path + images[i])
        if(myimg.width!=64 or myimg.height!=64): 
            print("以下图片不是64*64：")
            print(myimg.size)

# 加载图像数据
def load_data(): 
    res_path = 'res/images/'
    images = os.listdir(res_path)
    x_train = np.empty((36740, 64, 64, 3), dtype="float32")

    for i in range(len(images)):
        img = Image.open(res_path + images[i])
        arr = np.asarray(img, dtype="float32")
        x_train[i, :, :, :] = arr
    return x_train

# 定义GAN类
class GAN:
    # 初始化参数
    def __init__(self, optimizer, loss): 
        self.img_rows = 64
        self.img_cols = 64 
        self.channels = 3 # 图像层数
        self.latent_dim = 100
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        
        z = Input(shape=(self.latent_dim,)) # 噪声
        img = self.generator(z) # 将噪声送入生成器生成图像

        self.discriminator.trainable = False # 固定判别器，训练生成器
        validity = self.discriminator(img) # 将生成器生成的图像传入判别器判断

        self.combined = Model(z, validity) # 结合生成器和判别器
        self.combined.compile(loss=loss, optimizer=optimizer)

    # 建立生成器generator
    def build_generator(self):
        model = Sequential()

        model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(3, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        plot_model(model, show_shapes=True, show_layer_names=True, to_file='tmp/model_g.png')
        return Model(noise, img)

    # 建立判别器discriminator
    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=4, input_shape=(64, 64, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(ZeroPadding2D())
        model.add(Activation("relu"))
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(256, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        img = Input(shape=self.img_shape)
        validity = model(img)

        plot_model(model, show_shapes=True, show_layer_names=True, to_file='tmp/model_d.png')
        return Model(img, validity)

    # 模型训练
    def train(self, epochs, batch_size, sample_interval):
        x_train = load_data()
        x_train = x_train / 127.5 - 1. # 数据预处理-归一化

        valid = np.ones((batch_size, 1)) # 图像数据的判为1
        fake = np.zeros((batch_size, 1)) # 生成图像的判为0

        for epoch in range(epochs):
            idx = np.random.randint(0, x_train.shape[0], batch_size)  # 随机选取一个batch的图像送入训练
            images = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim)) # 产生噪声
            gen_images = self.generator.predict(noise) # 生成图像

            d_loss_real = self.discriminator.train_on_batch(images, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if (epoch % sample_interval == 0 or epoch==epochs): # 保存图像
                self.sample_images(epoch)

    # 保存生成结果
    def sample_images(self, epoch):
        row, col = 5, 5
        noise = np.random.normal(0, 1, (row * col, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(row, col)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("tmp/output_gen_%d.jpg" % epoch)
        plt.close()

if __name__ == "__main__":
    main()
