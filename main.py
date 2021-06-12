from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

class GAN:
    def __init__(self):
        # 图像尺寸
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # 优化器，使用Adam算法优化
        optimizer = Adam(0.0002, 0.5)

        # 生成器
        self.generator = self.build_generator()

        # 判别器
        self.discriminator = self.build_discriminator()
        # 判别器编译
        self.discriminator.compile(loss="binary_crossentropy",
                                   optimizer=optimizer,
                                   metrics=["accuracy"])

        # z是噪声，送入生成器生成图像
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # 固定判别器，训练生成器（这里不太清楚）
        self.discriminator.trainable = False

        # 将生成器生成的图像传入判别器判断
        validity = self.discriminator(img)

        # 结合生成器和判别器（这里不太清楚）
        self.combined = Model(z, validity)
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

    def build_generator(self):
        # 序列模型
        model = Sequential()

        model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((16, 16, 128)))
        # 上采样，将图像放大一倍
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(3, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=4, input_shape=(64, 64, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        # model.add(ZeroPadding2D())
        model.add(Activation("relu"))
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(256, kernel_size=4, padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # 读取图片数据
        x_train = load_data()
        # 归一化
        x_train = x_train / 127.5 - 1.

        # 图像数据的判别值为1
        valid = np.ones((batch_size, 1))
        # 生成图像的判别值为0
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # 随机选取一个batch的图像送入训练
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            images = x_train[idx]

            # 产生噪声，生成图像
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_images = self.generator.predict(noise)

            # 训练判别器
            d_loss_real = self.discriminator.train_on_batch(images, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_images, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # 保存图像
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # 保存生成结果
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


# 读取extra_data中的图片，并保存
def load_data():
    x_train = np.empty((36740, 64, 64, 3), dtype="float32")
    images = os.listdir('./res/image/')

    lens = len(images)
    for i in range(lens):
        img = Image.open('./res/image/' + images[i])
        arr = np.asarray(img, dtype="float32")
        x_train[i, :, :, :] = arr

    return x_train

def main():
    gan = GAN()
    gan.train(epochs=300, batch_size=32, sample_interval=20)

if __name__ == "__main__":
    main()