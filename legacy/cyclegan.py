from datetime import datetime

import tensorflow as tf
from matplotlib import pyplot as plt
# import tensorflow_datasets as tfds
# from tqdm import tqdm
import gc
import numpy as np
from random import random
import time

devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)
# tf.get_logger().setLevel('ERROR')
tf.config.run_functions_eagerly(False)


LAMBDA_CYCLE = 1
LAMBDA_IDENTITY = 5#5
LAMBDA_GEN = 10#5

LAMBDA_DISCRIMINATOR = 1

OUT_CHANNELS = 1

# BUFFER_SIZE = 50
NUM_TOKENS = 128
EMB_SIZE = 64
L_RATE = 1e-4
EPOCHS = 1

BATCH_SIZE = 25#50
IMG_WIDTH = NUM_TOKENS
IMG_HEIGHT = EMB_SIZE

#--------------------------------------------


class Encoder(tf.keras.layers.Layer):
    def __init__(self, units=32, num_heads=2):
        super(Encoder, self).__init__()

        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=2)
        self.add1 = tf.keras.layers.Add()
        self.norm1 = tf.keras.layers.LayerNormalization(axis=1)
        self.ff = tf.keras.layers.Dense(units, activation="relu")
        # self.ff1 = tf.keras.layers.Dense(units, activation="relu")
        # self.ff2 = tf.keras.layers.Dense(units, activation="relu")
        # self.ff3 = tf.keras.layers.Dense(units, activation="relu")
        self.add2 = tf.keras.layers.Add()
        self.norm2 = tf.keras.layers.LayerNormalization(axis=1)

    def call(self, inputs):
        att_1 = self.att(inputs, inputs)
        add1_1 = self.add1([inputs, att_1])
        norm1_1 = self.norm1(add1_1)
        ff_1 = self.ff(norm1_1)
        # ff1_1 = self.ff1(ff_1)
        # ff2_1 = self.ff2(ff1_1)
        # ff3_1 = self.ff3(ff2_1)
        # add2_1 = self.add2([ff3_1, norm1_1])
        add2_1 = self.add2([ff_1, norm1_1])
        return self.norm2(add2_1)
        # return self.norm2(add2_1)


def downsample(filters, size, batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    layers = [
        tf.keras.layers.Conv2D(filters, size, strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False),
        tf.keras.layers.BatchNormalization() if batchnorm else None,
        tf.keras.layers.LeakyReLU()
    ]
    layers = list(filter(lambda x: x is not None, layers))
    result = tf.keras.Sequential(layers)
    return result


def upsample(filters, size, dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    layers = [
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5) if dropout else None,
        tf.keras.layers.ReLU()
    ]
    layers = list(filter(lambda x: x is not None, layers))
    result = tf.keras.Sequential(layers)

    return result


def make_generator(width, heigth, num_filters=64, filter_size=4):
    # (bs, 128, 64, 1)
    inputs = tf.keras.layers.Input(shape=[width, heigth])

    # out1 = Encoder(units=heigth, num_heads=filter_size)(inputs)
    # out2 = Encoder(units=heigth, num_heads=filter_size)(out1)
    # out3 = Encoder(units=heigth, num_heads=filter_size)(out2)
    # out4 = Encoder(units=heigth, num_heads=filter_size)(out3)
    # out5 = Encoder(units=heigth, num_heads=filter_size)(out4)
    # out6 = Encoder(units=heigth, num_heads=filter_size)(out5)

    # model = tf.keras.Model(inputs=inputs, outputs=out3)
    # model.compile()
    # return model
    inputs = tf.keras.layers.Input(shape=[width, heigth])
    inp_shaped = tf.keras.layers.Reshape([width, heigth, 1])(inputs)

    down_stack = [
        # (bs, 64, 32, 64)
        downsample(num_filters, filter_size, batchnorm=False),  # (bs, 128, 128, 64)
        # (bs, 32, 16, 128)
        downsample(num_filters * 2, filter_size),  # (bs, 64, 64, 128)
        # (bs, 16, 4, 256)
        downsample(num_filters * 4, filter_size),  # (bs, 32, 32, 256)
        downsample(num_filters * 8, filter_size),  # (bs, 16, 16, 512)
        downsample(num_filters * 8, filter_size),  # (bs, 8, 8, 512)
        downsample(num_filters * 8, filter_size),  # (bs, 4, 4, 512)
#         downsample(num_filters * 8, filter_size),  # (bs, 2, 2, 512)
#         downsample(num_filters * 8, filter_size),  # (bs, 1, 1, 512)
    ]

    up_stack = [
#         upsample(num_filters * 8, filter_size, dropout=True),  # (bs, 2, 2, 1024)
#         upsample(num_filters * 8, filter_size, dropout=True),  # (bs, 4, 4, 1024)
        upsample(num_filters * 8, filter_size, dropout=True),  # (bs, 8, 8, 1024)
        upsample(num_filters * 8, filter_size),  # (bs, 16, 16, 1024)
        upsample(num_filters * 4, filter_size),  # (bs, 32, 32, 512)
        upsample(num_filters * 2, filter_size),  # (bs, 64, 64, 256)
        upsample(num_filters, filter_size),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (bs, 256, 256, 3)

    x = inp_shaped

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)
    x_shaped = tf.keras.layers.Reshape([width, heigth])(x)

    att_x_shaped = Encoder(units=heigth, num_heads=filter_size)(x_shaped)
    # out2 = Encoder(units=heigth, num_heads=filter_size)(out1)
    # att_x_shaped = stack_transformer_layer(x_shaped,nffn_units=heigth)

    model = tf.keras.Model(inputs=inputs, outputs=att_x_shaped)
    model.compile()
    #     print(model.summary())
    #     tf.keras.utils.plot_model(model, "generator.png", show_shapes=True)
    return model


# def discriminator(width=NUM_TOKENS, heigth=EMB_SIZE, target=True, num_filters=64, filter_size=4):
#
#     initializer = tf.random_normal_initializer(0., 0.02)
#
#     inp = tf.keras.layers.Input(shape=[width, heigth], name='input_image')
#
#     out1 = Encoder(units=heigth, num_heads=filter_size)(inp)
#     out2 = Encoder(units=heigth, num_heads=filter_size)(out1)  # tf.keras.layers.Dense(1, activation='relu')(out1)
#
#     out3 = Encoder(units=heigth, num_heads=filter_size)(out2)
#     out4 = Encoder(units=heigth, num_heads=filter_size)(out3)
#     out5 = Encoder(units=heigth, num_heads=filter_size)(out4)
#
#     out6 = tf.keras.layers.Dense(heigth, activation='relu')(out5)
#     out7 = tf.keras.layers.LayerNormalization(axis=1)(out6)
#
#     return tf.keras.Model(inputs=inp, outputs=out7)


def discriminator(width=NUM_TOKENS, heigth=EMB_SIZE, target=True, num_filters=64, filter_size=4):
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.
    Returns:
    Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[width, heigth], name='input_image')
    inp_shaped = tf.keras.layers.Reshape([width, heigth, 1])(inp)
    x = inp_shaped

    if target:
        tar = tf.keras.layers.Input(shape=[width, heigth], name='target_image')
        tar_shaped = tf.keras.layers.Reshape([width, heigth, 1])(tar)
        x = tf.keras.layers.concatenate([inp_shaped, tar_shaped])  # (bs, 256, 256, channels*2)

    down1 = downsample(num_filters, filter_size, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(num_filters * 2, filter_size)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(num_filters * 4, filter_size)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
      num_filters * 8, filter_size, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

#     if norm_type.lower() == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(
      1, filter_size, strides=1,
      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    last2= out1 = Encoder(units=heigth, num_heads=filter_size)(last)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last2)
    else:
        return tf.keras.Model(inputs=inp, outputs=last2)


def make_cycle_gan(n_tokens, emb_size, num_filters=64, filter_size=4):
    gen_g = make_generator(n_tokens, emb_size, num_filters, filter_size)
    gen_f = make_generator(n_tokens, emb_size, num_filters, filter_size)

    discr_x = discriminator(NUM_TOKENS, EMB_SIZE, target=False, num_filters=num_filters, filter_size=filter_size)
    discr_y = discriminator(NUM_TOKENS, EMB_SIZE, target=False, num_filters=num_filters, filter_size=filter_size)

    gen_g.compile()
    gen_f.compile()
    discr_x.compile()
    discr_y.compile()

    return gen_g, gen_f, discr_x, discr_y


def discriminator_loss(loss_obj, real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss# * 0.5


def generator_loss(loss_obj, generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return loss1 # LAMBDA *


def identity_loss(loss_obj, real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return loss # LAMBDA * 0.5 *


@tf.function
def train_step(
        real_x, real_y,
        gen_g, gen_f,
        discr_x, discr_y,
        gen_g_opt, gen_f_opt,
        discr_x_opt, discr_y_opt,
        loss_obj, prob=0.1
):
    with tf.GradientTape(persistent=True) as tape:

        fake_y = gen_g(real_x, training=True)
        cycled_x = gen_f(fake_y, training=True)

        fake_x = gen_f(real_y, training=True)
        cycled_y = gen_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = gen_f(real_x, training=True)
        same_y = gen_g(real_y, training=True)

        disc_real_x = discr_x(real_x, training=True)
        disc_real_y = discr_y(real_y, training=True)

        disc_fake_x = discr_x(fake_x, training=True)
        disc_fake_y = discr_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = LAMBDA_GEN * generator_loss(loss_obj, disc_fake_y)
        gen_f_loss = LAMBDA_GEN * generator_loss(loss_obj, disc_fake_x)

        #         print(real_x.shape, cycled_x.shape)
        #         print(real_y.shape, cycled_y.shape)

        total_cycle_loss = LAMBDA_CYCLE * (calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y))

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + LAMBDA_IDENTITY * identity_loss(loss_obj, real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + LAMBDA_IDENTITY * identity_loss(loss_obj, real_x, same_x)

        disc_x_loss = discriminator_loss(loss_obj, disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(loss_obj, disc_real_y, disc_fake_y)


    if random() <= prob:
        discriminator_x_gradients = tape.gradient(disc_x_loss,
                                                  discr_x.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss,
                                                  discr_y.trainable_variables)
        if disc_x_loss > 5e-1:
            discr_x_opt.apply_gradients(zip(discriminator_x_gradients,
                                            discr_x.trainable_variables))
        if disc_y_loss > 5e-1:
            discr_y_opt.apply_gradients(zip(discriminator_y_gradients,
                                            discr_y.trainable_variables))

    generator_g_gradients = tape.gradient(total_gen_g_loss,
                                          gen_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss,
                                          gen_f.trainable_variables)

    # Apply the gradients to the optimizer
    gen_g_opt.apply_gradients(zip(generator_g_gradients,
                                  gen_g.trainable_variables))

    gen_f_opt.apply_gradients(zip(generator_f_gradients,
                                  gen_f.trainable_variables))

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss  # total


def make_optimizers(lrate=2e-4):
    # gen_g_opt = tf.keras.optimizers.Adam(lrate, beta_1=0.5)
    # gen_f_opt = tf.keras.optimizers.Adam(lrate, beta_1=0.5)
    #
    # discr_x_opt = tf.keras.optimizers.Adam(lrate, beta_1=0.5)
    # discr_y_opt = tf.keras.optimizers.Adam(lrate, beta_1=0.5)

    gen_g_opt = tf.keras.optimizers.SGD(lrate)
    gen_f_opt = tf.keras.optimizers.SGD(lrate)

    discr_x_opt = tf.keras.optimizers.SGD(lrate)
    discr_y_opt = tf.keras.optimizers.SGD(lrate)
    return gen_g_opt, gen_f_opt, discr_x_opt, discr_y_opt


def plot_sample(dataset,gen_g, gen_f):
    for i in dataset.batch(1):
        A, B = i[:,0], i[:,1]

        fig, ax = plt.subplots(2, 3,figsize=(30,30))

        ax[0,0].imshow(A[0])
        ax[0,0].set_title("a")

        ax[0,1].imshow(gen_g(A)[0])
        ax[0,1].set_title("G(a)")

        ax[0,2].imshow(gen_f(gen_g(A))[0])
        ax[0,2].set_title("F(G(a))")

        ax[1,0].imshow(B[0])
        ax[1,0].set_title("b")

        ax[1,1].imshow(gen_f(B)[0])
        ax[1,1].set_title("F(b)")

        ax[1,2].imshow(gen_g(gen_f(B))[0])
        ax[1,2].set_title("G(F(b))")

        fig.savefig(r'C:\Users\1\Desktop\img\\'+str(datetime.timestamp(datetime.now()))+'.png')
        break


def fit(
        dataset,
        gen_g, gen_f,
        discr_x, discr_y,
        batch=BATCH_SIZE, n_epoch=EPOCHS, lrate=L_RATE, prob=0.1
):
    gen_g_opt, gen_f_opt, discr_x_opt, discr_y_opt = make_optimizers(lrate)
    loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    portion = 0
    log_step = 0.05
    nsamples = len(dataset)
    count = 0

    for epoch in range(n_epoch):
        count = 0
        print("Epoch", epoch)
        for data in dataset.batch(batch):
            gg_loss, gf_loss, dx_loss, dy_loss = train_step(data[:, 0], data[:, 1],
                                                            gen_g, gen_f,
                                                            discr_x, discr_y,
                                                            gen_g_opt, gen_f_opt,
                                                            discr_x_opt, discr_y_opt,
                                                            loss_obj, prob=prob)
            #             total = total.numpy()
            #             progress_bar.set_description(f"total = {total}")
            gg_loss, gf_loss, dx_loss, dy_loss = gg_loss.numpy(), gf_loss.numpy(), dx_loss.numpy(), dy_loss.numpy()
            count += 1
            percent = (count * batch) / nsamples
            if (percent) >= portion:
                portion += log_step
                print(f"time = {time.ctime()} percent = {percent*100}%\n gg_loss = {gg_loss} gf_loss = {gf_loss} dx_loss = {dx_loss} dy_loss = {dy_loss}")
        plot_sample(dataset, gen_g, gen_f)



# Dataset for CycleGAN
def get_dataset(path, cache):
    spec = tf.TensorSpec(shape=(2, NUM_TOKENS, EMB_SIZE), dtype=tf.float32)
    return tf.data.experimental.load(path, element_spec = spec).cache(cache)


def save_cycle_gan(gen_g, gen_f, discr_x, discr_y, save_dir=r"D:\DIPLOMA\model_gan"):
    gen_g.save(save_dir + r"\gen_g")
    gen_f.save(save_dir + r"\gen_f")
    discr_x.save(save_dir + r"\discr_x")
    discr_y.save(save_dir + r"\discr_y")

