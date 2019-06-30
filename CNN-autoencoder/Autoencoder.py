import tensorflow as tf
import os
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPooling2D, Input, UpSampling2D
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from random import randint

print("Tensorflow version", tf.__version__)

PATH = 'Images'
D = 32
PDF = PdfPages('plot.pdf')

### PREPARATION ###

def read_jpg(path):
    img = Image.open(path)
    w, h = img.size
    if (h > w):
        img = img.crop((5, 24, w - 5, h - 24))
    elif (w < h):
        img = img.crop((24, 5, w - 24, h -5))
    else:
        img = img.crop((5,5, w -5 ,h-5))
    img = img.resize((D, D), Image.ANTIALIAS)
    img = np.asarray(img)
    return img

def get_imgs(directory):
    imgs = []
    c = 0
    print("\nLoading images...\n")
    for root, dirs, files in os.walk(directory):
        print("{}/120 - {}".format(c, root))
        c += 1
        for file_name in files:
            img_path = root + os.sep + file_name

            if os.path.exists(img_path):
                img = read_jpg(img_path)
                imgs.append(img)

    return np.array(imgs)

def prepare_data():
    imgs = get_imgs(PATH)
    imgs = imgs.astype('float32') / 255
    imgs = imgs.reshape((len(imgs), D, D, 3))
    return imgs

### PLOTTING ###

def plot_reconstructions(imgs, recs, train):
    N = 6
    f, axarr = plt.subplots(nrows=N, ncols=N, figsize=(8, 6))

    for i in range(N * N // 2):
        axarr[2 * i // N, 2 * i % N].imshow((imgs[i] * 255).astype('uint8'), interpolation='nearest')
        axarr[2 * i // N, 2 * i % N].set_xticks([])
        axarr[2 * i // N, 2 * i % N].set_yticks([])
        axarr[(2 * i + 1) // N, (2 * i + 1) % N].imshow((recs[i] * 255).astype('uint8'), interpolation='nearest')
        axarr[(2 * i + 1) // N, (2 * i + 1) % N].set_xticks([])
        axarr[(2 * i + 1) // N, (2 * i + 1) % N].set_yticks([])

    if train == True:
        text = "Image reconstructions - training data"
    else:
        text = "Image reconstructions - validation data"

    plt.text(0.05, 0.92, text, transform=f.transFigure, size=16)
    plt.close()
    PDF.savefig(f)

def plot_img_loss(train, recs, pic, best):
    N = 6
    f, axarr = plt.subplots(nrows=N, ncols=N, figsize=(6, 8))

    for i in range(N * N // 2):
        axarr[2 * i // N, 2 * i % N].title.set_text("L:{:.4f}".format(pic[i][1]))
        axarr[2 * i // N, 2 * i % N].imshow((train[pic[i][0]] * 255).astype('uint8'), interpolation='nearest')
        axarr[2 * i // N, 2 * i % N].set_xticks([])
        axarr[2 * i // N, 2 * i % N].set_yticks([])
        axarr[(2 * i + 1) // N, (2 * i + 1) % N].imshow((recs[pic[i][0]] * 255).astype('uint8'),
                                                        interpolation='nearest')
        axarr[(2 * i + 1) // N, (2 * i + 1) % N].set_xticks([])
        axarr[(2 * i + 1) // N, (2 * i + 1) % N].set_yticks([])

    # plt.tight_layout()
    if best == False:
        text = 'Images with highest loss out of {}'.format(len(train))
    else:
        text = 'Images with lowest loss out of {}'.format(len(train))

    plt.text(0.05, 0.92, text, transform=f.transFigure, size=16)
    plt.close()
    PDF.savefig(f)

def plot_loss_graph(history_dict):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    PDF.savefig()
    plt.close()

### TRAINING ###

def train_model(train, val):
    conv = 5
    pool = 2
    model = keras.Sequential([
        Conv2D(3, (conv, conv), activation="relu", padding="same"),  # (32,32,3) = (D,D,3)
        MaxPooling2D((pool, pool)),
        Conv2D(8, (conv, conv), activation="relu", padding="same"),  # (16,16,8)
        MaxPooling2D((pool, pool)),
        Conv2D(16, (conv, conv), activation="relu", padding="same"),  # (8,8,16)
        MaxPooling2D((pool, pool)),
        Conv2D(24, (conv, conv), activation="relu", padding="same"),  # (4,4,24)
        Flatten(),
        Reshape(target_shape=(4, 4, 24)),
        UpSampling2D((pool, pool)),
        Conv2D(16, (conv, conv), activation="relu", padding="same"),  # (8,8,16)
        UpSampling2D((pool, pool)),
        Conv2D(8, (conv, conv), activation="relu", padding="same"),  # (16,16,8)
        UpSampling2D((pool, pool)),
        Conv2D(3, (conv, conv), activation="sigmoid", padding="same"),  # (32,32,3)
    ])

    # loss function and optimizer
    model.compile(optimizer="adam", loss='mean_squared_error')
    # train the model
    h = model.fit(train, train, epochs=25, verbose=1, validation_data=(val, val))

    # get predicted images
    predicted_train = model.predict(train)
    predicted_val = model.predict(val)

    # plot a few input - output comparisons
    K = 18  # num of pics
    idx = randint(0, (len(val) - K))
    plot_reconstructions(train[idx:idx + K], predicted_train[idx:idx + K], True)
    plot_reconstructions(val[idx:idx + K], predicted_val[idx:idx + K], False)

    pics = {}
    for i, pic in enumerate(train):
        loss = sum((train[i].reshape((D * D * 3,)) - predicted_train[i].reshape((D * D * 3,))) ** 2)
        pics[i] = loss
    pic_lst = list(sorted(pics.items(), key=lambda x: x[1], reverse=True))
    worst = pic_lst[:18]
    best = pic_lst[-18:]

    plot_img_loss(train, predicted_train, worst, False)
    plot_img_loss(train, predicted_train, best[::-1], True)

    plot_loss_graph(h.history)


if __name__ == "__main__":
    imgs = prepare_data()
    training = imgs[:15000]
    validation = imgs[15000:]

    train_model(training, validation)

    PDF.close()
    subprocess.Popen("start plot.pdf", shell=True)
