import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from PIL import Image
import shutil


def rescale(image, Rescale=True):
    if not Rescale:
        return image
    if Rescale:
        width, height = image.size
        while width > 1280 or height > 1280:
            image = image.resize((int(width // 2), int(height // 2)), Image.BICUBIC)
            width, height = image.size
        while width < 128 or height < 128:
            image = image.resize((int(width * 2), int(height * 2)), Image.BICUBIC)
            width, height = image.size
        return image


def plot_xy(data, label, title, epoch):
    """绘图"""
    print(len(data), data)
    print(len(label), label)
    mapping = {0: "2×", 1: "3×", 2: "4×", 3: "5×", 4: "6×", 5: "7×", 6: "8×"}
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(data)
    df = pd.DataFrame(x_tsne, columns=['x', 'y'])
    df['label'] = [mapping[x] for x in label]

    sns.scatterplot(x="x", y="y", hue=df.label.tolist(), palette=sns.color_palette("hls", 7), data=df)
    plt.title(title, fontsize=15)
    # plt.xticks([])
    # plt.yticks([])
    plt.axis('on')
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('./test{}.pdf'.format(epoch))
    plt.show()


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    pixelart_dir = './examples'
    image_dir = os.path.join(opt.dataroot,'Input')
    testA_dir = os.path.join(opt.dataroot,'testA') #testA,testB should be empty.
    testB_dir = os.path.join(opt.dataroot,'testB')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(testA_dir):
        os.makedirs(testA_dir)
    if not os.path.exists(testB_dir):
        os.makedirs(testB_dir)

    Rescale = True
    cell_size_lst=[int(i) for i in opt.cell_size.split(',')]
    for num,file in enumerate(os.listdir(image_dir)):

        name=file.split('.')[0] if opt.not_rename else num+1 #rename except for video frames.
        
        if file.endswith('png') or file.endswith('jpg'):
            image_path = os.path.join(image_dir, file)
            image = Image.open(image_path).convert('RGB')
            image = rescale(image, Rescale)
            for Cell_Size in cell_size_lst:
                save_path = os.path.join(testA_dir, '{}_{}.png'.format(Cell_Size,num))
                pixelart_path = os.path.join(pixelart_dir, '{}_1.png'.format(Cell_Size))
                pixelart_save_path = os.path.join(testB_dir, '{}_{}.png'.format(Cell_Size,num))
                image.save(save_path)
                shutil.copy(pixelart_path, pixelart_save_path)
            num += 1
        else:
            print('The format of input image should be jpg or png.')    


    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    
    webpage.save()  # save the HTML
    # plot_xy(model.TsneData, model.TsneLabel, "T-sne visualization for cell size code", opt.epoch)
    
