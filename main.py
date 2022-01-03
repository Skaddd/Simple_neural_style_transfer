import numpy as np
from numpy.lib.function_base import extract
import pandas as pd
import tensorflow as tf

import os
import utils
from utils import load_image, gram_matrix, save_display_img
import matplotlib.pyplot as plt

import argparse



#content ca peut etre tout, donc rajputer une selection


content_features_index = ['block5_conv2']

style_features_index = ['block1_conv1', 'block2_conv1','block3_conv1','block4_conv1','block5_conv1']

num_content_layers = len(content_features_index)

num_style_layers = len(style_features_index)


def vgg_layers(layer_names):

    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights = 'imagenet')
    vgg.trainable = False 

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)

    return model



class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()

        self.vgg = vgg_layers(style_layers+ content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)

        self.vgg.trainable = False

    def call(self,inputs):

        inputs = inputs*255.

        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = (outputs[:self.num_style_layers],outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]


        content_dict = {content_name : value 
                        for content_name, value 
                        in zip(self.content_layers,content_outputs)}

        style_dict = {content_name : value 
                        for content_name, value 
                        in zip(self.style_layers,style_outputs)}

        return {'content' : content_dict, 'style' : style_dict}


def neural_style_transfer(config):

    content_img_path = os.path.join(config['content_image_dir'],config['content_image_name'])
    style_img_path = os.path.join(config['style_image_dir'],config['style_image_name'])

    dump_path = config['output_img_dir']

    os.makedirs(dump_path, exist_ok = True)


    content_img = utils.prepare_image(content_img_path, config['height'])
    style_img = utils.prepare_image(style_img_path, config['height'])

    print(content_img.shape)


    if config['init_method'] =='content':

        print('fine')

        optimizing_img = tf.Variable(content_img)

    else :

        style_img_resized = utils.prepare_image(style_img_path, np.asarray(content_img.shape[2:]))

        optimizing_img = tf.Variable(style_img_resized)

    #On opti une image pas le réseau
    extractor = StyleContentModel(style_features_index,content_features_index)

    content_targets =extractor(content_img)['content']
    style_targets = extractor(style_img)['style']

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.02, beta_1=0.99, epsilon=1e-1)


    iterations = 100

    for k in range(iterations):
        print(k)
        training_loop(optimizing_img,optimizer,extractor,content_targets,style_targets)

    return optimizing_img


@tf.function
def training_loop(image, optimizer, extractor,content_targets,style_targets):
    with tf.GradientTape() as tape:

        outputs = extractor(image)
        total_loss = compute_loss(image, outputs['style'], outputs['content'], content_targets,style_targets)

    grad = tape.gradient(total_loss, image)
    optimizer.apply_gradients([(grad,image)])
    image.assign(utils.clipping_pixel(image))







def compute_loss(optimizing_img,style_outputs,content_outputs,content_targets,style_targets):


    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] -content_targets[name])**2) for name in content_outputs.keys()])

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])

    style_loss/= num_style_layers


    tv_loss = utils.total_variation(optimizing_img)



    total_loss = style_loss*config['style_weight'] +content_loss*config['content_weight'] + tv_loss * config['tv_weight']

    return total_loss






if __name__=="__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument("--content_image_name", type=str, default = 'YellowLabradorLooking_new.jpg')
    parser.add_argument("--style_image_name", type = str, default = 'kandinsky5.jpg')
    parser.add_argument("--height", type=int, default = (512,512))


    parser.add_argument("--content_weight", type = int, default = 1e5)
    parser.add_argument("--style_weight", type =int, default = 3e4)
    parser.add_argument("--tv_weight", type =int, default = 1.0)

    parser.add_argument("--init_method", type = str, default = 'content')

    args = parser.parse_args()




    default_resource_dir = os.path.join(os.path.dirname(__file__),'data')
    content_image_dir = os.path.join(default_resource_dir,'content-images')
    style_image_dir = os.path.join(default_resource_dir,'style-images')
    output_img_dir = os.path.join(os.path.join(default_resource_dir,'output-images'))


    config = dict()

    for arg in vars(args):
        config[arg]= getattr(args,arg)


    config['content_image_dir'] = content_image_dir
    config['style_image_dir'] = style_image_dir
    config['output_img_dir'] = output_img_dir



    img = neural_style_transfer(config)


    output_file = os.path.join(output_img_dir,'result.jpg')

    utils.save_display_img(img,output_file,should_display=True)





    

