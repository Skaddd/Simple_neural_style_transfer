import numpy as np
import pandas as pd
import tensorflow as tf

import os
import utils
from utils import load_image, gram_matrix, save_display_img
import matplotlib.pyplot as plt

import argparse



#content ca peut etre tout, donc rajputer une selection


content_features_index = ['block4_conv2']

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



class StyleModel(tf.keras.models.Model):
    def __init__(self, style_layers):
        super(StyleModel,self).__init__()


        self.vgg = vgg_layers(style_layers)
        self.style_layers = style_layers
        self.num_style_layers = len(self.style_layers)
        self.vgg.trainable = False

    def call(self, inputs):

        inputs = inputs*255.

        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        style_outputs = self.vgg(preprocessed_input)

        style_outputs = [utils.gram_matrix(style_output) for style_output in style_outputs]

        style_dict = {content_name : value 
                        for content_name, value 
                        in zip(self.style_layers,style_outputs)}

        return style_dict



class ContentModel(tf.keras.models.Model):
    def __init__(self, content_layers):
        super(ContentModel,self).__init__()


        self.vgg = vgg_layers(content_layers)
        self.content_layers = content_layers
        self.vgg.trainable = False

    def call(self, inputs):

        inputs = inputs*255.

        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        content_outputs = self.vgg(preprocessed_input)

        content_dict = {content_name : value 
                        for content_name, value 
                        in zip(self.content_layers,content_outputs)}

        return content_dict

        





def neural_style_transfer(config):

    content_img_path = os.path.join(config['content_image_dir'],config['content_image_name'])
    style_img_path = os.path.join(config['style_image_dir'],config['style_image_name'])

    dump_path = config['output_img_dir']

    os.makedirs(dump_path, exist_ok = True)


    content_img = utils.prepare_image(content_img_path, config['height'])
    style_img = utils.prepare_image(style_img_path, config['height'])

    print(content_img.shape)


    if config['init_method'] =='content':

        init_img = content_img

    else :

        style_img_resized = utils.prepare_image(style_img_path, np.asarray(content_img.shape[2:]))

        init_img = style_img_resized

    #On opti une image pas le réseau
    optimizing_img = tf.Variable(init_img, trainable=True)

    content_extractor = StyleModel(style_features_index)
    style_extractor = ContentModel(content_features_index)

    content_targets =content_extractor(content_img)
    style_targets = style_extractor(style_img)

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.02, beta_1=0.99, epsilon=1e-1)


    iterations = 2

    for k in range(iterations):
        print(k)

        with tf.GradientTape() as tape:
            style_outputs  = style_extractor(optimizing_img)
            content_outputs = content_extractor(optimizing_img)


            total_loss = compute_loss(optimizing_img, style_outputs, content_outputs,content_targets,style_targets,config)

        grad = tape.gradient(total_loss,optimizing_img)

        optimizer.apply_gradients([(grad,optimizing_img)])
        optimizing_img.assign(utils.clipping_pixel(optimizing_img))


    return optimizing_img









def compute_loss(optimizing_img,style_outputs,content_outputs,content_targets,style_targets, config):


    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] -content_targets[name])**2) for name in content_outputs.keys()])

    content_loss/= (num_content_layers)


    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])

    style_loss/= num_style_layers


    tv_loss = utils.total_variation(optimizing_img)



    total_loss = style_loss*config['style_weight'] +content_loss*config['content_weight'] + tv_loss * config['tv_weight']

    return total_loss






if __name__=="__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument("--content_image_name", type=str, default = 'lion1.jpg')
    parser.add_argument("--style_image_name", type = str, default = 'ben_giles.jpg')
    parser.add_argument("--height", type=int, default = 600)


    parser.add_argument("--content_weight", type = int, default = 1e-4)
    parser.add_argument("--style_weight", type =int, default = 1e-2)
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

    utils.save_display_img(img).show()





    

