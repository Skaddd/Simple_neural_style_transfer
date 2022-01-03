import numpy as np
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image
from tensorflow.python.ops.gen_math_ops import imag

def load_image(img_path,target_shape = None):

    if not os.path.exists(img_path):

        raise Exception(f'Path does not exist : {img_path}')


    image = tf.io.read_file(img_path)
    image = tf.image.decode_image(image)

    if target_shape is not None :
        if isinstance(target_shape, int) and target_shape >0: # si c'est un scalaire c'est la height de l'image

            height, width = image.shape[:2]
            new_height = target_shape
            new_width = int(width * (new_height/height))

            image = tf.image.resize(image,(new_height,new_width))

        else :

            image = tf.image.resize(image,target_shape)



    image = tf.cast(image, dtype = tf.float32)

    image = image/255.
    return image[tf.newaxis, :]

    
def prepare_image(img_path,target_shape):

    image = load_image(img_path,target_shape)
    #image = tf.keras.applications.vgg19.preprocess_input(image*255) #probablement pas utile car fait plutard

    #image = tf.image.resize(image,(224,224)) C'eest les dernières couches qui limites la taille de l'image! les FC layers or on les a viré!

    return image


def gram_matrix(input_tensor):

    """la matrice de Gram est simplement le produit scalaire des composantes d'une matrice"""
    """
    Dans le cas du Neural Style Transfer les Features Map sont Flatten.
    Puis on calcule le pdt scalaire entre chaque Feature Map.
    On obtient Alors une matrice qui décrit les dépendances entre Feature Map.

    bijc -> (Batch, I-lignes,J-Colonnes,C-Channels) ou D-Depth

    Mais technique c'est forcément une matrice carré symétrique, c = d
    
    """

    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)

    #print(input_shape)

    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)



def total_variation(image):

    """
    Total variation is a measure of the complexity of an image with respect to its spatial varation. 
    It can be computed using different norms. With the built-in fonc from tensorflow it's simply
    the absolute differences of neighbors pixels.

    """
    return tf.reduce_sum(tf.image.total_variation(image))



def clipping_pixel(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def save_display_img(tensor_img, output_file, should_display = False):

    tensor_img = tensor_img*255.
    tensor_img = np.asarray(tensor_img, dtype = np.uint8)
    if np.ndim(tensor_img)>3:
        tensor_img = np.squeeze(tensor_img)

    print(tensor_img.shape)

    out_image = PIL.Image.fromarray(tensor_img)
    out_image.save(output_file)

    if should_display:
        out_image.show()


    return 


if __name__=="__main__":


    img = load_image("/home/mateo/Documents/style_transfer/data/content-images/lion.jpg",target_shape=None)

    save_display_img(img).show()