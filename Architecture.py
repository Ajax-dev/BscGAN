#!/usr/bin/env python
# coding: utf-8

# ### Importing relevant modules

# In[1]:


from tensorflow.keras.layers import Dense, Input
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from keras.layers.advanced_activations import LeakyReLU, ReLU
import os


# In[2]:


dir_path = os.path.dirname(os.path.realpath("config.ipynb"))
get_ipython().run_line_magic('run', './config.ipynb')

import config

print(config.TRAIN.batch_size)


# Defining blocks for the generator as descibed in architecture set out in: 
# - https://arxiv.org/pdf/1609.04802.pdf

# In[3]:


def residual(model, filters, kernel_filter, stride):
    
    modelState = model
    
    model = Conv2D(filters = filters, kernel_size = kernel_filter, strides = stride, padding = "same")(model)
    # At the end convolved with the model to produce an output of tensors
    model = BatchNormalization(momentum = 0.5)(model)
    # momentum is momentum of moving averages
    
    model = ReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    # As defined in https://keras.io/api/layers/activation_layers/prelu/
    model = Conv2D(filters = filters, kernel_size = kernel_filter, strides = stride, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    
    model = tensorflow.keras.layers.Add([modelState,model])
    # add original model passed in with new block defined
    
    return model


# In[4]:


def up_sampling_block(model, filters, kernel_size, stride):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = stride, padding = "same")(model)
    # 2 x PixelShuffler
    model = UpSampling2D(size = 2)(model)
    # alpha defaults to 0.3 so left as is
    model = ReLU()(model)
    
    return model


# In[5]:


# Network architecture simplified from paper https://arxiv.org/pdf/1609.04802.pdf
def GenNet(shape):
        
    generator_input = Input(shape = shape)
    generator_model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding="SAME", input_shape = generator_input)
    generator_model = ReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(generator_model)

    generator_model.summary()
    modelTmp = generator_model

    # Start simple using 2 residual blocks
    for i in range(2):
        generator_model = residual(model, 3, 64, 1)

    generator_model = Conv2D(64, (3,3), (1,1), padding = "SAME")(generator_model)
    generator_model = BatchNormalization(momentum = 0.5)(model)
    generator_model = add(generator_model, modelTmp)

    
    # 2 UpSampling of PixelShuffling Blocks
    for i in range(2):
        #up_sampling_block(generator_model, 256, (3,3) , (1,1))
        model = Conv2D(filters = config.TRAIN.filters * 4, kernel_size = config.TRAIN.kernel, strides = config.TRAIN.stride, padding = "same")(model)
        # 2 x PixelShuffler
        model = UpSampling2D(size = 2)(model)
        # alpha defaults to 0.3 so left as is
        model = ReLU()(model)
        generator_model=model

    generator_model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding="SAME")(generator_model)
    generator_model = Activation('tanh')(generator_model)

    generator_model_complete = Model(inputs = generator_input, outputs = generator_model)
    
    
    print("This is the model summary")
    print(generator_model_complete.summary())
    return generator_model_complete
    


# In[6]:


def discriminator_block(model, filters, kernel_size, strides):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model


# In[7]:


# Network architecture simplified from paper stated above
def DisNet(shape):

    discriminator_input = Input(shape = shape)

    discriminator_model = Conv2D(config.TRAIN.filters, (3,3), 1, padding="SAME")(discriminator_input)
    discriminator_model = LeakyReLU()(discriminator_model)

    discriminator_model = discriminator_block(discriminator_model, config.TRAIN.filters, 3, 2)
    discriminator_model = discriminator_block(discriminator_model, config.TRAIN.filters*2, 3, 1)
    discriminator_model = discriminator_block(discriminator_model, config.TRAIN.filters*2, 3, 2)
    discriminator_model = discriminator_block(discriminator_model, config.TRAIN.filters*4, 3, 1)
    discriminator_model = discriminator_block(discriminator_model, config.TRAIN.filters*4, 3, 2)
    discriminator_model = discriminator_block(discriminator_model, config.TRAIN.filters*8, 3, 1)
    discriminator_model = discriminator_block(discriminator_model, config.TRAIN.filters*8, 3, 2)

    discriminator_model = Flatten()(discriminator_model)
    discriminator_model = Dense(1024)(discriminator_model)
    discriminator_model = LeakyReLU()(discriminator_model)

    discriminator_model = Dense(1)(discriminator_model)
    discriminator_model = Activation('sigmoid')(discriminator_model)

    complete_model = Model(inputs = discriminator_input, outputs = discriminator_model)

    
    print("This is the model summary")
    print(model_complete.summary())
    return complete_model


# In[ ]:





# In[ ]:




