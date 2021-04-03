#!/usr/bin/env python
# coding: utf-8

# In[4]:


from easydict import EasyDict as edict


# In[5]:


config = edict()
config.TRAIN = edict()


# In[6]:


## Adam optimizer settings
config.TRAIN.batch_size = 8

#initialise generator
config.TRAIN.epochs = 1

# initialise GAN
config.TRAIN.epochs = 20

# images
config.TRAIN.hr_images = '../BSR/BSDS500/data/images'
config.TRAIN.training_img = config.TRAIN.hr_images + '/train'
config.TRAIN.testing_img = config.TRAIN.hr_images + '/test'
config.TRAIN.validation = config.TRAIN.hr_images + '/val'
config.TRAIN.downscale = 4
config.TRAIN.image_shape = (256,256,3)
config.TRAIN.lr_init = 1e-3
config.TRAIN.filters = 64
config.TRAIN.kernel = (3,3)
config.TRAIN.stride = 1

