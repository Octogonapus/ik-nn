#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers

print(tf.__version__)
print(tf.keras.__version__)

# In[4]:


dataset_path = "ik_3dof_planar.csv"

# In[5]:


column_names = ["translation_x", "translation_y", "rot_z", "theta1", "theta2", "theta3"]
raw_dataset = pd.read_csv(dataset_path,
                          names=column_names,
                          na_values="?",
                          sep=",",
                          skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

print(dataset.isna().sum())
dataset = dataset.dropna()

# In[6]:


train_dataset = dataset.sample(frac=0.8, random_state=0)
train_labels = train_dataset[["theta1", "theta2", "theta3"]]
test_dataset = dataset.drop(train_dataset.index)
test_labels = test_dataset[["theta1", "theta2", "theta3"]]

# In[7]:


model = keras.Sequential([
    layers.Dense(100, activation=tf.nn.tanh, input_shape=[len(train_dataset.keys())]),
    layers.Dense(3, activation=tf.keras.activations.linear)
])

# In[8]:


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=tf.keras.losses.mean_squared_error,
              metrics=['accuracy', 'mean_squared_error'])

print(model.summary())


# In[9]:


class PrintDot(keras.callbacks.Callback):
    # noinspection PyMethodOverriding
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


# In[19]:


history = model.fit(train_dataset,
                    train_labels,
                    epochs=1000,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[PrintDot()])
print('')

# In[20]:


history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch
print(history_df.tail())


# In[36]:


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.semilogy()
    plt.legend()
    plt.savefig('history.png')


# In[37]:


plot_history(history)

# In[16]:


loss, accuracy, mse = model.evaluate(test_dataset, test_labels, verbose=0)
print("Test dataset Accuracy: {} Mean Square Error: {}".format(accuracy, mse))

# In[ ]:
