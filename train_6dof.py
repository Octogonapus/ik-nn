#!/usr/bin/env python
# coding: utf-8

# In[81]:


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


# In[82]:


dataset_path = "ik_6dof.csv"


# In[83]:


column_names = ["translation_x", "translation_y", "translation_z", "theta1", "theta2", "theta3",
                "theta4", "theta5", "theta6"]
raw_dataset = pd.read_csv(dataset_path,
                          names=column_names,
                          na_values="?",
                          sep=",",
                          skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

print(dataset.isna().sum())
dataset = dataset.dropna()


# In[84]:


train_dataset = dataset.sample(frac=0.8, random_state=0)
train_labels = train_dataset[["theta1", "theta2", "theta3", "theta4", "theta5", "theta6"]]
test_dataset = dataset.drop(train_dataset.index)
test_labels = test_dataset[["theta1", "theta2", "theta3", "theta4", "theta5", "theta6"]]


# In[85]:


model = keras.Sequential([
    layers.Dense(128, activation=tf.nn.tanh, input_shape=[len(train_dataset.keys())]),
    layers.Dense(128, activation=tf.nn.tanh),
    layers.Dense(len(train_labels.keys()), activation=tf.keras.activations.linear)
])


# In[86]:


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=tf.keras.losses.mean_squared_error,
              metrics=['accuracy', 'mean_squared_error'])

print(model.summary())


# In[87]:


class PrintDot(keras.callbacks.Callback):
    # noinspection PyMethodOverriding
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


# In[88]:


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_dataset,
                    train_labels,
                    epochs=1000,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[early_stop, PrintDot()])
print('')


# In[89]:


history_df = pd.DataFrame(history.history)
history_df['epoch'] = history.epoch
print(history_df.tail())


# In[90]:


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
    plt.show()


# In[91]:


plot_history(history)


# In[92]:


loss, accuracy, mse = model.evaluate(test_dataset, test_labels, verbose=0)
print("Test dataset Accuracy: {} Mean Square Error: {}".format(accuracy, mse))


# In[ ]:




