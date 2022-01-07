from tensorflow.keras.layers import Conv3D, MaxPooling2D, BatchNormalization, Dropout, Input
from tensorflow.keras.layers import Dense, Flatten, TimeDistributed
from tensorflow.keras import Model

import numpy as np
import h5py as h5

# импорт класса Callback для наследования и создания callback'a для F1
from tensorflow.keras.callbacks import Callback
# подсчёт F1 с помощью f1_score и матрицы ошибок
from sklearn.metrics import f1_score, confusion_matrix

import subprocess
import sys

# информация о GPU
def gpu_status():
  gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader"])#,noheader,nounits"])
  gpu_info = gpu_info.decode('UTF-8')
  print("gpu_info:", gpu_info, end = '')
  sys.stdout.flush()

class NumberOfClassesSmallerThen20(ValueError):
  pass 

def create_model():
    ''' Создание модели, основанной на 3D свертках '''

    # 8 is num of frames in samples. 20 it's num of samples for 1 video.
    input_shape = (8, 480, 640, 3)

    inputs = Input(shape=input_shape, name='Input_samples')

    # adding convolution block [pooling + conv + nonlinear activation]
    l_maxPool = MaxPooling2D(pool_size=2, name='maxPool2D_1')
    l_maxPool = TimeDistributed(l_maxPool)(inputs)

    # convolution
    l_Conv3D = Conv3D(filters=32, kernel_size=3, activation='relu',
                                                name='Conv3D_layer_1')(l_maxPool)

    # regularization 
    dropout = Dropout(rate=0.5, name='Dropout_1')(l_Conv3D)

    l_maxPool = MaxPooling2D(pool_size=3, name='maxPool2D_2')
    l_maxPool = TimeDistributed(l_maxPool)(dropout)

    # convolution
    l_Conv3D = Conv3D(filters=64, kernel_size=3, activation='relu',
                                                name='Conv3D_layer_2')(l_maxPool)

    # regularization 
    dropout = Dropout(rate=0.5, name='Dropout_2')(l_Conv3D)

    l_maxPool = MaxPooling2D(pool_size=3, name='maxPool2D_3')
    l_maxPool = TimeDistributed(l_maxPool)(dropout)

    # convolution
    l_Conv3D = Conv3D(filters=128, kernel_size=3, activation='relu',
                                                name='Conv3D_layer_3')(l_maxPool)

    # regularization 
    dropout = Dropout(rate=0.5, name='Dropout_3')(l_Conv3D) 

    batchnorm_layer = BatchNormalization(momentum=0.9)(dropout)

    flatten = Flatten(name='Flatten_layer')(batchnorm_layer)

    dense = Dense(128, activation='relu', name='Dense_layer')(flatten)

    dropout = Dropout(rate=0.5, name='Dropout_4')(dense)

    outputs = Dense(20, activation='softmax', name='Output_dist')(dropout)

    model = Model(inputs=inputs, outputs=outputs, name='Architecture_0')

    return model

class F1(Callback):

  # конструктор
  def __init__(self, generator):
    self.generator = generator
    self.validation_target = np.empty((20000, ))
    self.validation_predict = np.empty((20000, ))
    self.batch_size = self.generator.batch_size

  # вызывается при начале обучения модели
  def on_train_begin(self, logs=None):
    # инициализация листа со значениями F1 score для каждой эпохи
    self.F1_metric = []
    self.conf_matrix= []

  # вызывается в конце каждой эпохи
  def on_epoch_end(self, epoch, logs=None):

    i = 0
    for X, y in self.generator:
      self.validation_predict[self.batch_size * i:self.batch_size * (i + 1)] = np.argmax(np.asarray(self.model.predict(X)), axis=1)
      self.validation_target[self.batch_size * i:self.batch_size * (i + 1)] = y
      i += 1

    # F1 метрика для каждого класса
    _val_F1_for_all_class = f1_score(self.validation_target, self.validation_predict, average=None)
    # F1 для всей валидации
    _val_F1 = f1_score(self.validation_target, self.validation_predict, average='macro')
    # матрица ошибок
    _сonf_matr = confusion_matrix(self.validation_target, self.validation_predict)

    if self.verbose:
      print(' — val_F1_classes: {}'.format(_val_F1_for_all_class), end=' ')
      print(' — val_F1: {}'.format(_val_F1))
      print(f' — Confusion matrix: \n {_сonf_matr}')

    # добавление в массив F1 для данной эпохи
    self.F1_metric.append(_val_F1)
    # добавление матрицы ошибок за эпоху
    self.conf_matrix.append(_сonf_matr)

    with open('/data/d3vyatk4ru/Test_ARCH[0]/F1_metric.txt', 'a') as file:
      for cls in _val_F1_for_all_class:
        file.write(str(cls) + ' ')
      file.write('\n')

    with open('/data/d3vyatk4ru/Test_ARCH[0]/F1_metric_all_cls.txt', 'a') as file:
      file.write(str(_val_F1) + '\n')

    with open('/data/d3vyatk4ru/Test_ARCH[0]/confusion_matrix.txt', 'a') as file:
      file.write(str(_сonf_matr) + '\n')
