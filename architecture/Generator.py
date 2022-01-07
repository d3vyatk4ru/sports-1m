# наследование от класса Sequence для подачи наддных в модель
from tensorflow.keras.utils import Sequence
# для чтения из hdf5 файла
from h5py import File

import numpy as np
import random
from datetime import datetime 

# наследуемся от класса Sequence
class DataGenerator(Sequence):
    """ Класс DataGenerator наследуется от Sequence
        и реализует генератор данных для модели глубокого обучения.
    """

    HEIGHT = 480
    WIDTH = 640
    RGB = 3
    FRAMES = 8

    # конструктор класса
    def __init__(self, batch_size=8, shuffle=False, directory=None, Xkey='X_TR', Ykey='Y_TR'):

        # размер батча (пакета)
        self.batch_size = batch_size

        # директория файла с данными 
        self.directory = directory

        # перемешивание
        self.shuffle = shuffle

        # ключ к обучающим данным
        self.Xkey = Xkey

        # ключ к тестовым данным
        self.Ykey = Ykey

        # Общее количество данных в наборе
        self.n_data = __class__._number_of_data(self, directory)

        # список с данными, которые загружаются из чанка в файле.
        self.data = np.empty((self.batch_size, __class__.FRAMES, __class__.HEIGHT, 
                                          __class__.WIDTH, __class__.RGB), dtype=np.float32)

        # список с метками классов для данных в data. Тоже загружаются из чанка в файле.
        self.target = np.empty((self.batch_size), dtype=np.float32)

        # запускаем метод после конца '0' эпохи
        self.on_epoch_end()

    def __len__(self):
        """ Возвращает количество батчей """
        return int(self.n_data / self.batch_size)

    # Откуда index???
    def __getitem__(self, index):
        """ Создание одного пакета (batch) данных """

        # номер батча, который будет загружен в модель на данном шаге в эпохе.
        # Загрузка из диапазона [num_chanks * self.batch_size:(num_chanks + 1) * self.batch_size]
        num_chanks = self.indexes[index]

        # загрузка 1 пакета данных 
        self.__data_generation(num_chanks)

        return self.data, self.target

    def __data_generation(self, num_chanks):
        """ Процесс загрузки данных из чанка в файле. Загрузка 1 батча данных."""
        
        # считываем из файла батч с данными для одной итерации в эпохе.
        # батч - это набор из 4 чанков.
        with File(self.directory, 'r') as file:

            self.data[:] = file[self.Xkey][num_chanks * self.batch_size:(num_chanks + 1) * self.batch_size] / 255.0

            #
            self.target[:] = file[self.Ykey][num_chanks * self.batch_size:(num_chanks + 1) * self.batch_size]
        
        return 0

    def on_epoch_end(self, epoch):
        """ 
            Выполняется после каждой эпохи или через явный вызов.
            Данная реализация выполняет генерацию номеров батчей.
            На новой эпохе порядок взятия чанков (chunk) будет 
            другой. Это сделано, чтобы избежть переобучения. 
        """
        
        # массив с номерами батчей. 1 батч хранится в чанке.
        self.indexes = np.arange(self.n_data // self.batch_size)

        if self.shuffle and epoch % 2 == 1:
            self.indexes = self.indexes[::-1]

    # <--- Вспомогательные методы для работы класса --->
    
    def _number_of_data(self, directory):
        """ 
        Возвращает количество данных в наборе. 
        n_data содержит общее количество данных;
        """

        with File(directory, 'r') as file:
            n_data = file[self.Xkey].shape[0]

        return n_data

