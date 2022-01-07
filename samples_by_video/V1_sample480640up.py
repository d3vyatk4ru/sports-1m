import cv2 as cv
import numpy as np
import h5py as h5
import os
import pandas as pd
import math

HEIGHT = 480
WIDTH = 640
RGB = 3

def get_metadata(path):
    ''' Получаем метданные по видеороликам.
        path - путь к файлу hdf5 одного класса.
    '''
    
    with h5.File(path, 'r') as file:
        df = pd.DataFrame({'name' : file['name'][:],
                           'duration' : file['real_duration'][:],
                           'fps' : file['fps'][:],
                           'frame' : file['real_frame_count'][:],
                           'height' : file['height'][:],
                           'width' : file['width'][:]})
    return df

def get_titles(root):
    ''' Получение путей ко всем файлам.
        root - корневая папка со всеми файлами hdf5.
    '''

    titles = []
    for _, _, files in os.walk(root):
        for file in files:
            titles.append(root + '\\' + file)
    return titles

def get_sample(video, num_frames=8, freq=3, num_samples=20, isFrame=True):
    ''' Получаем 20 сэмплов из видеоролика. количество кадров в сэмпле равно 8-ми с частотой 3.
    
        video - строка из dataframe c характеристиками по одному видео;
        num_frames - количество кадров в сэмпле;
        freq - частота извлечения кадров из одного fps;
        num_samples - количество сэмплов;
        isFrame - логическая переменная для указания работы с кадрами или теоретической частью сэмплов;
    '''
    
    # множитель для увеличения шага в случае большой длительности видео (по умолчанию 1);
    # video[1] - длительность видеоролика.
    if video[1] > 500:
        RATE = 10
    else:
        RATE = 1

    # параметр для управления номером столбца при записи в матрицу
    step = 0
    
    if isFrame:
        
        # это матрица сэмплов. Строки - это сэмплы. Элементы строки - кадры, то есть элменты сэмпла.
        # Элемент - это кадр с разрешением 480x640. 3 - RGB.
        samples = np.zeros((num_samples, num_frames, HEIGHT, WIDTH, RGB), dtype=np.uint8)

        if type(video[0]) == bytes:
            tit = video[0].decode()

        cap = cv.VideoCapture('C:\\Users\\Danya\\Desktop\\Диплом 0_0\\data_for_test_acrh\\Video\\' + tit)
    
        if cap.isOpened():
        
            # проход по каждому каждому "кадру" в сэмпле
            for index in range(num_samples*num_frames):

                # обновление множителя step при прохождения 1 сэпмла <=> смена строки в матрице samples
                # чтобы можно было использовать один цикл для заполнения матрицы
                if index % 8 == 0 and index != 0:
                    step += 1

                cap.set(cv.CAP_PROP_POS_FRAMES, RATE * index * video[2] / freq)
                
                # res - True, если прочитано, иначе False.
                res, frame = cap.read()
                
                # если удалось прочиать кадр, записываем его
                if res:
                    
                    # запись кадр в матрицу и переход от BGR RGB
                    samples[index // num_frames][index - num_frames * step] = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # в случае неудачной попытки чтения кадр записываем предыдущий
                else:
                    try:
                        # случай нахождения на одной строке двух кадров
                        samples[index // num_frames][index - num_frames * step] = \
                                samples[index // num_frames][index - num_frames * step - 1]
                    except:
                        # в случае перехода на следующую строку записываем последний эл-т из предыдущей строки
                        # [*****, *****, ..., *****, *****]
                        # [*****, *****, ..., *****, отсюда]
                        # [<тут>, *****, ..., *****, *****]
                        samples[index // num_frames][index - num_frames * step] = \
                            samples[index // num_frames - 1][index - num_frames * step + num_frames - 1]

            cap.release()
        else:
           print(f'File {video[0]} was not open!')
            
# -------------------------------------------------- считаем сэпмлы без видеороликов  --------------------------------------       
    else:

        # сюда записываются сэмплы. строка = 1 сэмпл.
        samples = np.zeros((num_samples, num_frames), dtype=np.uint16)

        # проход по каждому кадру во всех сэмплах
        for index in range(num_samples*num_frames):

            # обновление множителя step при прохождения 1 сэпмла <=> смена строки в матрице samples
            if index % 8 == 0 and index != 0:
                step += 1

            # проверка на выход за диапазон кадров;
            # video[2] - fps видеоролика;
            # video[3] - кол-во кадров в видеоролике.
            if math.ceil(RATE * index * video[2] / freq) < video[3]:
                
                # записываем кадр RATE * num * video[2] / freq в сэмпл с номером index
                samples[index // num_frames][index - num_frames * step] = math.ceil(RATE * index * video[2] / freq)
                # print(sample_list[index // num_frames][index - num_frames * step], end=' ')

            # если вышли за диапазон, то в настоящий элемент матрицы записываем предыдущий
            else:
                try:
                    # случай нахождения на одной строке
                    samples[index // num_frames][index - num_frames * step] = \
                        samples[index // num_frames][index - num_frames * step - 1]

                except:
                    # в случае перехода на следующую строку записываем последний эл-т из предыдущей строки
                    # [*****, *****, ..., *****, *****]
                    # [*****, *****, ..., *****, отсюда]
                    # [<тут>, *****, ..., *****, *****]
                    samples[index // num_frames][index - num_frames * step] = \
                        samples[index // num_frames - 1][index - num_frames * step + num_frames - 1]

    return samples

def get_class_samples(path, isFrame=False):
    ''' Возвращает сэмплы для всех элементов класса. 
        path - путь к файлу hdf5.
    '''
    
    classes_samples = []
    
    # получение датафрэйма для данных в классе.
    df = get_metadata(path)
    
    for num in range(len(df)):
        
        # df.loc[num][4] - высота видеоролика;
        # df.loc[num][5] - ширина видеоролика.
        
        if df.loc[num][4] >= 480 and df.loc[num][5] >= 640:
            # получение сэмпла для одного видео в классе
            samples = get_sample(df.loc[num], isFrame=isFrame)

            write2hdf5(data=samples)
            # запись сэмплов в список
            # classes_samples.append(samples)
            
    return np.array(classes_samples, dtype=object)

def get_class_samples(path, num_frames=8, freq=3, num_samples=20, isFrame=False):
    ''' Возвращает сэмплы для всех элементов класса. 
        path - путь к файлу hdf5.
    '''
    
    classes_samples = []
    
    # получение датафрэйма для данных в классе.
    df = get_metadata(path)
    
    for num in range(len(df)):
        
        # df.loc[num][4] - высота видеоролика;
        # df.loc[num][5] - ширина видеоролика.
        
        if df.loc[num][4] >= 480 and df.loc[num][5] >= 640:
            # получение сэмпла для одного видео в классе
            samples = get_sample(df.loc[num], num_frames=num_frames, 
                                freq=freq, num_samples=num_samples, isFrame=isFrame)

            write2hdf5(data=samples, num=num)

            # запись сэмплов в список
            # classes_samples.append(samples)
            
    return np.array(classes_samples, dtype=object)

def write2hdf5(path='C:\\Users\\Danya\\Desktop\\Диплом 0_0\\data_for_test_acrh\\Samples\\samples_test_new.hdf5', data=None, num=0):

    if data is not None:
        
        if num == 0:
            with h5.File(path, 'w') as file:
                file.create_dataset(name='samples', data=data, chunks=True, maxshape=(None, 8, 480, 640, 3))
                file.create_dataset(name='targets', data=(np.array([num] * data.shape[0])).transpose(), 
                                    chunks=True, maxshape=(None,))
        else:
            with h5.File(path, 'a') as file:
                file['samples'].resize((file['samples'].shape[0] + data.shape[0]), axis=0)
                file['samples'][-data.shape[0]:] = data

                file['targets'].resize((file['targets'].shape[0] + data.shape[0]), axis=0)
                file['targets'][-data.shape[0]:] = (np.array([num] * data.shape[0])).transpose()

