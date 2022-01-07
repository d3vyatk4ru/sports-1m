import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import os
import pandas as pd
import math

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
    
    # список для 1 сэмпла.
    sample = []
    # сюда записываются сэмплы.
    sample_list = []
    
    # множитель для увеличения шага в случае большой длительности видео (по умолчанию 1);
    # video[1] - длительность видеоролика.
    if video[1] > 500:
        RATE = 10
    else:
        RATE = 1
    
    if isFrame:

        if type(video[0]) == bytes:
            video[0] = video[0].decode()

        cap = cv.VideoCapture(video[0])
    
        if cap.isOpened():
        
            # проход по каждому кадру из всех сэмплов.
            for n_frame in range(num_samples * num_frames):
                
                # устанавливаем номер кадра (n_frame + 1) * video[2] / freq), RATE - длина шага.
                # video[2] - fps видеоролика
                cap.set(cv.CAP_PROP_POS_FRAMES, RATE * (n_frame + 1) * video[2] / freq)
                
                # res - True, если прочитано, иначе False.
                res, frame = cap.read()
                
                if res:

                    if n_frame % num_frames != 0 or n_frame == 0:

                        # перeход от BGR к RGB (особенности openCV) и запись в список
                        sample.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                    elif n_frame % num_frames == 0 and n_frame != 0:

                        # добавление сэпмла в список
                        sample_list.append(sample)
                        # очистка списка для кадров в сэмлпе
                        del sample
                        sample = []
                        # добавление кадра в новый сэмпл
                        sample.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                else:
                    sample_list.append(np.array(sample))
                    del sample[:]
                    break
            
            cap.release()
            sample_list = np.array(sample_list)
        else:
            print(f'File {video[0]} was not open!')
            
# -------------------------------------------------- считаем сэпмлы без видеороликов  --------------------------------------       
    else:
        # проход по каждому кадру из всех сэмплов.
        for n_frame in range(num_samples * num_frames):
            
            # n_frame - номер кадра

            # проверка на выход за диапазон кадров;
            # video[2] - fps видеоролика;
            # video[3] - кол-во кадров в видеоролике.
            if math.ceil(n_frame * video[2] / freq) < video[3]:
                
                # проверка на кратность номера кадра к общему кол-ву кадров в сэмпле,
                # если кратно, значит записываем новый сэмпл. Ноль тоже учитывается.
                if n_frame % num_frames != 0 or n_frame == 0:
                    sample.append(math.ceil(n_frame * video[2] / freq))
                    # print(math.ceil(n_frame * video[2] / freq), end=' ')

                elif n_frame % num_frames == 0 and n_frame != 0:
                    # добавление сэпмла в список
                    sample_list.append(sample)
                    # очистка списка для кадров в сэмлпе
                    del sample
                    sample = []
                    # добавление кадра в новый сэмпл
                    sample.append(math.ceil(n_frame * video[2] / freq))
                    # print(math.ceil(n_frame * video[2] / freq), end=' ')
            # при выходе за диапазон кадров, запись получившегося "неполного" сэмпла.
            else:
                sample_list.append(np.array(sample))
                del sample
                break

    return sample_list

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
        
            # запись сэмплов в список
            classes_samples.append(samples)
            
    return np.array(classes_samples, dtype=object)