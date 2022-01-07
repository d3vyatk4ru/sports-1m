import V1_sample480640up as nsf
from datetime import datetime
import os

path = os.path.abspath('test_for_arch.hdf5')

# получение пути для каждого файла hdf5
titles = nsf.get_titles(path)

print('Программа запущена...')

# список с сэпмлами из каждого класса
samples = []

# вычисление сэмплов для классов
start = datetime.now()

for title in titles:    
    class_samples = nsf.get_class_samples(title, isFrame=True)
    samples.append(class_samples)

print(f"Время работы функции: {(datetime.now() - start).total_seconds()}")



