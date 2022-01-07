import cv2 as cv
# import matplotlib.pyplot as plt
import numpy as np
import h5py as h5

def get__title_fps(path):
    
    with h5.File(path, 'r') as file:
        title = file['name'][:]
        fps = file['fps'][:]
        frame = file['real_frame_count'][:]
        
    return title, fps, frame

def get_sample2array(path, fps, frame, sample_list, n_frames=8, freq=3):
    '''
    path --- путь к файлу (URL или обычный путь на локальном ПК);
    
    fps --- количество кадров в секунду. 
    
    n_frames --- количество кадров в семпле;
    
    freq --- частота кадров, которые необходимо поместить в сэмпл, за 1 fps.
    '''
    # считываем видео ???
    cap = cv.VideoCapture(path)
    
    if cap.isOpened():
        # список для сохранения кадров
        sample = []
        
        for num_frame in range(frame):
            # устанавливаем номер кадра (num_frame * frame_step);
            # cv.CAP_PROP_POS_FRAMES - константа равная 1.
            cap.set(cv.CAP_PROP_POS_FRAMES, int(num_frame * fps / freq))
            
            # res - логический рез-т операции. Если прочитано - True, иначе False.
            res, frame_inv_col = cap.read()
            
            # если выходим за диапазон (видео кончается из-за большого шага),
            # то в недостающие кадры записываем последний считанный фрагмент (кадр).
            if res:
                # переходим от BGR к RGB
                frame = cv.cvtColor(frame_inv_col, cv.COLOR_BGR2RGB)
                
                if num_frame % n_frames != 0 or num_frame == 0:
                    sample.append(frame)
                    
                elif num_frame % n_frames == 0 and num_frame != 0:
                    sample_list.append(sample)
                    del sample
                    sample = []
                    sample.append(frame)
            else:
                break
        
        cap.release()
        return sample
    else:  
        print('File was not open!')
        cap.release()
        return None

path = 'C:\\Users\\Danya\\Desktop\\Диплом 0_0\\sports1m_test2\\test2\\++video_data2_001.hdf5'
root = 'https://www.youtube.com/watch?v='
sample_list = []

title, fps, frame = get__title_fps(path)

url = root + str(title[0])[2:-5]

print(url)

a = get_sample2array('C:\\Users\\Danya\\Desktop\\Диплом 0_0\\Video\\2.mp4', 24, 792, sample_list, n_frames=8, freq=3)