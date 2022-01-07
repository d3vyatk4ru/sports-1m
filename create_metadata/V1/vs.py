import cv2 as cv
import os
import h5py as h5

class Video(object):

    def __init__(self, video_path : str) -> None:
        '''
        Инициализация экземпляров класса. Записываем путь видеофайла и проверяем версию OpenCV;
        
        video_path: путь к видеофайлу.
        '''

        self.video_path = video_path

        self.VERSION3 = self._is_cv34()

    @staticmethod
    def _is_cv34():
        '''
        Функция _is_cv34() возвращает True, если OpenCV имеет 3-ью версию и выше. Иначе False. 
        '''
        (major, _, _) = cv.__version__.split('.')

        print(cv.__version__)

        return True if int(major) >= 3 else False

    def _custom_video_time(self, cap, n_frames):
        '''
        Функция возвращает продолжительность, но без встроенного методы библиотеки OpenCV.
        Делаем обходным способом: сначала считаем частоту кадров, дальше делим 
        частоту кадров на количество кадров;

        cap: указатель на видео;
        n_frames: количество кадров.
        '''
        # check if we are using OpenCV 3-4
        if self.VERSION3:
            fps = cap.get(cv.CAP_PROP_FPS)
        # we are using OpenCV < 3
        else:
            fps = cap.get(cv.cv.CV_CAP_PROP_FPS)

        # if fps is equal zero then it's bad.
        try:
            video_timing = n_frames / fps
        # calls exception
        except:
            video_timing = -1

        return video_timing

    def _video_time(self, cap, n_frames):
        '''
        Функиця возвращает продлжительность видео, используя методы библиотеки OpenCV. 
        Если по каким-то причинам эти методы возвращают неправильно значение, то используется 
        пользовательская функция _custom_video_time().
        '''

        video_timing = -1

        try:
            if self.VERSION3:
                video_timing = round(cap.get(cv.CAP_PROP_POS_MSEC) / 1000, 2)
            else:
                video_timing = round(cap.get(cv.cv.CV_CAP_PROP_POS_MSEC) / 1000, 2)
        except:

            video_timing = self._custom_video_time(cap, n_frames)

        finally:
            if int(video_timing) == 0:
                video_timing = self._custom_video_time(cap, n_frames)

        return video_timing

    def _custom_count_frames(self, cap):
        '''
        Функция count_frames_manual() считает кол-во кадров без использования 
        встроенных методов OpenCV. Это делается с помощью обходных функций;

        cap: указатель на видео.
        '''
        # initialize value
        n_frames = 0

        while True:

            # read frame by frame
            # ret --- boolean value. If frame has been read - True, else False.
            # _ --- current frame. Using name '_', because won't use this value
            ret, _ = cap.read()

            # when is the end of the video --- exit
            if not ret:
                break
    
            n_frames += 1

        # transfer to first frame
        if self.VERSION3:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        else:
            cap.set(cv.cv.CV_CAP_PROP_POS_FRAMES, 0)

        return n_frames

    def _custom_resolution(self, cap):
        '''
        Функция возвращает высоту и ширину видео. Для этого берется 1 кадр. Так как кадр 
        является numpy array, то смотрятся его размеры;

        cap: указатель на видео.
        '''

        _, current_frame = cap.read()

        return current_frame.shape[0:2]

    def _count_frames(self, cap):
        '''
        Функция count_frames() возвращает количество кадров n_frames. Сначала используется 
        подход со встроенными функциями OpenCV для подсчета кадров. Если этого 
        не  удается сделать с помощью встроенных фукция, то вызывается 
        пользовательская функция _custom_count_frames();

        self: аналог указатель this в С++. Указаывает на атрибуты созданного объекта.
        path: путь к файлу.
        '''
        # initialize frames value
        n_frames = 0

        try:
            # check if we are using OpenCV 3-4
            if self.VERSION3:
                n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            # we are using OpenCV < 3
            else:
                n_frames = int(cap.get(cv.cv.CV_CAP_PROP_FRAME_COUNT))
    
        # It's a problems with build-in methods. We'll do self method
        except:
            # Own method
            n_frames = self._custom_count_frames(cap)

        finally:
            #  36k because it's number of frame by 10 minutes video
            if n_frames > 36000 or n_frames < 0:
                n_frames = self._custom_count_frames(cap)

        return n_frames

    def _resolution(self, cap):
        '''
        Функция resolution возвращает высоту и ширину видео. Сначала используеются встроенные функции
        в библиотеке OpenCV, если с помощью них не удается этого сделать, то управление
        передается пользовательской функции _custom_resolution().

        cap: указатель на видео.
        '''

        # first initialization width and height frame's
        width, height = 0, 0

        try:
            if self.VERSION3:

                width  = cap.get(cv.CAP_PROP_FRAME_WIDTH)  # float too?
                height = cap.get(cv.CAP_PROP_FRAME_HEIGHT) # float too?
            else:

                width  = cap.get(cv.cv.CV_CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv.cv.CV_CAP_PROP_FRAME_HEIGHT) 
        except:

            height, width = self._custom_resolution(cap)

        return width, height

    def get_video_characteristics(self, verbose=True):
        '''
        Функция get_characteristics() получает на вход видео и возвращает его характеристики.
        Характеристиками являются: fps (количество кадров в секунду), экранное разрешение (resolution) ---
        количество пикселей по горизонтали и вертикали, длительность видео.

        path: путь к файлу.

        В случае успешного выполнения открытия файла и выполнения кода 
        возвращает все необходимые характеристики.
        Если не удалось открыть видеофайл, то возвращает None для всех характеристик.


        '''
    
        # take a pointer to video file
        cap = cv.VideoCapture(self.video_path)
    
        if cap.isOpened():

            # get video's resolution. It's width and height
            width, height = self._resolution(cap)

            # compution number of frames
            n_frames = round(self._count_frames(cap))
    
            # Я не знаю, почему это так, но если мы ставим другой кадр n_frames,
            # то время видео будет 0. Но если поставить последний кадр (отсчет с нуля),
            # и считать его, то время видео нормальное...
            cap.set(cv.CAP_PROP_POS_FRAMES, n_frames - 1)
            _, _ = cap.read()

            # calculation video timing and divide by 1000 because ms 
            video_timing = self._video_time(cap, n_frames)

            try:
                fps = n_frames / video_timing 
            except:
                fps = -1

            cap.release()  
        
            if verbose:
                print('Frames are {:.1f} [frames]'.format(fps))
                print('Video timing is {:.1f} [sec]'.format(video_timing))
                print('Height and width are {:} and {:} [pix]'.format(height, width))       
            
            return n_frames, video_timing, width, height

        else: 
            print('Video was not open!')
            return None, None, None, None
            # raise Exception('Video was no open!')    

    @staticmethod
    def write2hdf5(videodata, file_name='video_data.hdf5', video_title=None):
        '''
        Функция write2File принимает numpy array для записи в файл *.hdf5.
    
        videodata: характеристики видеофайла;
    
        file_name: имя файла для записи характеристик;

        save_video_title: список с именами видеофайлов. По умолчанию он пуст
        и его запись не происходит.
        '''
        with h5.File(file_name, 'w') as file:
            file.create_dataset('id', data=range(1, len(videodata)))

            if video_title is not None:
                file.create_dataset('title', data=video_title)

            file.create_dataset('frames', data=videodata[:, 0])
            file.create_dataset('timing', data=videodata[:, 1])
            file.create_dataset('width', data=videodata[:, 2])
            file.create_dataset('height', data=videodata[:, 3])

class Path(object):

    def __init__(self, main_path : str) -> None:
        '''
        Получает путь к папке, где лежат видео данные;

        main_path: путь к папке с видеофайлами. 
        '''
        self.main_path = main_path

    def get_paths(self) -> list:
        '''
        Возвращает путь к каждому видео файлу. 
        '''

        all_path = []

        for root, _, files in os.walk(self.main_path):
            for file in files:
                # create path to the video and add this path to list
                all_path.append(os.path.join(root, file))

        return all_path

    @classmethod
    def get_title_video(cls, path):
        '''
        Возвращает список имен каждого видео файла. Используется без 
        создания экземпляра класса. 
        '''
        # it's list with titles of videos
        video_title = []

        for _, _, files in os.walk(path):
            for file in files:
                video_title.append(file)

        return video_title