import cv2 as cv

HEIGHT = 480
WIDTH = 640

class ErrorReadVideoFile(FileNotFoundError):
    pass

def reSizeResolution(path, where='C:\\Users\\Danya\\Desktop\\Диплом 0_0\\Уменьшение разрешения видео\\'):

    cap = cv.VideoCapture(path)

    if not cap.isOpened():

        raise ErrorReadVideoFile(f'No such file or directory: {path}')

    # выбор кодека
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    # fps видеофайла
    fps = cap.get(cv.CAP_PROP_FPS)

    title = where + 'PRESS_' + path.split('\\')[-1]
    
    # создание шаблона видео
    out = cv.VideoWriter(title, fourcc, fps, (WIDTH, HEIGHT))

    ret, frame = cap.read()

    while ret:
        new_frame = cv.resize(frame, (WIDTH, HEIGHT), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
        out.write(new_frame)

        ret, frame = cap.read()

    # каонец работы со всеми объектами
    cap.release()
    out.release()