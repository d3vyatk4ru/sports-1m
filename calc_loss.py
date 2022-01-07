import h5py as h5
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from os import walk
import numpy as np

FOLDER_LABEL_TR = 'C:\\Users\\Danya\\Desktop\\Dense_pred_TR\\'
FOLDER_LABEL_TS = 'C:\\Users\\Danya\\Desktop\\Dense_pred_TS\\'

N_TS = 20_000
N_TR = 200_000

class NoDataSetType(ValueError):

    def __init__(self, message='Не указан тип датасета') -> None:
        super().__init__(message)

class NoFolderWithFiles(ValueError):

    def __init__(self, message='Не указана папка с данными!') -> None:
        super().__init__(message)

class NoNameFile(ValueError):

    def __init__(self, message='Не указан файл с данными!') -> None:
        super().__init__(message)

class Loss():

    def __init__(self, files_folder, ds_type) -> None:

        self.files_folder = files_folder
        self.ds_type = ds_type

    def _file_names(self) -> str:

        for _, _, files in walk(self.files_folder):

            for file in files:
                
                if file.split('.')[-1] in ['h5', 'hdf5']:

                    yield file

    def _load_data(self, y_true=False) -> np.array(list()):

        if self.files_folder is None:
            raise NoFolderWithFiles

        if self.ds_type is None:
            raise NoDataSetType

        if y_true:
            key = 'y_true'
        else:
            key = 'y_softmax'

        if self.ds_type == 'TR':
            key += '_tr'

        if self.ds_type == 'TS':
            key += '_ts'

        for fname in _file_names(self):
            with h5.File(self.files_folder + fname, 'r') as f:
                arr = f[key][...]

            return arr

def load_data(fname=None, ds_type=None, y_true=False):

    if fname is None:
        raise NoNameFile

    if ds_type is None:
        raise NoDataSetType

    if y_true:
        key = 'y_true'
    else:
        key = 'y_softmax'

    if ds_type == 'TR':
        key += '_tr'

    if ds_type == 'TS':
        key += '_ts'

    with h5.File(FOLDER_LABEL_TS + fname, 'r') as f:
        arr = f[key][...]

    return arr

def get_fname(path):

    fnames = []

    for _, _, files in walk(path):

        for file in files:
            if file.split('.')[-1] in ['h5', 'hdf5']:
                fnames.append(file)

    return fnames


if __name__ == '__main__':

    arr_Y_true = np.empty((N_TR, ), dtype=np.uint8)
    arr_Y_softmax = np.empty((N_TR, 20, ), dtype=np.float32)

    file_names = get_fname(FOLDER_LABEL_TS)

    arr_Y_true = load_data(file_names[0], ds_type='TS', y_true=True)

    loss = []

    scce = SparseCategoricalCrossentropy()

    for file in file_names:

        arr_Y_softmax = load_data(file, ds_type='TS')

        loss.append(scce(arr_Y_true, arr_Y_softmax).numpy())

        with open('C:\\Users\\Danya\\Desktop\\loss_dense_TS.txt', 'a') as f:
            f.write(str(loss[-1]) + '\n')

        print(file)


    print(loss)