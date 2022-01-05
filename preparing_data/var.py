import h5py as h5
import numpy as np

SAVE_PATH = '/data/d3vyatk4ru/july_training/corr_between_frames/var_TS_Sports-1M.hdf5'
SIZE_TS = 20_000
PATH_TS = '/dcache/sports1m/sports1m_ts_big.hdf5'

def get_data(path): 

    with h5.File(path, 'r') as f:
        return f['X_TS'][...]

def write2h5(arr_var):
    with h5.File(SAVE_PATH, 'a') as file:
        file.create_dataset('corr', (SIZE_TS, 7), dtype=np.float32, data=arr_var)

def computing_var(path):

    # массив для дисперсии между соседними кадрами
    arr_var = np.empty((SIZE_TS, 7), dtype=np.float32)

    sample = np.empty((SIZE_TS, 8, 480, 640, 3), dtype=np.float32)

    sample = get_data(path)

    for idx, smpl in enumerate(sample):
        for frame in range(len(smpl) - 1):
            arr_var[idx, frame] = np.var(smpl[0] - smpl[frame + 1])
        print(arr_var[idx])

    write2h5(arr_var)

if __name__ == '__main__':

    computing_var(PATH_TS)