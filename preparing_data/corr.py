import h5py as h5
import numpy as np

def correlation_coefficient(img1, img2):
    ''' Calculate correlation between 2 RGB img '''

    product = np.mean((img1 - img1.mean()) * (img2 - img2.mean()))
    std = img1.std() * img2.std()

    if std < 1e-5:
        return 0
    else:
        product /= std
        return product

def get_data(path): 

    with h5.File(path, 'r') as f:
        return f['X_TS'][...]

def computing_corr(path):

    # массив для корреляции между соседними кадрами
    arr_corr = np.empty((SIZE_TS, 7), dtype=np.float32)

    sample = np.empty((SIZE_TS, 8, 480, 640, 3), dtype=np.float32)

    sample = get_data(path)

    for idx, smpl in enumerate(sample):
        for frame in range(len(smpl) - 1):
            arr_corr[idx, frame] = correlation_coefficient(smpl[0], smpl[frame + 1])
        print(arr_corr[idx])

    write2h5(arr_corr)

def write2h5(arr_corr):
    with h5.File(SAVE_PATH, 'a') as file:
        file.create_dataset('corr', (SIZE_TS, 7), dtype=np.float32, data=arr_corr)
        
if __name__ == '__main__':

    SIZE_TS = 20_000
    PATH_TS = '/dcache/sports1m/sports1m_ts_big.hdf5'
    SAVE_PATH = '/data/d3vyatk4ru/july_training/corr_between_frames/corr_TS_Sports-1M.hdf5'

    computing_corr(PATH_TS)
