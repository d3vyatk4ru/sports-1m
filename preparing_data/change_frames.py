import numpy as np
import h5py as h5

PATH_TS = "D:\\sports1m_ts_big.hdf5"
STEP = 50
SIZE_TS = 20_000
PATH_TS_SHUFFLE = ''

def get_data(l_border=0):
    with h5.File(PATH_TS, 'r') as f:
        return f['X_TS'][l_border: l_border + STEP]

def shuffle_frames():

    frames = np.array((STEP, 8, 480, 640, 3), dtype=np.float32)

    with h5.File(PATH_TS_SHUFFLE, 'w') as f:
        f.create_dataset('X_TS',  (SIZE_TS, 8, 480, 640, 3), dtype = np.float32))

    for border in range(0, SIZE_TS, STEP):

        frames = get_data(border)

        for i in range(len(TS_S)):
            np.random.shuffle(TS_S[i])