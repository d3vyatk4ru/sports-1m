import h5py as h5
import numpy as np
import os
import sys
import tensorflow as tf

# ссылка на файл с тренировочными данными
FILE_TR = '/dcache/sports1m/sports1m_tr_big.hdf5'

# ссылка на файл с тестовыми данными
FILE_TS = '/dcache/sports1m/sports1m_ts_big.hdf5'

# the sample's size
FRAMES, HEIGHT, WIDTH, RGB = 8, 480, 640, 3
PART = 1_000

# the train dataset size
N_TR = 200_000
# the test dataset size
N_TS = 20_000

BATCH_SIZE = 100

def load_data(ds_type='TS', num_part=0):

    ''' Read data from hdf5 file '''
  
    if ds_type == 'TS':
        file_name = FILE_TS

    if ds_type == 'TR':
        file_name = FILE_TR

    with h5.File(file_name, 'r') as file:

        X = file["X_" + ds_type][PART * num_part: PART * (num_part + 1)]
        Y = file["Y_" + ds_type][PART * num_part: PART * (num_part + 1)]

    return X, Y

def parse_function(sample, label):
    ''' Normalizing one sample from dataset and 
        transforming sample's type to tf.float32.

        sample: sequence of 8 frames;
        label: sample's target. 
    '''

    # Don't use tf.image.decode_image, or the output shape will be undefined
    #image = tf.image.decode_jpeg(image_string, channels=3)

    #This will convert to float values in [0, 1]
    image = tf.cast(sample, dtype = tf.float32)
    image /= 255.0

    # image = tf.image.resize_images(image, [64, 64])
    return image, label

def get_paths(path):

    ''' Getting all path to file with weight/checkpoint and file's name

        path: it's path to folder with weight/checkpoint files.
    '''

    all_path = []
    file_names = []

    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] in ['.h5']:

                all_path.append(os.path.join(root, file))
                file_names.append(file)

    return all_path, file_names


if __name__ == '__main__':

    num_part = 0

    where_models = '/data/d3vyatk4ru/architecture/GAP_train2/checkpoint'

    all_path, file_names = get_paths(where_models)

    arr_Y_true = np.empty((N_TS, ), dtype=np.uint8)
    arr_Y_pred = np.empty((N_TS, ), dtype = np.uint8)
    arr_Y_softmax = np.empty((N_TS, 20), dtype = np.float32)

    print('[main.py] Модель загружена')
    sys.stdout.flush()

    for i in range(len(all_path)):

        print('[main.py]' + all_path[i])
        sys.stdout.flush()
        
        model = tf.keras.models.load_model(all_path[i],
                                           custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU})

        with h5.File(file_names[i] + '_results.h5', 'a') as f:
            f.create_dataset(name = 'y_pred', maxshape=(N_TR, ), dtype=arr_Y_pred.dtype)
            f.create_dataset(name = 'y_softmax', maxshape=(N_TR, 20), dtype=arr_Y_softmax.dtype)

        for num_part in range(0, N_TR, PART):
    
            X, Y = load_data(ds_type='TS', num_part=num_part)
            print("X:", X.shape, X.dtype)
            print("Y:", Y.shape, Y.dtype)
            sys.stdout.flush()

            dataset = tf.data.Dataset.from_tensor_slices((X, Y))
            dataset = dataset.map(parse_function, num_parallel_calls=4)
            dataset = dataset.batch(BATCH_SIZE)
            dataset = dataset.prefetch(1)

            print('[main.py] dataset created')
            sys.stdout.flush()

            arr_Y_softmax = model.predict(dataset, verbose=1,
                            max_queue_size=10, workers=8)
            print('[main.py] predicted was end')
            sys.stdout.flush()

            arr_Y_pred = np.argmax(arr_Y_softmax, axis=1).astype(np.uint8)

            with h5.File(file_names[i] + '_results.h5', 'a') as f:
                f['y_pred'].resize((f['y_pred'].shape[0] + arr_Y_pred.shape[0]), axis=0)
                f['y_pred'][-arr_Y_pred.shape[0]:] = arr_Y_pred

                f['y_softmax'].resize((f['y_softmax'].shape[0] + arr_Y_softmax.shape[0]), axis=0)
                f['y_softmax'][-arr_Y_softmax.shape[0]:] = arr_Y_softmax


    
    print('[main.py] finished!')