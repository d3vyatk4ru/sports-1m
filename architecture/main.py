import arch
from tensorflow.keras import optimizers, callbacks
from Generator import DataGenerator

# путь к файлу hdf5 с сэмплами
train_directory = '/dcache/sports1m/sports1m_tr_big.hdf5'
test_directory = '/dcache/sports1m/sports1m_ts_big.hdf5'

train_generator = DataGenerator(80, shuffle=True, directory=train_directory, Xkey='X_TR', Ykey='Y_TR')
test_generator = DataGenerator(80, shuffle=True, directory=test_directory, Xkey='X_TS', Ykey='Y_TS')

# создание модели
model = arch.create_model()

model.compile(optimizer=optimizers.Adam(learning_rate=3e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

# paths for save callbacks
path_checkpoint ='/data/d3vyatk4ru/architecture/ARCH[V0]/checkpoint/{epoch:02d}-{val_loss:.2f}.hdf5'
path_weight = '/data/d3vyatk4ru/architecture/ARCH[V0]/weight/{epoch:02d}_weight.hdf5'

# callbacks

# init f1 mrtric for validation data (work like callback)
f1 = arch.F1(test_generator)

# arch.gpu_status()

cb = [callbacks.ModelCheckpoint(save_best_only=True, filepath=path_weight),
      callbacks.ModelCheckpoint(filepath=path_checkpoint),
      f1]

history = model.fit(x=train_generator, 
                    epochs=20, verbose=1, workers=6, 
                    use_multiprocessing=True)

model.save('/data/d3vyatk4ru/architecture/ARCH[V0]/saved_model/ARCH0.hdf5')