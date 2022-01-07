import vs
import numpy as np

# main path, where is video existence 
PATH = 'C:\\Users\\Danya\\Desktop\\Диплом 0_0\\Video'  # <--- insert the path

# paths to all video files
videos_path = vs.Path(PATH).get_paths()

video_title = vs.Path.get_title_video(PATH)

data = np.zeros((len(videos_path), 5), dtype=np.float32)

for id, path in enumerate(videos_path):
    data[id, 0], data[id, 1], data[id, 2], data[id, 3] = \
        vs.Video(videos_path[id]).get_video_characteristics(verbose=False)

vs.Video.write2hdf5(data, video_title=video_title)

# path = Path(PATH)

#a = path.get_paths()


