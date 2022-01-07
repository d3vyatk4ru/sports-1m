import os
import h5py as h5

name = ['_1SiMrs57Qo.mp4', '6yd28WCrhkQ.mp4', '7u1YDqG-2R0.mp4', '9bmcxNesqlI.mp4', '516zgzgVSYk.mp4', 'bQe6Gnmcsjo.mp4', 
        'BqkssPK3xK0.mp4', 'dfB8MRsww5g.mp4', 'EFFXNLmTVs4.mp4', 'FS4hDDs9azg.mp4', 'FZ_PQ39XsGo.mp4', 'GgEW4vtLC4A.mp4', 
       'hMhBbmE92v8.mp4', 'HMyXdPEAeuk.mp4', 'L7PXAEZI_T0.mp4', 'PY66VX5B9BI.mp4', 'QiSobhJagSM.mp4', 'U09V_peHzPw.mp4', 
       'yT_yyQqHr-k.mp4', 'zm_QusYMG1A.mp4']
duration = [77.844437, 318.694305, 231.666672, 321.100006, 21.167999, 449.338989, 37.799999, 88.800003, 866.76001,
           50.016632, 72.959999, 48.281612, 197.199997, 139.500000, 394.833008, 87.086998, 37.799999, 337.839996, 
           764.679993, 37.799999]
fps = [29.970030, 29.959745, 30.00000, 30.000000, 30.00000, 15.050997, 20.000000, 25.000000, 25.000000, 29.970030, 
      25.00000, 29.97000, 30.00000, 10.000000, 30.00000, 29.970030, 30.0, 25.00000, 25.000000, 25.00000]
frame = [2333.0, 9548.0, 6950.0, 9633.0, 636.0, 6763.0, 756.0, 2220.0, 21669.0, 1499.0, 824.0, 1447.0, 5916.0, 1395.0, 
        11845.0, 2610.0, 1134.0, 8446.0, 19117.0, 945.0]
height = [480.0, 480.0, 480.0, 480.0, 480.0, 480.0, 480.0, 480.0, 480.0, 480.0, 480.0, 480.0, 480.0, 480.0, 480.0, 480.0,
          480.0, 480.0, 480.0, 480.0]
width = [640.0, 640.0, 640.0, 640.0, 640.0, 640.0, 640.0, 640.0, 640.0, 640.0, 640.0, 640.0, 640.0, 640.0, 640.0, 640.0,
         640.0, 640.0, 640.0, 640]

with h5.File('C:\\Users\\Danya\\Desktop\\Диплом 0_0\\Video\\test_for_arch.hdf5', 'w') as file:
    file.create_dataset('name', data=name)

    file.create_dataset('duration', data=duration)

    file.create_dataset('fps', data=fps)
    file.create_dataset('frame', data=frame)
    file.create_dataset('height', data=height)
    file.create_dataset('width', data=width)