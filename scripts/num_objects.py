import numpy as np


# path = '../trainvalRaw.mat'
path = '../trainval.mat'
camera_num = 2


if 'Raw' in path:
    import h5py
    file = h5py.File(path, 'r')
    data = file['trainData']
else:
    import scipy.io
    file = scipy.io.loadmat(path)
    data = file['trainData'].T
    # data = np.asarray(data)
    print(list(file.keys()))
    print(data)

print(data.shape)

camera_idx = np.equal(data[0], camera_num)
frame, left, top, width, height = data[2:7, camera_idx]

print(data.shape)
print(frame.shape)


def foo(a, b):
    """

    :param a:
    :param b:
    :return:
    """