import pickle
import numpy as np

train_data = np.zeros((50000, 32, 32, 3), dtype='int')
test_data = np.zeros((10000, 32, 32, 3), dtype='int')


def load_batch(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        imgs = dict[b'data']
        return np.concatenate([imgs[:, :1024].reshape(-1, 32, 32, 1),
                               imgs[:, 1024:2048].reshape(-1, 32, 32, 1),
                               imgs[:, 2048:].reshape(-1, 32, 32, 1)], axis=-1)


for i in range(5):
    train_data[i * 10000:(i + 1) * 10000] = load_batch(f'./cifar-10-batches-py/data_batch_{i + 1}')

test_data[:] = load_batch(f'./cifar-10-batches-py/test_batch')

# Normalise
train_data = train_data.astype('float32') / 255.
test_data = test_data.astype('float32') / 255.

np.savez('./cifar10.npz', train_data=train_data, test_data=test_data)
