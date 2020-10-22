import numpy as np
import os
import urllib.request as urllib
import gzip
import pickle as pickle


def mnist_generator(data, batch_size, n_labelled, limit=None, k =10 , add_noise = False, add_v =None):
    images, labels = data
    images_chunks = np.split(images.reshape((-1,784)),k)
    labels_chunks = np.split(labels,k)
    
    if add_noise:
        for j in range(k):
            images_chunks[j] = images_chunks[j] + add_v[j]

    def get_epoch():
        for i in range(k):
            rng_state = np.random.get_state()
            np.random.shuffle(images_chunks[i])
            np.random.set_state(rng_state)
            np.random.shuffle(labels_chunks[i])
        for i in range(int(len(images) / (k*batch_size))):
            yield [(images_chunks[j][i*batch_size:(i+1)*batch_size,:], labels_chunks[j][i*batch_size:(i+1)*batch_size]) for j in range(k)]

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None, k=10, add_noise = False, add_v = None):
    filepath = 'MNIST/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print ("Couldn't find MNIST dataset in /tmp, downloading...")
        urllib.urlretrieve(url, filepath)

    with gzip.open(filepath, 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f, encoding='latin1')

    return (
        mnist_generator(train_data, batch_size, n_labelled, k=k, add_noise = add_noise, add_v = add_v), 
        mnist_generator(dev_data, test_batch_size, n_labelled, k=1), 
        mnist_generator(test_data, test_batch_size, n_labelled, k=1)
    )