import fsspec
import h5py
import requests

def get_dataset(URL='https://cs282-datasets.s3.us-west-1.amazonaws.com/embeddings.h5'):
    remote_f = fsspec.open(URL, mode='rb')

    if hasattr(remote_f, 'open'):
        remote_f = remote_f.open()

    f = h5py.File(remote_f)
    dset = f['embeddings']

    # print("Keys: ", f.keys())
    #print("Shape: ", dset.shape)
    #print("First element: ", dset[0])
    return f, dset

def get_dataset_request_bytes(filename='embeddings.h5', bytes=1000, URL='https://cs282-datasets.s3.us-west-1.amazonaws.com/embeddings.h5'):
    headers = {"Range": "bytes=0-{}".format(bytes)}
    r = requests.get(URL, headers=headers, allow_redirects=True)
    open(filename, 'wb').write(r.content)