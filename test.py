import h5py

f = h5py.File('models/keras_model.h5', 'r')
print(f.attrs.get('keras_version'))