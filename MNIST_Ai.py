import numpy as np
import struct
import pymindcorecupy as pm
import gzip
import idx2numpy



def read_idx(filename):
    try:
        with gzip.open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    except:
        ndarr = idx2numpy.convert_from_file(filename)
        if len(ndarr.shape)==3:
            ndarr=ndarr.reshape(10000,784)
        return ndarr
    



def one_hot_encode(labels, num_classes):
    # Create an array of zeros with shape (len(labels), num_classes)
    one_hot_labels = np.zeros((len(labels), num_classes))
    
    # Set the appropriate element to one
    one_hot_labels[np.arange(len(labels)), labels] = 1
    
    return one_hot_labels

# One-hot encode the train labels

# Read the images from the IDX file
train_images = read_idx('t10k-images.idx3-ubyte')  # Update this path to the path to your MNIST images installation
train_labels = read_idx('t10k-labels.idx1-ubyte')  # Update this path to the path to your MNIST labels installation

Y_train_one_hot = one_hot_encode(train_labels, 10)
# Flatten the 28x28 pixel images into 1D arrays of 784 features each
X_train_flattened = train_images.reshape(train_images.shape[0], -1)

# Normalize the grayscale images by dividing by 255 to get values between 0 and 1
X_train_normalized = X_train_flattened / 255.0

inputs=784
outputs=10
middle=[16,16]
print("About to train")
nn = pm.Optomised_neural_network(input_neurons=inputs, hidden_layers=middle, output_neurons=outputs, activation="sig")
optomizer="ADAM"
nn.train(X_train_normalized, Y_train_one_hot, epochs=1000, learning_rate=0.01,optimizer=optomizer)

print("Trained on 1000 epochs")

