# %%
import sys
import matplotlib.pyplot as plt
import os
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(
            len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels


# %%
X_train, y_train = load_mnist('/tf/minst/', kind='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('/tf/minst/', kind='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))
# %%

fig, ax = plt.subplots(nrows=2, ncols=5,
                       sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

#%%
np.savez_compressed('/tf/minst/mnist_scaled.npz',
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test)

# %%
mnist = np.load('./minst/mnist_scaled.npz')

# %%
X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]

print('Rows: %d, columns: %d' % (len(X_train), len(y_train)))
print('Rows: %d, columns: %d' % (len(X_test), len(y_test)))
#%%
from neuralnet import NeuralNetMLP

nn = NeuralNetMLP(n_hidden=100,
                  l2=0.01,
                  epochs=200,
                  eta=0.0005,
                  minibatch_size=100,
                  shuffle=True,
                  seed=1)
nn.fit(X_train=X_train[:55000],
       y_train=y_train[:55000],
       X_valid=X_train[55000:],
       y_valid=y_train[55000:])

# %%
import matplotlib.pyplot as plt
plt.plot(range(nn.epochs), nn.eval_['cost'])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

#%%
plt.plot(range(nn.epochs), nn.eval_['train_acc'],
         label='training')
plt.plot(range(nn.epochs), nn.eval_['valid_acc'],
         label='validation', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

#%%
y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred)
      .astype(np.float) / X_test.shape[0])
print('Training accuracy: %.2f%%' % (acc * 100))

#%%
