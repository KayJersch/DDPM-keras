import numpy as np

from skimage.io import imsave
from tensorflow.keras.callbacks import Callback

class imageCallback(Callback):
    def __init__(self,model,steps,grid_size):
        self.model = model
        self.steps = steps
        self.grid_size = grid_size

    def on_epoch_end(self, epoch, logs=None):
        images = np.random.normal(size=(self.grid_size**2,28,28,1))

        for i in range(self.steps):
            step = np.zeros((self.grid_size**2,self.steps))
            step[:,i] = 1
            images = self.model.predict([images,step])
        
        rows = []
        for i in range(self.grid_size):
            row = images[i*self.grid_size:i*self.grid_size+self.grid_size]
            row = np.hstack(row)
            rows.append(row)
        
        grid = np.vstack(rows)

        imsave('images/{}.png'.format(epoch),grid)