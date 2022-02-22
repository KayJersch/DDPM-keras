import numpy as np

from tensorflow.keras.datasets.mnist import load_data

def data_generator(batch_size,max_steps):
    (x_train, _), _ = load_data()

    while True:
        x_imgs_batch = []
        x_step_batch = []
        y_batch = []
        for b in range(batch_size):
            img = x_train[np.random.randint(x_train.shape[0])]
            img = img.reshape((28,28,1))
            img = img/(255/2)
            img -= 1

            steps = np.random.randint(max_steps-1)
            step_onehot = np.zeros(max_steps)
            step_onehot[steps] = 1
            
            noise = np.random.normal(size=img.shape)
            noise = noise/max(np.max(noise),-np.min(noise))

            x = np.average(
                (noise,img),
                axis=0,
                weights=[(steps+1)/max_steps,1-((steps+1)/max_steps)]
            )
            
            noise = np.random.normal(size=img.shape)
            noise = noise/max(np.max(noise),-np.min(noise))

            y = np.average(
                (noise,img),
                axis=0,
                weights=[steps/max_steps,1-(steps/max_steps)]
            )

            x_imgs_batch.append(x)
            x_step_batch.append(step_onehot)
            y_batch.append(y)
        
        x_imgs_batch = np.array(x_imgs_batch)
        x_step_batch = np.array(x_step_batch)
        y_batch = np.array(y_batch)

        yield [x_imgs_batch, x_step_batch], y_batch