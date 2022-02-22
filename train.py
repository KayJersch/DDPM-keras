from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from model import build_unet
from callback import imageCallback
from data_generator import data_generator

batch_size = 32
noising_steps = 100

unet = build_unet(noising_steps)
unet.summary()

callbacks = [
    imageCallback(unet,noising_steps,5),
    ModelCheckpoint(filepath='saved/model-{epoch}.h5'),
    TensorBoard(log_dir='./logs'),
]

unet.fit(
    data_generator(batch_size,noising_steps),
    epochs=1000,
    callbacks=callbacks,
    steps_per_epoch=int(6000000/batch_size)
)