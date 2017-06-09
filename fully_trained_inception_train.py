import numpy as np
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('tf')
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.layers import Input

BATCH_SIZE = 128
IMAGE_SIZE = 140
lr = 0.01
EPOCH_STEP = 20
EPOCHS = 120

DATASET_ROOT_PATH='/'

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / EPOCH_STEP))

train_datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=False,
        samplewise_center=False,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        horizontal_flip=True,
        vertical_flip=True,
fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    DATASET_ROOT_PATH + 'Final_Training/Images',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_ROOT_PATH + 'Final_Validation/Images',
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

base_model = InceptionV3(weights=None, input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)), classes=43, include_top=True)

checkpointer = ModelCheckpoint(filepath=DATASET_ROOT_PATH + "full_trained/weights_size140full.hdf5", save_best_only=True)

base_model.compile(optimizer=SGD(lr=lr, momentum=0.9, decay=1e-6, nesterov=True), loss='categorical_crossentropy', metrics=["accuracy"])


cost = base_model.fit_generator(train_generator,
            samples_per_epoch=train_generator.samples,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            nb_epoch=EPOCHS,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE, callbacks=[checkpointer, LearningRateScheduler(lr_schedule)])

np.savetxt("loss_history_fully_trained_all_140.txt", np.array(cost.history["loss"]), delimiter=",")
np.savetxt("acc_history_fully__trained_all_140.txt", np.array(cost.history["acc"]), delimiter=",")

base_model.save_weights('base_full_inceptionV3_weights.h5')