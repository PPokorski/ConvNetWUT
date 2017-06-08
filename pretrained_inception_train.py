import numpy as np
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
K.set_image_dim_ordering('tf')
from keras.callbacks import ModelCheckpoint

BATCH_SIZE = 128
IMAGE_SIZE = 140
lr = 0.01
EPOCH_STEP = 7
EPOCHS_TOP_LAYER = 20
EPOCHS_BOTTOM_LAYER = 30

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

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(43, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

checkpointer_hi = ModelCheckpoint(filepath=DATASET_ROOT_PATH + "pretrained_inception/weights_hi_size140_fully_aug_.hdf5", save_best_only=True)

hi_hostory = model.fit_generator(train_generator,
            samples_per_epoch=train_generator.samples,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            nb_epoch=EPOCHS_TOP_LAYER,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE, callbacks=[checkpointer_hi])

np.savetxt("loss_history_hi_size14_fully_aug_0.txt", np.array(hi_hostory.history["loss"]), delimiter=",")
np.savetxt("acc_history_hi_size140_fully_aug_.txt", np.array(hi_hostory.history["acc"]), delimiter=",")



for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=lr, momentum=0.9, decay=1e-6, nesterov=True), loss='categorical_crossentropy', metrics=["accuracy"])

checkpointer_low = ModelCheckpoint(filepath=DATASET_ROOT_PATH + "pretrained_inception/weights_low_size__fully_aug_140.hdf5", save_best_only=True)
low_host = model.fit_generator(train_generator,
            samples_per_epoch=train_generator.samples,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=EPOCHS_BOTTOM_LAYER,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            callbacks=[checkpointer_low, LearningRateScheduler(lr_schedule)])

np.savetxt("loss_history_low_size_fully_aug_140.txt", np.array(low_host.history["loss"]), delimiter=",")
np.savetxt("acc_history_low_size_fully_aug__140.txt", np.array(low_host.history["acc"]), delimiter=",")

model.save_weights('inceptionV3_weights_pre_size_fully_aug__140.h5')