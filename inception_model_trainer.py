import numpy as np
import math as m
from keras import backend as K
from keras import utils
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping, LearningRateScheduler, CSVLogger
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('tf')
import image_loader as il

number_of_classes = 43
input_size = (224, 224)
input_shape = (224, 224, 3)
batch_size = 64

initial_lr = 0.001
momentum = 0.9
drop = 0.96
epochs_drop = 8.0

min_delta = 0.01
patience = 1


def learningRateDecay(epoch):
    lrate = initial_lr * m.pow(drop, m.floor((1 + epoch)/epochs_drop))
    return lrate


X_test, Y_test = il.loadTestingDataset('/home/pokor/Downloads/SNR/GTSRB_Test/Final_Test/Images')
X_test = np.array(X_test)
Y_test = utils.to_categorical(Y_test, number_of_classes)

print('Loaded test dataset!')

train_datagen = ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=False,
        featurewise_std_normalization=True,
        samplewise_std_normalization=False,
        zca_whitening=False,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='nearest')

train_datagen.fit(X_test)

print('Fitted data!')


train_generator = train_datagen.flow_from_directory(
    '/home/pokor/Downloads/SNR/GTSRB_Train/Final_Training/Images',
    target_size=input_size,
    batch_size=batch_size,
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    '/home/pokor/Downloads/SNR/GTSRB_Train/Final_Validation/Images',
    target_size=input_size,
    batch_size=batch_size,
    shuffle=True
)

print('Loaded train and test dataset the keras way!')

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(number_of_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

early_stopping_callback = EarlyStopping(monitor='val_accuracy',
                                        min_delta=min_delta,
                                        patience=patience,
                                        verbose=1,
                                        mode='auto')

learning_rate_callback = LearningRateScheduler(schedule=learningRateDecay)

csv_logger_callback = CSVLogger('log.csv', append=True, separator=';')

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=SGD(lr=initial_lr, momentum=momentum),
              loss='categorical_crossentropy', metrics=["accuracy"])

history = model.fit_generator(train_generator,
            samples_per_epoch=train_generator.samples,
            nb_epoch=20,
            validation_data=validation_generator,
            nb_val_samples=validation_generator.samples,
            callbacks=[early_stopping_callback,
                       learning_rate_callback,
                       csv_logger_callback])

model.save_weights('inceptionV3_weights.h5')

evaluation = model.evaluate(X_test, Y_test, batch_size=batch_size)

print(evaluation)