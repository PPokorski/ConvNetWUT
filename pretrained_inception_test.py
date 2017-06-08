import image_loader as il
import numpy as np
import tflearn
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('tf')

TEST_BATCH_SIZE = 64
IMAGE_SIZE = 140

test_datagen = ImageDataGenerator(rescale=1./255)

X_test, Y_test = il.loadTestingDataset2('Final_Test/Images')
X_test = np.array(X_test)
Y_test = tflearn.data_utils.to_categorical(Y_test, 43)

test_datagen.fit(X_test)

base_model = InceptionV3(weights=None, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), classes=43, include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(43, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
model.load_weights('inceptionV3_weights_pre_size_fully_aug__140.h5')

test_generator = test_datagen.flow(X_test, Y_test, batch_size=TEST_BATCH_SIZE)
print (model.evaluate_generator(test_generator, Y_test.shape[0] // TEST_BATCH_SIZE))
