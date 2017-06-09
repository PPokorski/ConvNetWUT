import numpy as np
import tflearn
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('tf')
import image_loader as il
from keras.optimizers import SGD
from keras.layers import Input

test_datagen = ImageDataGenerator(rescale=1./255)

IMAGE_SIZE = 140
TEST_BATCH_SIZE = 64

X_test, Y_test = il.loadTestingDataset2('Final_Test/Images')
X_test = np.array(X_test)
Y_test = tflearn.data_utils.to_categorical(Y_test, 43)

test_datagen.fit(X_test)

base_model = InceptionV3(weights=None, input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)), classes=43, include_top=True)

base_model.load_weights('base_full_inceptionV3_weights.h5')
base_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

test_generator = test_datagen.flow(X_test, Y_test, batch_size=TEST_BATCH_SIZE)

print (base_model.evaluate_generator(test_generator, Y_test.shape[0] // TEST_BATCH_SIZE))
