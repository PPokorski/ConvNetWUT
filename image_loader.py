import csv
import numpy as np
from keras.preprocessing import image

def loadTestingDataset(testing_set_root_path):
    images = []
    labels = []
    testing_set_csv_path = testing_set_root_path + '/GT-final_test.csv'
    with open(testing_set_csv_path) as result_f:
        csv_reader = csv.reader(result_f, delimiter=';')
        next(csv_reader)
        for row in csv_reader:
            images.append(np.array(image.load_img(testing_set_root_path + '/' + row[0], target_size=(150, 150, 3))))
            labels.append(row[7])
    return images, labels

def loadTrainingDataset(rootpath):
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            try:
                images.append(np.array(image.load_img(prefix + row[0], target_size=(150, 150, 3)))) # the 1th column is the filename
            except FileNotFoundError as e:
                pass

            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return images, labels