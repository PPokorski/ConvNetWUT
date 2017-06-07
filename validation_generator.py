import os
import random


def generate_validation(source_path, target_path, validation_split = 0.05):
    if(source_path[-1] != '/'):
        source_path.append('/')

    if(target_path[-1] != '/'):
        target_path.append('/')

    dirs = os.listdir(source_path)
    for dir in dirs:
        os.mkdir(target_path + dir, 0o777)

    for dir in dirs:
        source_dirs = os.listdir(source_path + '/' + dir)
        dirs_to_move = random.sample(range(len(source_dirs)), int(round(len(source_dirs) * validation_split)))

        for index in dirs_to_move:
            os.rename(source_path + dir + '/' + source_dirs[index],
                      target_path + dir + '/' + source_dirs[index])

generate_validation('/home/pokor/Downloads/SNR/GTSRB_Train/Final_Training/Images/',
                    '/home/pokor/Downloads/SNR/GTSRB_Train/Final_Validation/Images/')