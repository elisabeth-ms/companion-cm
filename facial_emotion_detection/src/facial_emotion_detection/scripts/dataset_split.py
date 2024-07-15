import os
import shutil
import argparse
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class DatasetSplit():

    def dataset_reading(self, dataset_path):
        self.labels=[]
        self.emotions=[]
        self.file_paths = []
        l=0

        for root, dirs, files in os.walk(dataset_path):           
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]

            for fl in file_paths:
                self.file_paths.append(fl)
                self.labels.append(l)

            self.emotions.append(root.split(os.sep)[-1])
            l=l+1

        return self.file_paths, self.labels

    def dataset_split(self, labels, landmark_data_list):
        num_classes=8

        Y = keras.utils.to_categorical(labels, num_classes)
        x,y = shuffle(landmark_data_list,Y, random_state=2)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)

        return X_train, X_test, y_train, y_test, X_val, y_val
    
    def splitted_dataset_creation(self, X_train, X_test, X_val):

        for train_path in X_train:

            if not os.path.exists(f'{os.path.dirname(os.path.dirname(train_path))}/splitted_dataset/'):
                os.mkdir(f'{os.path.dirname(os.path.dirname(train_path))}/splitted_dataset/')
                os.mkdir(f'{os.path.dirname(os.path.dirname(train_path))}/splitted_dataset/train/')

            if not os.path.exists(f'{os.path.dirname(os.path.dirname(train_path))}/splitted_dataset/train/{train_path.split(os.sep)[-2]}/'):
                os.mkdir(f'{os.path.dirname(os.path.dirname(train_path))}/splitted_dataset/train/{train_path.split(os.sep)[-2]}/')

            destination_train_path = f'{os.path.dirname(os.path.dirname(train_path))}/splitted_dataset/train/{train_path.split(os.sep)[-2]}/'
            
            self.copy_and_move_file(train_path, destination_train_path)
            
        for test_path in X_test:    

            if not os.path.exists(f'{os.path.dirname(os.path.dirname(test_path))}/splitted_dataset/test/'):
                os.mkdir(f'{os.path.dirname(os.path.dirname(test_path))}/splitted_dataset/test/')

            if not os.path.exists(f'{os.path.dirname(os.path.dirname(test_path))}/splitted_dataset/test/{test_path.split(os.sep)[-2]}/'):
                os.mkdir(f'{os.path.dirname(os.path.dirname(test_path))}/splitted_dataset/test/{test_path.split(os.sep)[-2]}/')

            destination_test_path = f'{os.path.dirname(os.path.dirname(test_path))}/splitted_dataset/test/{test_path.split(os.sep)[-2]}/'
            
            self.copy_and_move_file(test_path, destination_test_path)

        for val_path in X_val:    

            if not os.path.exists(f'{os.path.dirname(os.path.dirname(val_path))}/splitted_dataset/val/'):
                os.mkdir(f'{os.path.dirname(os.path.dirname(val_path))}/splitted_dataset/val/')

            if not os.path.exists(f'{os.path.dirname(os.path.dirname(val_path))}/splitted_dataset/val/{val_path.split(os.sep)[-2]}/'):
                os.mkdir(f'{os.path.dirname(os.path.dirname(val_path))}/splitted_dataset/val/{val_path.split(os.sep)[-2]}/')

            destination_val_path = f'{os.path.dirname(os.path.dirname(val_path))}/splitted_dataset/val/{val_path.split(os.sep)[-2]}/'
            
            self.copy_and_move_file(val_path, destination_val_path)
            
    def copy_and_move_file(self, source_path, destination_path):
        try:
            # Copy the file to the destination
            shutil.copy2(source_path, destination_path)

            # Optional: If you want to remove the original file after copying
            # os.remove(source_path)

            print(f"File copied successfully from {source_path} to {destination_path}")
        except FileNotFoundError:
            print("Source file not found.")
        except PermissionError:
            print("Permission error. Make sure you have the necessary permissions.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program for training a tensorflow model that detects facial emotions.')
    # Add arguments
    parser.add_argument('-f', '--folder_images_path', type=str, help='Folder path of the dataset.')
    # Parse the command-line arguments
    args = parser.parse_args()
    # Access the values of the arguments
    folder_path = args.folder_images_path

    split = DatasetSplit()

    paths, labels = split.dataset_reading(folder_path)

    X_train, X_test, y_train, y_test, X_val, y_val = split.dataset_split(labels, paths)

    split.splitted_dataset_creation(X_train, X_test, X_val)