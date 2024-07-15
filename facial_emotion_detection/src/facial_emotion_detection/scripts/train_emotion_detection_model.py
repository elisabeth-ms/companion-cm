import cv2
import numpy as np
import os
import argparse
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten, Attention
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
# from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.datasets import make_classification
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

class FacialEmotionModel():
    def __init__(self):
        pass

    def landmarks_data_extractor(self, dataset_path):

        self.total_landmark_data_list_train = np.zeros(136)
        self.labels_train=[]
        self.emotions=[]
        self.paths_file = []
        l=0

        for root, dirs, files in os.walk(dataset_path + '/train/labels/total/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 136:
                        self.total_landmark_data_list_train = np.vstack([self.total_landmark_data_list_train, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.labels_train.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.total_landmark_data_list_train = self.total_landmark_data_list_train[1:]
        print(self.total_landmark_data_list_train.shape)

        self.total_landmark_data_list_test = np.zeros(136)
        self.labels_test=[]
        self.emotions=[]
        self.paths_file = []
        l=0

        for root, dirs, files in os.walk(dataset_path + '/test/labels/total/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 136:
                        self.total_landmark_data_list_test = np.vstack([self.total_landmark_data_list_test, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.labels_test.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.total_landmark_data_list_test = self.total_landmark_data_list_test[1:]
        print(self.total_landmark_data_list_test.shape)

        self.total_landmark_data_list_val = np.zeros(136)
        self.labels_val=[]
        self.emotions=[]
        self.paths_file = []
        l=0

        for root, dirs, files in os.walk(dataset_path + '/val/labels/total/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 136:
                        self.total_landmark_data_list_val = np.vstack([self.total_landmark_data_list_val, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.labels_val.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.total_landmark_data_list_val = self.total_landmark_data_list_val[1:]
        print(self.total_landmark_data_list_val.shape)

        print(self.emotions)

        return self.labels_train, self.total_landmark_data_list_train, self.labels_test, self.total_landmark_data_list_test, self.labels_val, self.total_landmark_data_list_val

    def dataset_preparation(self, labels_train, labels_test, labels_val, landmark_data_list_train, landmark_data_list_test, landmark_data_list_val):
        num_classes=8

        # Y = keras.utils.to_categorical(labels, num_classes)
        # x,y = shuffle(landmark_data_list,Y, random_state=2)
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        y_train = keras.utils.to_categorical(labels_train, num_classes)
        y_test = keras.utils.to_categorical(labels_test, num_classes)
        y_val = keras.utils.to_categorical(labels_val, num_classes)

        X_train = landmark_data_list_train
        X_test = landmark_data_list_test
        X_val = landmark_data_list_val

        print('Train dimension:');print(X_train.shape)
        print('Test dimension:');print(X_test.shape)
        print('Val dimension:');print(X_val.shape)
        print('Train labels dimension:');print(y_train.shape)
        print('Test labels dimension:');print(y_test.shape)
        print('Val labels dimension:');print(y_val.shape)

        return X_train, X_test, X_val, y_train, y_test, y_val

    def create_model(self):
        hidden_layer_1_size = 100
        hidden_layer_2_size = 100
        hidden_layer_3_size = 500
        output_size = 8

        model = Sequential()
        # model.add(Flatten())
        model.add(Dense(hidden_layer_1_size, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(hidden_layer_2_size, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(hidden_layer_3_size, activation='relu'))
        # model.add(Dropout(0.2))
        
        model.add(Dense(output_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='RMSprop')

        return model
    
    def train_model(self, checkpoint_path, model):
        # checkpoint_path = "/content/drive/MyDrive/phd_result/best.ckpt"
        # checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1)

        epochs = 1000
        batch_size = 512
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[cp_callback], verbose=1)

        return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program for training a tensorflow model that detects facial emotions.')
    # Add arguments
    parser.add_argument('-f', '--folder_images_path', type=str, help='Folder path of the dataset.')
    # Parse the command-line arguments
    args = parser.parse_args()
    # Access the values of the arguments
    folder_path = args.folder_images_path

    model = FacialEmotionModel()

    labels_train, landmark_data_list_train, labels_test, landmark_data_list_test, labels_val, landmark_data_list_val = model.landmarks_data_extractor(folder_path)

    X_train, X_test, X_val, y_train, y_test, y_val = model.dataset_preparation(labels_train, labels_test, labels_val, landmark_data_list_train, landmark_data_list_test, landmark_data_list_val)

    mlp_model = model.create_model()

    mlp_model.build(X_train.shape)

    mlp_model.summary()

    # plot_model(mlp_model, show_layer_names=False, show_shapes=True, to_file='model_total.png')

    checkpoint_path = "/home/oem/companion_ws/src/facial_emotion_detection/data/best.ckpt"

    training_history = model.train_model(checkpoint_path, mlp_model)

    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    mlp_model.load_weights('/home/oem/companion_ws/src/facial_emotion_detection/data/best.ckpt')

    loss, acc = mlp_model.evaluate(X_test, y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    mlp_model.save("/home/oem/companion_ws/src/facial_emotion_detection/data/model_total_landmarks.h5")


