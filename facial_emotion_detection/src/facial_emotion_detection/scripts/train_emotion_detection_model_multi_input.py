import cv2
import numpy as np
import os
import argparse
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten, Attention, concatenate, BatchNormalization
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
from keras import Input, Model

class FacialEmotionModel():
    def __init__(self):
        pass

    def landmarks_data_extractor(self, dataset_path):
        # Eyes
        self.eyes_landmark_data_list_train = np.zeros(44)
        self.eyes_labels_train=[]
        self.emotions=[]
        self.paths_file = []
        
        l=0

        for root, dirs, files in os.walk(dataset_path + '/train/labels/eyes/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 44:
                        self.eyes_landmark_data_list_train = np.vstack([self.eyes_landmark_data_list_train, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.eyes_labels_train.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.eyes_landmark_data_list_train = self.eyes_landmark_data_list_train[1:]
        print(self.eyes_landmark_data_list_train.shape)

        self.eyes_landmark_data_list_test = np.zeros(44)
        self.eyes_labels_test=[]
        self.emotions=[]
        self.paths_file = []
        l=0

        for root, dirs, files in os.walk(dataset_path + '/test/labels/eyes/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 44:
                        self.eyes_landmark_data_list_test = np.vstack([self.eyes_landmark_data_list_test, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.eyes_labels_test.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.eyes_landmark_data_list_test = self.eyes_landmark_data_list_test[1:]
        print(self.eyes_landmark_data_list_test.shape)

        self.eyes_landmark_data_list_val = np.zeros(44)
        self.eyes_labels_val=[]
        self.emotions=[]
        self.paths_file = []
        l=0

        for root, dirs, files in os.walk(dataset_path + '/val/labels/eyes/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 44:
                        self.eyes_landmark_data_list_val = np.vstack([self.eyes_landmark_data_list_val, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.eyes_labels_val.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.eyes_landmark_data_list_val = self.eyes_landmark_data_list_val[1:]
        print(self.eyes_landmark_data_list_val.shape)

        # Nose
        self.nose_landmark_data_list_train = np.zeros(18)
        self.nose_labels_train=[]
        self.emotions=[]
        self.paths_file = []
        l=0

        for root, dirs, files in os.walk(dataset_path + '/train/labels/nose/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 18:
                        self.nose_landmark_data_list_train = np.vstack([self.nose_landmark_data_list_train, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.nose_labels_train.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.nose_landmark_data_list_train = self.nose_landmark_data_list_train[1:]
        print(self.nose_landmark_data_list_train.shape)

        self.nose_landmark_data_list_test = np.zeros(18)
        self.nose_labels_test=[]
        self.emotions=[]
        self.paths_file = []
        l=0

        for root, dirs, files in os.walk(dataset_path + '/test/labels/nose/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 18:
                        self.nose_landmark_data_list_test = np.vstack([self.nose_landmark_data_list_test, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.nose_labels_test.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.nose_landmark_data_list_test = self.nose_landmark_data_list_test[1:]
        print(self.nose_landmark_data_list_test.shape)

        self.nose_landmark_data_list_val = np.zeros(18)
        self.nose_labels_val=[]
        self.emotions=[]
        self.paths_file = []
        l=0

        for root, dirs, files in os.walk(dataset_path + '/val/labels/nose/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 18:
                        self.nose_landmark_data_list_val = np.vstack([self.nose_landmark_data_list_val, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.nose_labels_val.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.nose_landmark_data_list_val = self.nose_landmark_data_list_val[1:]
        print(self.nose_landmark_data_list_val.shape)

        # Mouth
        self.mouth_landmark_data_list_train = np.zeros(40)
        self.mouth_labels_train=[]
        self.emotions=[]
        self.paths_file = []
        l=0

        for root, dirs, files in os.walk(dataset_path + '/train/labels/mouth/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 40:
                        self.mouth_landmark_data_list_train = np.vstack([self.mouth_landmark_data_list_train, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.mouth_labels_train.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.mouth_landmark_data_list_train = self.mouth_landmark_data_list_train[1:]
        print(self.mouth_landmark_data_list_train.shape)

        self.mouth_landmark_data_list_test = np.zeros(40)
        self.mouth_labels_test=[]
        self.emotions=[]
        self.paths_file = []
        l=0

        for root, dirs, files in os.walk(dataset_path + '/test/labels/mouth/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 40:
                        self.mouth_landmark_data_list_test = np.vstack([self.mouth_landmark_data_list_test, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.mouth_labels_test.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.mouth_landmark_data_list_test = self.mouth_landmark_data_list_test[1:]
        print(self.mouth_landmark_data_list_test.shape)

        self.mouth_landmark_data_list_val = np.zeros(40)
        self.mouth_labels_val=[]
        self.emotions=[]
        self.paths_file = []
        l=0

        for root, dirs, files in os.walk(dataset_path + '/val/labels/mouth/'):
            if not root.split(os.sep)[-1]:
                continue
            file_paths = [os.path.join(root, f) for f in files]
            for fl in file_paths:
                with open(fl, 'r') as f:
                    content = np.loadtxt(f)
                    if content.size == 40:
                        self.mouth_landmark_data_list_val = np.vstack([self.mouth_landmark_data_list_val, content])
                        # self.total_landmark_data_list += [content]
                        self.paths_file.append(fl)
                        self.mouth_labels_val.append(l)
                    f.close()
            self.emotions.append(root.split(os.sep)[-1])
            l=l+1
        self.mouth_landmark_data_list_val = self.mouth_landmark_data_list_val[1:]
        print(self.mouth_landmark_data_list_val.shape)

        print(self.emotions)

        return self.eyes_labels_train, self.eyes_landmark_data_list_train, self.eyes_labels_test, self.eyes_landmark_data_list_test, self.eyes_labels_val, self.eyes_landmark_data_list_val, self.nose_labels_train, self.nose_landmark_data_list_train, self.nose_labels_test, self.nose_landmark_data_list_test, self.nose_labels_val, self.nose_landmark_data_list_val, self.mouth_labels_train, self.mouth_landmark_data_list_train, self.mouth_labels_test, self.mouth_landmark_data_list_test, self.mouth_labels_val, self.mouth_landmark_data_list_val

    def dataset_preparation(self, labels_train, labels_test, labels_val, landmark_data_list_train, landmark_data_list_test, landmark_data_list_val):
        num_classes=8

        y_train = keras.utils.to_categorical(labels_train, num_classes)
        y_test = keras.utils.to_categorical(labels_test, num_classes)
        y_val = keras.utils.to_categorical(labels_val, num_classes)
        # x,y = shuffle(landmark_data_list,Y, random_state=2)
        # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

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
        # define two sets of inputs
        input_eyes = Input(shape=(44,))
        input_nose = Input(shape=(18,))
        input_mouth = Input(shape=(40,))

        hidden_layer_1_size = 100
        hidden_layer_2_size = 100
        hidden_layer_3_size = 500
        output_size = 8

        # the first branch operates on the first input
        x = Dense(hidden_layer_1_size, activation="relu")(input_eyes)
        # x = BatchNormalization()(x)
        # x = Dropout(0.4)(x)
        x = Dense(hidden_layer_2_size, activation="relu")(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.4)(x)
        x = Dense(hidden_layer_3_size, activation="relu")(x)
        # x = BatchNormalization()(x)
        # x = Dropout(0.4)(x)
        x = Dense(output_size, activation="relu")(x)
        x = Model(inputs=input_eyes, outputs=x)

        # the second branch operates on the second input
        y = Dense(hidden_layer_1_size, activation="relu")(input_nose)
        # y = BatchNormalization()(y)
        # y = Dropout(0.4)(y)
        y = Dense(hidden_layer_2_size, activation="relu")(y)
        # y = BatchNormalization()(y)
        # y = Dropout(0.4)(y)
        y = Dense(hidden_layer_3_size, activation="relu")(y)
        # y = BatchNormalization()(y)
        # y = Dropout(0.4)(y)
        y = Dense(output_size, activation="relu")(y)
        y = Model(inputs=input_nose, outputs=y)

        # the third branch operates on the third input
        z = Dense(hidden_layer_1_size, activation="relu")(input_mouth)
        # z = BatchNormalization()(z)
        # z = Dropout(0.4)(z)
        z = Dense(hidden_layer_2_size, activation="relu")(z)
        # z = BatchNormalization()(z)
        # z = Dropout(0.4)(z)
        z = Dense(hidden_layer_3_size, activation="relu")(z)
        # z = BatchNormalization()(z)
        # z = Dropout(0.4)(z)
        z = Dense(output_size, activation="relu")(z)
        z = Model(inputs=input_mouth, outputs=z)

        # combine the output of the three branches
        combined = concatenate([x.output, y.output, z.output])

        # apply a FC layer and then a regression prediction on the
        # combined outputs
        h = Dense(hidden_layer_1_size, activation="relu")(combined)
        h = BatchNormalization()(h)
        h = Dropout(0.2)(h)
        h = Dense(hidden_layer_2_size, activation="relu")(h)
        h = BatchNormalization()(h)
        h = Dropout(0.2)(h)
        h = Dense(hidden_layer_3_size, activation="relu")(h)
        h = BatchNormalization()(h)
        h = Dropout(0.2)(h)
        h = Dense(output_size, activation="softmax")(h)


        # the model will accept the inputs of the three branches and
        # then output a single value
        model = Model(inputs=[x.input, y.input, z.input], outputs=h)

        # model = Sequential()
        # # model.add(Flatten())
        # model.add(Dense(hidden_layer_1_size, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(hidden_layer_2_size, activation='relu'))
        # model.add(Dropout(0.2))
        # model.add(Dense(hidden_layer_3_size, activation='relu'))
        # model.add(Dropout(0.2))
        
        # model.add(Dense(output_size, activation='softmax'))

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

        return model
    
    def train_model(self, checkpoint_path, model):
        # checkpoint_path = "/content/drive/MyDrive/phd_result/best.ckpt"
        # checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1)

        epochs = 1000
        batch_size = 512
        # history = model.fit([X_train_eyes, X_train_nose, X_train_mouth], y_train, batch_size=batch_size, epochs=epochs, callbacks=[cp_callback], verbose=1)
        history = model.fit([X_train_eyes, X_train_nose, X_train_mouth], y_train, batch_size=batch_size, epochs=epochs, validation_data=([X_val_eyes, X_val_nose, X_val_mouth], y_val), callbacks=[cp_callback], verbose=1)

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

    eyes_labels_train, eyes_landmark_data_list_train, eyes_labels_test, eyes_landmark_data_list_test, eyes_labels_val, eyes_landmark_data_list_val, nose_labels_train, nose_landmark_data_list_train, nose_labels_test, nose_landmark_data_list_test, nose_labels_val, nose_landmark_data_list_val, mouth_labels_train, mouth_landmark_data_list_train, mouth_labels_test, mouth_landmark_data_list_test, mouth_labels_val, mouth_landmark_data_list_val = model.landmarks_data_extractor(folder_path)

    X_train_eyes, X_test_eyes, X_val_eyes, y_train, y_test, y_val = model.dataset_preparation(eyes_labels_train, eyes_labels_test, eyes_labels_val, eyes_landmark_data_list_train, eyes_landmark_data_list_test, eyes_landmark_data_list_val)

    X_train_nose, X_test_nose, X_val_nose, y_train, y_test, y_val = model.dataset_preparation(nose_labels_train, nose_labels_test, nose_labels_val, nose_landmark_data_list_train, nose_landmark_data_list_test, nose_landmark_data_list_val)

    X_train_mouth, X_test_mouth, X_val_mouth, y_train, y_test, y_val = model.dataset_preparation(mouth_labels_train, mouth_labels_test, mouth_labels_val, mouth_landmark_data_list_train, mouth_landmark_data_list_test, mouth_landmark_data_list_val)

    multi_mlp_model = model.create_model()

    multi_mlp_model.build([X_train_eyes.shape, X_train_nose.shape, X_train_mouth.shape])

    multi_mlp_model.summary()

    # plot_model(multi_mlp_model, show_layer_names=False, show_shapes=True, to_file='model_multi_input_mlp.png')

    checkpoint_path = "/home/oem/companion_ws/src/facial_emotion_detection/data/best1.ckpt"

    training_history = model.train_model(checkpoint_path, multi_mlp_model)

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

    multi_mlp_model.load_weights('/home/oem/companion_ws/src/facial_emotion_detection/data/best1.ckpt')

    loss, acc = multi_mlp_model.evaluate([X_test_eyes, X_test_nose, X_test_mouth], y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    multi_mlp_model.save("/home/oem/companion_ws/src/facial_emotion_detection/data/model_multi_input_landmarks.h5")


