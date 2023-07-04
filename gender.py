import glob
import os
import cv2
import keras.models
import numpy as np
from keras.regularizers import Regularizer, l2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt


class GenderCNN:
    def __init__(self, batch_size = 32, epochs = 10, lr= 0.001, img_dims = (64,64,3), image_dims2D = (64,64), regularizerValue = 0.0001, modelNumber = -1):
        self.class_dict = {}
        self.images = []
        self.labels = []
        self.val_images = []
        self.val_labels = []
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.model = None
        self.num_classes = 2
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.img_dims = img_dims
        self.image_dims2D = image_dims2D
        self.regularizerValue = regularizerValue
        self.modelNumber = modelNumber

    def load_images(self):
        image_files = [f for f in glob.glob(r'/Users/nikolasehovac/Desktop/ORI/genderCNN/data/training/'
                                            + "/**/*", recursive=True) if not os.path.isdir(f)]
        np.random.shuffle(image_files)
        for img in image_files:

            image = cv2.imread(img)

            image = cv2.resize(image, self.image_dims2D)
            self.images.append(image)
            label = img.split(os.path.sep)[-2]

            if label == "male":
                label = 0
            else:
                label = 1
            self.labels.append([label])

        self.train_images = np.array(self.images) / 255.0
        self.train_labels = np.array(self.labels)

        image_files_validation = [f for f in glob.glob(r'/Users/nikolasehovac/Desktop/ORI/genderCNN/data/validation/'
                                            + "/**/*", recursive=True) if not os.path.isdir(f)]

        np.random.shuffle(image_files_validation)
        for img in image_files_validation:

            image = cv2.imread(img)

            image = cv2.resize(image, self.image_dims2D)
            self.val_images.append(image)
            label = img.split(os.path.sep)[-2]

            if label == "male":
                label = 0
            else:
                label = 1
            self.val_labels.append([label])

        self.val_images = np.array(self.val_images) / 255.0
        self.val_labels = np.array(self.val_labels)

        image_files_test = [f for f in glob.glob(r'/Users/nikolasehovac/Desktop/ORI/genderCNN/data/test/'
                                            + "/**/*", recursive=True) if not os.path.isdir(f)]

        np.random.shuffle(image_files_test)

        for img in image_files_test:

            image = cv2.imread(img)
            image = cv2.resize(image, self.image_dims2D)
            self.test_images.append(image)
            label = img.split(os.path.sep)[-2]

            if label == "male":
                label = 0
            else:
                label = 1
            self.test_labels.append([label])

        self.test_images = np.array(self.test_images) / 255.0
        self.test_labels = np.array(self.test_labels)


        self.test_labels = to_categorical(self.test_labels, self.num_classes)
        self.val_labels = to_categorical(self.val_labels, self.num_classes)
        self.train_labels = to_categorical(self.train_labels, self.num_classes)


        print("Number of train images:", len(self.train_images))
        print("Number of train labels:", len(self.train_labels))
        print("Number of validation images:", len(self.val_images))
        print("Number of validation labels:", len(self.val_labels))
        print("Number of test images:", len(self.test_images))
        print("Number of test labels:", len(self.test_labels))
        print("Number of classes:", self.num_classes)

    def split_data(self):
        train_images, val_images, train_labels, val_labels = train_test_split(
            self.images, self.labels, test_size=0.2, random_state=42
        )

        train_labels = to_categorical(train_labels, self.num_classes)
        val_labels = to_categorical(val_labels, self.num_classes)

        self.train_images = train_images / 255.0
        self.val_images = val_images / 255.0
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = to_categorical(self.test_labels, self.num_classes)

    def build_model1(self):
        self.modelNumber = 1
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.img_dims, kernel_regularizer=l2(self.regularizerValue)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def build_model2(self):
        self.modelNumber = 2
        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), activation='relu', input_shape=self.img_dims, kernel_regularizer=l2(self.regularizerValue)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))

    def build_model3(self):
        self.modelNumber = 3
        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), activation='relu', input_shape=self.img_dims, kernel_regularizer=l2(self.regularizerValue)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))


    def train_model(self):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        print("seli")
        datagen.fit(self.train_images)

        opt = Adam(learning_rate=0.0001)
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        h = self.model.fit(
            datagen.flow(self.train_images, self.train_labels, batch_size=self.batch_size),
            epochs=self.epochs,
            validation_data=(self.val_images, self.val_labels),
        )

        self.plot_training(h)
    def evaluate_model(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_images,
                                                       self.test_labels)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)

    def save_model(self):
        self.model.save(f'genderCNN{self.modelNumber}-lr_{self.lr}-bs_{self.batch_size}-ep_{self.epochs}.h5')

    def load_model(self, modelNumber):
        self.model = keras.models.load_model(f'genderCNN{modelNumber}-lr_{self.lr}-bs_{self.batch_size}-ep_{self.epochs}.h5')
        if self.model is None:
            return False
        return True

    def predict_gender(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.image_dims2D)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        prediction = self.model.predict(img)
        print(prediction)
        predicted_class = np.argmax(prediction)

        if predicted_class == 0:
            return 'male'
        elif predicted_class == 1:
            return 'female'

        return "Unknown"

    def plot_training(self, history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'genderCNN{self.modelNumber}-lr_{self.lr}-bs_{self.batch_size}-ep_{self.epochs}.png')
        plt.draw()
