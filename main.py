# This is a sample Python script.

import pandas as pd
import numpy as np
import keras
import struct
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def read_idx(filename):
    """Function to read IDX files."""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


path_to_images = "./train-images.idx3-ubyte"
path_to_labels = "./train-labels.idx1-ubyte"

images = read_idx(path_to_images)
labels = read_idx(path_to_labels)

# Reshape images to 2D arrays (one image per row)
num_images = images.shape[0]
image_size = images.shape[1] * images.shape[2]
images = images.reshape(num_images, image_size)

# Convert to pandas DataFrames
df_images = pd.DataFrame(images)
df_labels = pd.DataFrame(labels, columns=['label'])

X_train,X_tmp,y_train, y_tmp = train_test_split(df_images.values, df_labels['label'].values, test_size=0.2, random_state=42)




X_valid, X_test, y_valid, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_valid = X_valid.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_valid = keras.utils.to_categorical(y_valid, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Build the model
model = keras.Sequential([
    keras.layers.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Specify the learning rate for the Adam optimizer
learning_rate = 0.0001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model with the specified learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid), callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Plot learning curves
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()





































































### Diabites example
# df = pd.read_csv("diabetes.csv")
# print(df.describe())

# for label in df.columns[:-1]:
#     plt.hist(df[df['Outcome'] == 1][label], color='blue', label='Diabetes', alpha=0.6, density=True, bins=15)
#     plt.hist(df[df['Outcome'] == 0][label], color='red', label='No diabetes', alpha=0.6, density=True, bins=15)
#     plt.title(label)
#     plt.ylabel('Probability')
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()

# x = df[df.columns[:-1]].values
# y = df['Outcome'].values
#
# scaler = StandardScaler()
#
# x = scaler.fit_transform(X=x)
#
# data = np.hstack((x, np.reshape(y, (-1, 1))))
#
# new_df = pd.DataFrame(data, columns=df.columns)
#
# # for label in new_df.columns[:-1]:
# #     plt.hist(new_df[new_df['Outcome'] == 1][label], color='blue', label='Diabetes', alpha=0.6, density=True, bins=15)
# #     plt.hist(new_df[new_df['Outcome'] == 0][label], color='red', label='No diabetes', alpha=0.6, density=True, bins=15)
# #     plt.title(label)
# #     plt.ylabel('Probability')
# #     plt.xlabel(label)
# #     plt.legend()
# #     plt.show()
#
# over = RandomOverSampler()
# print(len(x),len(y))
# x, y = over.fit_resample(x ,y)
#
# print(len(x),len(y))
#
# X_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=0)
# x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)
#
# model = keras.Sequential([
#     keras.layers.Dense(16,activation='relu'),
#     keras.layers.Dense(16,activation='relu'),
#     keras.layers.Dense(1,activation='sigmoid'),
#
# ])
#
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
#               loss=keras.losses.BinaryCrossentropy(),
#               metrics=['accuracy'])
#
# model.fit(X_train,y_train,batch_size=16, epochs=20, validation_data=(x_valid,y_valid))
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
# model.evaluate(x_test,y_test)