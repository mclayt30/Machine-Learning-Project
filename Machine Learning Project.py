import tensorflow as tf
import keras.layers as layers
import matplotlib.pyplot as plt

dataset = tf.keras.datasets.mnist



(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
x_train, x_test = train_images/255.0, test_images/255.0

x_val = x_train[-10000:]
y_val = train_labels[-10000:]

x_train = x_train[:-10000]
train_labels = train_labels[:-10000]

model = tf.keras.models.Sequential()

# Model 1: 97.63% accuracy, loss = .0809
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation = 'relu'))
# model.add(layers.Dense(10))

# Model 1 with Dropout: 97.6% accuracy, loss = 0.0766
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation = 'relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(10))

# Model 2: 98.26% accuracy, loss = 0.0554
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(10))

# Model 2 with Dropout: 98.11% accuracy, loss = 0.0581
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Flatten())
# model.add(layers.Dense(10))

# Model 2 with Dropout: 98.11% accuracy, loss = 0.0581
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(10))

# MODEL 3 IN POWERPOINT
# Model 4: 98.89% accuracy, loss = 0.0412
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(10))

# Model 5: 99.2% accuracy, loss = 0.0245
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(10))



# Model 6: 99.11% accuracy, loss = 0.0237
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.Flatten())
# model.add(layers.Dense(10))

# Model 6: 99.23% accuracy, loss = 0.0225
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(10))


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
model.compile(optimizer = 'adam',
              loss = loss_fn,
              metrics = 'accuracy')

epochs = 5
history = model.fit(x_train, train_labels, epochs = epochs, validation_data = (x_val, y_val))

model.summary()

model.evaluate(x_test, test_labels, verbose = 2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


loss = history.history['loss']
val_loss = history.history['val_loss']



epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')


plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
