import msgpack
import numpy as np
from PIL import Image
import io

def load_and_preprocess_image(msgpack_record):
    image_bytes = msgpack_record['image']
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((300, 300))
    image_array = np.array(image, dtype=np.float32)
    image_array /= 255.0
    return image_array

dataset = []
count = 0

#count so it only extracts the first 10,000 images to stop running out of ram
with open('/content/shard_0.msg', 'rb') as f:
    unpacker = msgpack.Unpacker(f, raw=False)
    for record in unpacker:
        if count >= 10000:
            break
        image_array = load_and_preprocess_image(record)
        dataset.append((image_array, record['latitude'], record['longitude']))
        count += 1

#model architecture
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split

model = tf.keras.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(300, 300, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (5, 5), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    #output latitude and longitude predictions
    layers.Dense(2)
])

learning_rate = 0.0003
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mae'])


#training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dataset = dataset[:8000]
test_dataset = dataset[2000:]

train_images = np.array([data[0] for data in train_dataset])
train_labels = np.array([(data[1], data[2]) for data in train_dataset])
test_images = np.array([data[0] for data in test_dataset])
test_labels = np.array([(data[1], data[2]) for data in test_dataset])

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2)

mean_lat = np.mean(train_labels[:, 0])
std_lat = np.std(train_labels[:, 0])
mean_lon = np.mean(train_labels[:, 1])
std_lon = np.std(train_labels[:, 1])

datagen = ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
)

batch_size = 32
steps_per_epoch = len(train_images) // batch_size

train_generator = datagen.flow(train_images, train_labels, batch_size=batch_size)

epochs = 10
history = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=(val_images, val_labels),
                    epochs=epochs)



#graphing loss and validation loss curves
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, epochs + 1)

plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, max(max(loss), max(val_loss)) * 1.5])  #Set a custom y-axis limit
plt.legend()
plt.show()

#predicting uploaded image
import tensorflow as tf
from PIL import Image
import numpy as np

new_image_path = "/content/sd.png"
new_image = Image.open(new_image_path)

new_image = new_image.resize((300, 300))
new_image_array = np.array(new_image, dtype=np.float32) / 255.0

new_image_array = np.expand_dims(new_image_array, axis=0)

predictions = model.predict(new_image_array)

print("Predicted Latitude:", predictions[0, 0])
print("Predicted Longitude:", predictions[0, 1])
