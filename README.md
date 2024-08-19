# Leaf-Disease-Classification
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from PIL import Image

image_height, image_width = 128, 128
batch_size = 32
epochs = 30  # Adjust as needed
num_classes = 10  # Assuming you have 10 disease classes
validation_split = 0.2  # 20% of data for validation

disease_names = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot', 'neck_blast', 'rice_hispa', 'sheath_blight', 'tungro']

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    '/content/Rice_Leaf_Diease/Rice_Leaf_Diease/train',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

test_generator = train_datagen.flow_from_directory(
    '/content/Rice_Leaf_Diease/Rice_Leaf_Diease/train',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

disease_names = ['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast', 'leaf_scald', 'narrow_brown_spot', 'neck_blast', 'rice_hispa', 'sheath_blight', 'tungro']

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(np.arange(0, epochs+1, 5))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('Training and Validation Accuracy')
plt.show()

model.save('rice_leaf_model.h5')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load the saved model
model = load_model('/content/drive/MyDrive/rice_leaf_model.h5')

# Define image dimensions and batch size
image_height, image_width = 128, 128
batch_size = 32

# Prepare test data generator with data preprocessing
test_datagen = ImageDataGenerator(rescale=1./255)  # Rescale pixel values to [0, 1]
test_generator = test_datagen.flow_from_directory(
    '/content/Rice_Leaf_Diease/Rice_Leaf_Diease/test',
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',  # Set to 'categorical' for multi-class classification
    shuffle=False  # Maintain order of data
)

# Evaluate the model on the test data
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())


# Print confusion matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)

# Calculate overall accuracy
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
