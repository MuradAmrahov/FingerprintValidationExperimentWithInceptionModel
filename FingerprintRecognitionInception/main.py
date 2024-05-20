import zipfile
import os
from PIL import Image
import numpy as np

# Path to the uploaded .zip file
zip_path = 'C:/Users/user/Desktop/DB4_B.zip'
extract_path='C:/Users/user/Desktop/DB4_B'
# Function to extract zip file
def extract_zip(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return os.listdir(extract_path)  # List files to confirm extraction

# Extract the zip file
extracted_files = extract_zip(zip_path, extract_path)


def load_and_preprocess_images(file_paths, target_size=(299, 299)):
    images = []
    labels = []  # To store labels based on file naming convention

    for file_path in file_paths:
        # Open the image file
        with Image.open(f"{extract_path}/{file_path}") as img:
            # Resize image to match Inception model input
            img = img.resize(target_size)
            # Convert image to numpy array, normalize it, and add a channel dimension
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=-1)  # Add a channel dimension
            images.append(img_array)

            # Extract label from the filename, assuming format 'ID_X.tif'
            label = int(file_path.split('_')[0])
            labels.append(label)

    return np.array(images), np.array(labels)

def grayscale_to_rgb(images):
    # Assumes images have shape (batch_size, height, width, 1)
    return np.repeat(images, 3, axis=-1)
image_files = [f for f in extracted_files if f.endswith('.tif')]
images, labels = load_and_preprocess_images(image_files)



# Convert grayscale images to RGB
images_rgb = grayscale_to_rgb(images)

# List of file paths for images, excluding any directories
image_files = [f for f in extracted_files if f.endswith('.tif')]

# Load and preprocess images
images_rgb, labels = load_and_preprocess_images(image_files[:10])  # Limiting to first 10 for testing
images_rgb.shape, labels

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load the pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Adding custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(np.unique(labels)), activation='softmax')(x)

# Final model setup
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers which you don't want to train
for layer in base_model.layers:
    layer.trainable = False

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Create a label encoder object
label_encoder = LabelEncoder()

# Fit the label encoder to your labels and transform them to normalized labels
labels_encoded = label_encoder.fit_transform(labels)
# Assuming 'images' and 'labels' are your datasets loaded previously
X_train, X_val, y_train, y_val = train_test_split(images_rgb, labels_encoded, test_size=0.2, random_state=42)
print("Transformed label examples:", labels_encoded[:10])
predictions = Dense(np.unique(labels_encoded).size, activation='softmax')(x)

# Convert grayscale images to RGB
X_train_rgb = grayscale_to_rgb(X_train)
X_val_rgb = grayscale_to_rgb(X_val)


# Fit the model
history = model.fit(X_train_rgb, y_train, epochs=20, batch_size=32, validation_data=(X_val_rgb, y_val))

import matplotlib.pyplot as plt

# Evaluate the model on the test set
eval_result = model.evaluate(X_val_rgb, y_val)
print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}")

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model dəqiqliyi')
plt.ylabel('Dəqiqlik')
plt.xlabel('Dövr')
plt.legend(['Təlim', 'Doğrulama'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model itkisi')
plt.ylabel('İtki')
plt.xlabel('Dövr')
plt.legend(['Təlim', 'Doğrulama'], loc='upper left')
plt.show()
#---------------------
#----------------
#---------------
#from tensorflow.keras.utils import to_categorical

# Make predictions
y_pred_prob = model.predict(X_val_rgb)
y_pred = np.argmax(y_pred_prob, axis=1)

# If you used label encoding, convert labels back to original encoding for better interpretability
y_val_original = label_encoder.inverse_transform(y_val)
y_pred_original = label_encoder.inverse_transform(y_pred)

#-------------
from sklearn.metrics import confusion_matrix, accuracy_score

conf_matrix = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Assuming binary classification for simplicity; adjust indices accordingly for multiclass
TP = conf_matrix[1, 1]
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
#--------------------------------
TPR = TP / (TP + FN)  # True Positive Rate (Sensitivity)
TNR = TN / (TN + FP)  # True Negative Rate (Specificity)
FPR = FP / (TN + FP)  # False Positive Rate
FNR = FN / (TP + FN)  # False Negative Rate
accuracy = (TP + TN) / (TP + TN + FP + FN)  # Overall accuracy

print("True Positive Rate (Sensitivity):", TPR)
print("True Negative Rate (Specificity):", TNR)
print("False Positive Rate:", FPR)
print("False Negative Rate:", FNR)
print("Accuracy:", accuracy)
#---------------------------------
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_val, y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='QİX əyrisi  (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Müsbət Dərəcə')
plt.ylabel('Həqiqi Müsbət Dərəcə')
plt.title('Qəbuledicinin İşləmə Xarakteristikası')
plt.legend(loc="lower right")
plt.show()



