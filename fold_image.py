import os
import cv2
from deepface import DeepFace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from time import time
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
from skimage import feature
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Function to calculate embeddings using DeepFace ArcFace
def calc_embs(model, X):
    pd = []
    for image in X:
        img_embedding = DeepFace.represent(img_path=image, model_name='ArcFace', enforce_detection=False)
        pd.append(img_embedding[0]['embedding'])
    return np.array(pd)

def save_image_with_error_handling(dest_folder, class_name, n_images, image):
    try:
        cv2.imwrite(os.path.join(dest_folder, os.path.join(class_name, str(n_images) + '_faces.jpg')), image)
    except cv2.error as e:
        print(f"An error occurred while saving image {str(n_images)}: {e}")

# Data location
folder = "data/rvf10k/train"
dest_folder = "data/roi_dataset"
cascade_path = "data/haarcascade_frontalface_default.xml"

# Counter for the number of classes in the dataset
nclasses = 0
# Counter for samples per class
nperclass = []
# Label for each class (name of the subfolder)
classlabels = []

n_images = 0

# Lists to store image paths and labels
image_paths = []
labels = []

# Assumes that there is a subfolder per class in the given path
for class_name in os.listdir(folder):
    # Each subfolder implies one more class
    nclasses += 1
    # Initially, this class has no samples
    nsamples = 0

    # Compose the path
    class_folder = os.path.join(folder, class_name)
    for file_name in os.listdir(class_folder):
        # Assumes images are in jpg format
        if file_name.endswith('.jpg'):
            # Read the image
            image = cv2.imread(os.path.join(class_folder, file_name))

            # Extract face as ROI
            faceCascade = cv2.CascadeClassifier(cascade_path)
            faces = faceCascade.detectMultiScale(
                image,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) > 0:
                for x, y, w, h in faces:
                    image = image[y:y + h, x:x + w]
                    save_image_with_error_handling(dest_folder, class_name, n_images, image)
                    n_images += 1
            else:
                cv2.imwrite(
                    os.path.join(dest_folder, os.path.join(class_name, str(n_images) + '_faces.jpg')), image)
                n_images += 1

            # Append image path and label
            image_paths.append(os.path.join(class_folder, file_name))
            labels.append(nclasses - 1)

# Convert lists to numpy arrays
image_paths = np.array(image_paths)
labels = np.array(labels, dtype='float64')

# Show information about the read dataset
print("Number of images:", len(image_paths))
print("Number of classes:", nclasses)

# Deepface
# Available models ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
model_deepface = DeepFace.build_model("ArcFace")

target_size = (model_deepface.input_shape[0], model_deepface.input_shape[1])
dim = (int(target_size[0]), int(target_size[1]))

# StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

# DataFrame to store image paths for each fold
folds_df = pd.DataFrame(columns=['Fold', 'Train_Images', 'Test_Images'])

fold = 1
folds_data = []

for train_index, test_index in skf.split(image_paths, labels):
    print(f"Fold {fold}")
    print("# samples in training set:", train_index.shape[0])
    print("# samples in test set:", test_index.shape[0])
    
    # Get train and test image paths
    train_images = image_paths[train_index]
    test_images = image_paths[test_index]
    
    # Append to DataFrame
    folds_data.append({'Fold': fold, 'Train_Images': train_images, 'Test_Images': test_images})
    
    fold += 1

# Convert list of dicts to DataFrame
folds_df = pd.DataFrame(folds_data)

# Save DataFrame to CSV
folds_df.to_csv('folds_image_paths.csv', index=False)

print("Image paths for each fold have been saved to folds_image_paths.csv")
