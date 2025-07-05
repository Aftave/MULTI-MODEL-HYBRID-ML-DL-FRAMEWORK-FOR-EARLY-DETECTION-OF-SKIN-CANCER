import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential, Model, load_model,save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, ZeroPadding2D, BatchNormalization, Activation, Add, AveragePooling2D, GlobalAveragePooling2D
import numpy as np
from tensorflow.keras import activations
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from keras.applications import vgg19, EfficientNetB0, ResNet50
from PIL import Image
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
import keras
from sklearn.svm import SVC
import matplotlib.image as mpimg
from glob import glob
from keras.initializers import glorot_uniform
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib

img_part_1 = ('/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_1')

img_part_2 = ('/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_images_part_2')
os.makedirs('/kaggle/working/final_dataset')
final_dataset = ('/kaggle/working/final_dataset')

def copy_images(source_path, destination_path):
        for filename in os.listdir(source_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                shutil.copy(os.path.join(source_path, filename), destination_path)


copy_images(img_part_1, final_dataset)
copy_images(img_part_2, final_dataset)

meta_data = pd.read_csv('/kaggle/input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')

meta_data.head()
meta_data['dx'].value_counts()

os.makedirs('/kaggle/working/images')

meta_data['dx'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Lesion Type')
plt.ylabel('Count')
plt.show()
plt.savefig('/kaggle/working/images/distribution.png')

meta_data['Image_path'] = meta_data['image_id'].apply(lambda x: os.path.join(final_dataset, f"{x}.jpg"))


le = LabelEncoder()
meta_data['label'] = le.fit_transform(meta_data["dx"])
meta_data['label'] = meta_data['label'].astype(str)


meta_data.head()

n_samples = 6705
# Separating each class
df_nv = meta_data[meta_data['dx'] == 'nv']
df_mel = meta_data[meta_data['dx'] == 'mel']
df_bkl = meta_data[meta_data['dx'] == 'bkl']
df_bcc = meta_data[meta_data['dx'] == 'bcc']
df_akiec = meta_data[meta_data['dx'] == 'akiec']
df_vasc = meta_data[meta_data['dx'] == 'vasc']
df_df = meta_data[meta_data['dx'] == 'df']


df_nv_balanced = resample(df_nv, replace=False, n_samples=n_samples, random_state=42)
df_mel_balanced = resample(df_mel, replace=True, n_samples=n_samples, random_state=42)
df_bkl_balanced = resample(df_bkl, replace=True, n_samples=n_samples, random_state=42)
df_bcc_balanced = resample(df_bcc, replace=True, n_samples=n_samples, random_state=42)
df_akiec_balanced = resample(df_akiec, replace=True, n_samples=n_samples, random_state=42)
df_vasc_balanced = resample(df_vasc, replace=True, n_samples=n_samples, random_state=42)
df_df_balanced = resample(df_df, replace=True, n_samples=n_samples, random_state=42)


balanced_meta_data = pd.concat([df_nv_balanced, df_mel_balanced, df_bkl_balanced,
                                    df_bcc_balanced, df_akiec_balanced, df_vasc_balanced,
                                    df_df_balanced])

label_mapping = {
    'nv': 'Melanocytic Nevus',
    'mel': 'Melanoma',
    'bkl': 'Benign Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc': 'Vascular Lesion',
    'df': 'Dermatofibroma'
}

balanced_meta_data['dx'] = balanced_meta_data['dx'].replace(label_mapping)

print(balanced_meta_data['dx'].value_counts())
balanced_meta_data = balanced_meta_data.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Balanced dataset class distribution:\n{balanced_meta_data['dx'].value_counts()}")


balanced_meta_data.head()




train_meta, test_meta = train_test_split(balanced_meta_data, test_size=0.3, random_state=42)
train_meta, val_meta = train_test_split(train_meta, test_size=0.3, random_state=42)

print(f"Training set size: {len(train_meta)}")
print(f"Validation set size: {len(val_meta)}")
print(f"Testing set size: {len(test_meta)}")

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)
# Train generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_meta,
    directory=final_dataset,
    x_col='Image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical'
)

    # Validation generator
val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_meta,
    directory=final_dataset,
    x_col='Image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

    # Test generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_meta,
    directory=final_dataset,
    x_col='Image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)


def identity_block(x,f,filters):
    f1, f2, f3 = filters
    x_shortcut = x

    #Layer 1
    x = Conv2D(filters= f1, kernel_size = (1,1),strides = (1,1), padding = 'valid')(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)

    #Layer 2
    x = Conv2D(filters = f2, kernel_size = (f,f), strides =(1,1), padding='same')(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)

    #Layer 3
    x = Conv2D(filters = f3, kernel_size = (1,1),strides = (1,1), padding = 'valid')(x)
    x = BatchNormalization(axis = 3)(x)

    #Adding the shortcut value
    x = Add()([x,x_shortcut])
    x = Activation('relu')(x)


    return x



def convolutional_block(x,f,filters, s=2):

    f1,f2,f3 = filters

    x_shortcut = x

    #Layer 1
    x = Conv2D(filters = f1,kernel_size = (1,1), strides = (s,s))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)


    #Layer 2
    x = Conv2D(filters = f2, kernel_size = (f,f), strides = (1,1),padding = 'same')(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)


    #Layer 3
    x = Conv2D(filters = f3, kernel_size = (1,1), strides = (1,1), padding = 'valid')(x)
    x = BatchNormalization(axis = 3)(x)

    # Shortcut
    x_shortcut = Conv2D(filters = f3, kernel_size = (1,1), strides = (s,s), padding = 'valid')(x_shortcut)
    x_shortcut = BatchNormalization(axis = 3)(x_shortcut)

    x = Add()([x,x_shortcut])
    x = Activation('relu')(x)

    return x




def resnet50(input_shape = (224,224,3), classes = 7):

    x_input = Input(input_shape)

    x = ZeroPadding2D((3,3))(x_input)

    #Stage 1
    x = Conv2D(64,(7,7),strides = (2,2))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = MaxPooling2D((3,3), strides = (2,2))(x)
    


    # Stage 2
    x = convolutional_block(x,f=3, filters=[64,64,256], s=1)
    x = identity_block(x,3,[64,64,256])
    x = identity_block(x,3,[64,64,256])


    #stage 3
    x = convolutional_block(x,f=3, filters=[128,128,512], s=2)
    x = identity_block(x,3,[128,128,512])
    x = identity_block(x,3,[128,128,512])
    x = identity_block(x,3,[128,128,512])

    #Stage 4
    x = convolutional_block(x,f=3, filters=[256,256,1024], s=2)
    x = identity_block(x,3,[256,256,1024])
    x = identity_block(x,3,[256,256,1024])
    x = identity_block(x,3,[256,256,1024])
    x = identity_block(x,3,[256,256,1024])
    x = identity_block(x,3,[256,256,1024])


    #Stage 5
    x = convolutional_block(x,f=3, filters=[512,512,2048], s=2)
    x = identity_block(x,3,[512,512,2048])
    x = identity_block(x,3,[512,512,2048])

    #avg pool
    x = AveragePooling2D((2,2), name = 'avg_pool')(x)
    x = Flatten()(x)
    x = Dense(classes, activation = 'softmax', name = 'fc'+ str(classes), kernel_initializer = glorot_uniform(seed=0))(x)
    model = Model(inputs = x_input, outputs = x, name= 'ResNet50')

    return model


resnet50_model = resnet50(input_shape=(224,224,3), classes = 7)
resnet50_model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics = ['accuracy'])
resnet50_model.summary()

early_stopping = EarlyStopping(monitor='val_accuracy',patience=10,restore_best_weights=True)


history = resnet50_model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

train_loss, train_accuracy = resnet50_model.evaluate(train_generator)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

test_loss, test_accuracy = resnet50_model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

os.makedirs('/kaggle/working/saved_models', )

save_model(resnet50_model, "/kaggle/working/saved_models/resnet50_final.h5")

test_generator.reset()
predictions = resnet50_model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes


class_labels = {
    0: 'Actinic Keratosis',    # akiec
    1: 'Basal Cell Carcinoma', # bcc
    2: 'Benign Keratosis',     # bkl
    3: 'Dermatofibroma',       # df
    4: 'Melanoma',            # mel
    5: 'Melanocytic Nevi',    # nv
    6: 'Vascular Lesion'      # vasc
}


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
# plt.save('/kaggle/working/images')


print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels.values()))


accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')


print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1-Score (Weighted): {f1:.4f}")


#  Combine ResNet50 with SVM, RF, KNN and XGBoost

#extract the features from the ResNet50 model excluding the fully connected or the dense layer

layer_name = 'flatten'
intermediate_layer_model = Model(inputs=resnet50_model.input,outputs=resnet50_model.get_layer(layer_name).output)


def extract_features(generator, model, num_samples):
    features = np.zeros((num_samples, model.output_shape[1]))  
    labels = np.zeros((num_samples,))  
    
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = intermediate_layer_model.predict(inputs_batch)
        # Check the shape of the features batch
        print(f"Batch {i}: features_batch shape: {features_batch.shape}")
        
        batch_size = inputs_batch.shape[0]  # Get the actual size of the current batch
        
        # Ensure the current index is within bounds
        start = i * generator.batch_size
        end = start + batch_size
        
        # Prevent exceeding the number of samples
        end = min(end, num_samples)
        
        if start >= num_samples: 
            break
        
        features[start:end] = features_batch[:end - start]
        labels[start:end] = np.argmax(labels_batch[:end-start], axis=1)
        
        i += 1
        
        # Break the loop when enough samples have been processed
        if end >= num_samples:
            break

    return features, labels
num_train_samples = train_generator.samples
num_test_samples = test_generator.samples

train_features, train_labels = extract_features(train_generator,intermediate_layer_model, num_train_samples)
test_features, test_labels = extract_features(test_generator,intermediate_layer_model, num_test_samples)

np.savez('resnet50_features_final.npz', 
         train_features=train_features, train_labels=train_labels,
         test_features=test_features, test_labels=test_labels)

knn_resnet = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform')  
knn_resnet.fit(train_features, train_labels)


y_val_pred = knn_resnet.predict(test_features)

print(classification_report(test_labels, y_val_pred, target_names=class_labels.values()))

accuracy = accuracy_score(test_labels, y_val_pred)
precision = precision_score(test_labels, y_val_pred, average='weighted')
recall = recall_score(test_labels, y_val_pred, average='weighted')
f1 = f1_score(test_labels, y_val_pred, average='weighted')


print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1-Score (Weighted): {f1:.4f}")


cm = confusion_matrix(test_labels, y_val_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


joblib.dump(knn_resnet, 'knn_resnet.joblib')