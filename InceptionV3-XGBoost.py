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
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical'
)

    # Validation generator
val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_meta,
    directory=final_dataset,
    x_col='Image_path',
    y_col='label',
    target_size=(299, 299),
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
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

img_input = Input(shape=(299, 299, 3))
classes=7
channel_axis=3
def conv2d_bn(x,filters,num_row,num_col,padding='same',strides=(1, 1)):
   
    x = keras.layers.Conv2D(filters, (num_row, num_col),strides=strides,padding=padding)(x)
    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
    x = keras.layers.Activation('relu')(x)
    return x

def inc_block_a(x):    
    branch1x1 = conv2d_bn(x, 64, 1, 1)  # 64 filters of 1*1
    branch5x5 = conv2d_bn(x, 48, 1, 1)  #48 filters of 1*1
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis)
    return x

def reduction_block_a(x):  
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool],axis=channel_axis)
    return x

def inc_block_b(x):
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis)
    return x

def reduction_block_b(x): 
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn( branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis)
    return x

# def auxiliary_classifier(prev_Layer):
#     x = AveragePooling2D(pool_size=(5,5) , strides=(3,3)) (prev_Layer)
#     x = conv_with_Batch_Normalisation(x, nbr_kernels = 128, filter_Size = (1,1))
#     x = Flatten()(x)
#     x = Dense(units = 768, activation='relu') (x)
#     x = Dropout(rate = 0.2) (x)
#     x = Dense(classes, activation='softmax') (x)
#     return x

def inc_block_c(x):        
    branch1x1 = conv2d_bn(x, 320, 1, 1)
    branch3x3 = conv2d_bn(x, 384, 1, 1)
    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
    branch3x3 = keras.layers.concatenate([branch3x3_1, branch3x3_2],axis=channel_axis)
    branch3x3dbl = conv2d_bn(x, 448, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
    branch3x3dbl = keras.layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)
    branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = keras.layers.concatenate( [branch1x1, branch3x3, branch3x3dbl, branch_pool],axis=channel_axis)
    return x

x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid') # 149 x 149 x 32
x = conv2d_bn(x, 32, 3, 3, padding='valid')  # 147 x 147 x 32
x = conv2d_bn(x, 64, 3, 3) # 147 x 147 x 64

x = MaxPooling2D((3, 3), strides=(2, 2))(x)   # 73  x 73 x 64
x = conv2d_bn(x, 80, 1, 1, padding='valid') # 73 x 73 x 80
x = conv2d_bn(x, 192, 3, 3, padding='valid')  # 71 x 71 x 192
x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 35 x 35 x 192


x=inc_block_a(x) #35, 35, 256
x=inc_block_a(x) #35, 35, 256
x=inc_block_a(x) #35, 35, 256

x=reduction_block_a(x) #17, 17, 736

x=inc_block_b(x) #17, 17, 768
x=inc_block_b(x) #17, 17, 768
x=inc_block_b(x) #17, 17, 768
x=inc_block_b(x) #17, 17, 768

# Aux = auxiliary_classifier(prev_Layer = x)

x=reduction_block_b(x) #shape=(None, 8, 8, 1280)

x=inc_block_c(x) # shape=(None, 8, 8, 2048) 
x=inc_block_c(x) # shape=(None, 8, 8, 2048) 

x = GlobalAveragePooling2D(name='avg_pool')(x) # shape=(None, 2048)

x = Dense(classes, activation='softmax', name='predictions')(x) #shape=(None, 1000) 


inputs = img_input
model =  Model(inputs, x, name='inception_v3')
model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_accuracy',patience=10,restore_best_weights=True)
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stopping]
)


train_loss, train_accuracy = model.evaluate(train_generator)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

os.makedirs('/kaggle/working/saved_models', )

save_model(model, "/kaggle/working/saved_models/Inception_v3_final.h5")

test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
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

#  Combine Inception V3 with SVM, RF, KNN and XGBoost

#extract the features from the Inception V3 model excluding the fully connected or the dense layer

layer_name = 'avg_pool'
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)


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

np.savez('Inception_v3_features_final.npz', 
         train_features=train_features, train_labels=train_labels,
         test_features=test_features, test_labels=test_labels)

Inception_xgb = xgb.XGBClassifier(objective='multi:softmax', num_class=7, random_state=42, n_estimators = 100, max_depth = 8, )

Inception_xgb.fit(train_features, train_labels)

val_predictions = Inception_xgb.predict(test_features)

print(classification_report(test_labels, val_predictions, target_names=class_labels.values()))

accuracy = accuracy_score(test_labels, val_predictions)
precision = precision_score(test_labels, val_predictions, average='weighted')
recall = recall_score(test_labels, val_predictions, average='weighted')
f1 = f1_score(test_labels, val_predictions, average='weighted')


print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Precision (Weighted): {precision:.4f}")
print(f"Recall (Weighted): {recall:.4f}")
print(f"F1-Score (Weighted): {f1:.4f}")


cm = confusion_matrix(test_labels, val_predictions)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels.values(), yticklabels=class_labels.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

joblib.dump(Inception_xgb, 'Inception-xgb_model.joblib')