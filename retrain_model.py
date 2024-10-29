import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from tensorflow.keras.layers import (Input, BatchNormalization, AveragePooling1D, MaxPooling1D,
                                     Conv1D, Bidirectional, LSTM, Dense, Dropout, LeakyReLU)


from tensorflow.keras.optimizers.legacy import Adam
from scipy.signal import resample
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import warnings

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.__version__)

import platform 
print(platform. python_version())

def build_model(x_train, dropout):
    '''
    Builds a model with the same architecture as the base model, but with a different input shape.
    '''
    dropout_rate = dropout

    model = tf.keras.models.Sequential()
    model.add(Input(shape=(x_train.shape[1:])))          
    model.add(Conv1D(filters=32, kernel_size=9, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=3, strides=3))
    
    model.add(Conv1D(filters=48, kernel_size=7, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=2, strides=2))
    
    model.add(Conv1D(filters=64, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=2, strides=2))
    
    model.add(Conv1D(filters=80, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Bidirectional(LSTM(96, dropout=dropout_rate), 
                            merge_mode='sum'))
    model.add(Dense(128, activation=LeakyReLU()))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

date_prefix = datetime.now().strftime("%Y_%b_%d")

# Set training params
training_size = 0.8
testing_size = 'None'
dropout = 0.5 
learn_rate = 2e-8
epsilon = 1e-8
patience = 15
epochs = 100
batch_size = 32
training_data = './model_training/training_data/complete_training_dataset.h5'
base_model_name = './models/GC_lstm_model.h5'


settings = {}
settings['training_size'] = training_size
settings['testing_size'] = testing_size
settings['learn_rate'] = learn_rate
settings['epsilon'] = epsilon
settings['patience'] = patience
settings['epochs'] = epochs
settings['batch_size'] = batch_size
settings['dropout'] = dropout
settings['training_data'] = training_data
settings['base_model'] = base_model_name

text_format = ''
for i in settings:
    text_format += f'{i}: {settings[i]}\n'

    # Load the training data

with h5py.File(training_data, 'r') as f:
    x = f['events'][:]
    y = f['scores'][:]

fig, axs = plt.subplots(1, 2)
axs[0].plot(x[1])
axs[0].set_title('before resampling and inverting\n')

# Resample the data to match the 600 datapoints in the base model:
x = resample(x, 600, axis = 1)

# # Invert the events: The base model was trained to detected downward deflections. In our experience it helps to invert the data when looking at events
# # that have a different direction. When using the model, remember to adjust the event_direction paramter to: direction = 'positive' when generating the
# # EventDetection object. 
# x *= -1


# Define the output folder
out_folder = './model_training/out_model_traininig/'

new_dir_path = os.path.join(out_folder, datetime.now().strftime("%b_%Y"))  # Format: Mon_YYYY

# Create the directory if it doesn't exist
os.makedirs(new_dir_path, exist_ok=True)

print(f"Results will be saved in : {new_dir_path}")

axs[1].plot(x[1])
axs[1].set_title('after resampling and inverting\n')
plt.tight_layout()
plt.savefig(new_dir_path + '/' + date_prefix +'_resampling_and_inverting.png')
plt.close()

# Scale and split the data.
scaled_data = minmax_scale(x, feature_range=(0,1), axis=1) #scales all data to be (0,1) 
scaled_data = np.expand_dims(scaled_data, axis=2)
merged_y = np.expand_dims(y, axis=1)

print(f'loaded events with shape {scaled_data.shape}')
print(f'loaded scores with shape {merged_y.shape}') 
print(f'ratio of pos/neg scores: {merged_y.sum()/(merged_y.shape[0]-merged_y.sum()):.2f}')

if merged_y.sum()/(merged_y.shape[0]-merged_y.sum()) > 1.05:
    warnings.warn("unbalanced dataset: ratio of positive and negative scores >1.05")
elif merged_y.sum()/(merged_y.shape[0]-merged_y.sum()) < 0.95:
    warnings.warn("unbalanced dataset: ratio of positive and negative scores < 0.95")


x_train, x_test, y_train, y_test = train_test_split(scaled_data, merged_y, train_size=training_size, random_state=1234)


# open old model
old_model = tf.keras.models.load_model(base_model_name, compile=compile)

# save weights from GC model
old_model.save_weights(new_dir_path + '/' + date_prefix + '_gc_weights')

# re-create model with different input shape
new_model = build_model(x_train, dropout)
new_model.summary()

# Load old model weights into new model
new_model.load_weights(new_dir_path + '/' + date_prefix + '_gc_weights')

# Freeze / unfreeze layers that should / should not be trained
for ind, layer in enumerate(new_model.layers):
    if layer.name=="BatchNormalization":
        new_model.layers[ind].trainable = False
        print(f'layer {ind} ({layer}) now untrainable')

    else:
        if ind < len(new_model.layers)-4:
            new_model.layers[ind].trainable = False
            print(f'layer {ind} ({layer}) now untrainable')
        else:
            new_model.layers[ind].trainable = True
            print(f'layer {ind} ({layer}) now trainable')
              
                
new_model.compile(optimizer=Adam(learning_rate=learn_rate, epsilon=epsilon, amsgrad=True),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['Accuracy'])

new_model.summary()

# Train chosen layers of new model        
checkpoint_filepath = new_dir_path + '/' + date_prefix + '_model_{epoch:02d}_{val_Accuracy:.03f}.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_Accuracy',
    mode='max',
    save_best_only=True)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True)

start = time.time()

history = new_model.fit(x_train,y_train,
            verbose=1,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test,y_test),
            shuffle=True,
            callbacks=[early_stopping_callback])

print('')
print(f'train shape: {x_train.shape}')
print(f'score on val: {new_model.evaluate(x_test,y_test)[1]}')
print(f'score on train: {new_model.evaluate(x_train,y_train)[1]}')

end = time.time()
print(end-start)

acc = history.history['Accuracy']
val_acc = history.history['val_Accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'y', label='Training')
plt.plot(epochs, val_acc, 'r', label='Validation')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(new_dir_path + '/' + date_prefix + '_acuracy.png')
plt.close()

best_epoch = val_acc.index(max(val_acc)) + 1
print(f'Best epoch: {best_epoch} (accuracy={max(val_acc):.4f})')
n_epochs = len(history.history['loss'])
loss = history.history['loss']
val_loss = history.history['val_loss']


plt.plot(epochs, loss, 'y', label='Training')
plt.plot(epochs, val_loss, 'r', label='Validation')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(new_dir_path + '/' + date_prefix + '_loss.png')
plt.close()

loss_val = new_model.evaluate(x_test,y_test)[0]
acc_val = new_model.evaluate(x_test,y_test)[1]

# Roc curve
y_preds = new_model.predict(x_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_preds)
plt.plot([0, 1], [0, 1], 'y--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.savefig(new_dir_path + '/' + date_prefix + '_ROC_curve_dataset2.png')
plt.close()

print('Area under curve, AUC = ', auc(fpr, tpr))
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
print('Optimal threshold value is:', optimal_threshold)

y_pred2 = (new_model.predict(x_test) >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.savefig(new_dir_path + '/' + date_prefix + '_confusion_matrix_dataset1.png')
plt.close()

# save results        

model_name = './models/transfer_learning/human_pyramids_L2_3/' + date_prefix + '_lstm_transfer'
new_model.save(model_name + '.h5')

np.savetxt(new_dir_path + '/' + date_prefix + '_fpr.txt', fpr)
np.savetxt(new_dir_path + '/' + date_prefix + '_tpr.txt', tpr)
np.savetxt(new_dir_path + '/' + date_prefix + '_accuracy.txt', [epochs, acc, val_acc])
np.savetxt(new_dir_path + '/' + date_prefix + '_loss.txt', [epochs, loss, val_loss])

with open(new_dir_path + '/' + date_prefix + '_training_settings.txt', 'w') as text_file:
    text_file.write(text_format)

