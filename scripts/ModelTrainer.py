from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import sys

import ModelFactory

if (len(sys.argv) != 2):
    print('Usage: python <modelName>')
    print('\t<modelName>: one of "simple", "vgg_finetune"')
    sys.exit()

MODEL_NAME = sys.argv[1]
SEED = 42
BATCH_SIZE=32
STEPS_PER_EPOCH = 500
IMG_SIZE = (224,224)

#Define the data generator parameters
train_generator = ImageDataGenerator(
        rescale=1./255.,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        rotation_range=20,
        vertical_flip=True)

val_generator = ImageDataGenerator(rescale=1./255.)
test_generator = ImageDataGenerator(rescale=1./255.)

#Create model
model = ModelFactory.MakeModel(MODEL_NAME)

#Print the summary
model.summary()

#Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0.01, patience = 10)
model_checkpoint = ModelCheckpoint('{0}_model.{1}-{2}.h5'.format(MODEL_NAME, '{epoch:02d}', '{val_loss:.7f}'), save_best_only = False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience = 5)

#Initialize data generators
train_datagenerator = train_generator.flow_from_directory(
    '../data/images/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED)

val_datagenerator = val_generator.flow_from_directory(
    '../data/images/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED)

test_datagenerator = test_generator.flow_from_directory(
    '../data/images/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED, 
    shuffle=False)

#Fit the model
history = model.fit_generator(
    train_datagenerator,
    STEPS_PER_EPOCH,
    epochs=400,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    validation_data=val_datagenerator,
    validation_steps=val_datagenerator.samples / BATCH_SIZE)

#Predict
predictions = model.predict_generator(
    test_datagenerator,
    steps = test_datagenerator.samples / BATCH_SIZE)
    
#Save results to file (annoyingly, there isn't a builtin for this...)
with open('predictions_{0}.tsv'.format(MODEL_NAME), 'w') as f:
    f.write('Is_Sandwich\tIs_Not_Sandwich\tFile_Name\n')
    for i in range(0, predictions.shape[0], 1):
        f.write('{0}\t{1}\t{2}\n'.format(predictions[i][0], predictions[i][1], test_datagenerator.filenames[i]))

#Save the model so it can be loaded later
model.save('{0}.h5'.format(MODEL_NAME))
