import numpy as np
import os
from os import listdir
from os.path import join, isdir
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from contextlib import redirect_stdout


# tf.enable_eager_execution()


def get_files_labels(data_directory, list_class_names):
    """Searches data_dir for directories named after the entries of classes and returns the file paths and labels."""

    files = []
    labels = []
    for label_name in list_class_names:
        class_path = join(data_directory, label_name)

        # Create list of file paths
        file_list = [
            join(class_path, f)
            for f in listdir(join(class_path))
            if f.endswith('.jpeg')
        ]
        # Create label list with equal length
        label_list = [label_name] * len(file_list)

        files.extend(file_list)
        labels.extend(label_list)

    return files, labels


def _get_read_func(target_height, target_width):

    def _image_read(filepath, label):
        image_string = tf.read_file(filepath)
        image_decoded_bw = tf.image.decode_and_crop_jpeg(image_string, [160, 580, 880, 820], channels=1)
        image_resized = tf.image.resize_images(image_decoded_bw, [target_height, target_width])
        image_stand = tf.image.per_image_standardization(image_resized)

        # plt.imshow(np.squeeze(image_resized.numpy()), cmap='gray')

        return image_stand, label

    return _image_read


def _get_read_func_augment(target_height, target_width):

    def _image_read_augment(filepath, label):
        image_string = tf.read_file(filepath)
        image_decoded_bw = tf.image.decode_and_crop_jpeg(image_string, [160, 580, 880, 820], channels=1)
        image_resized = tf.image.resize_images(image_decoded_bw, [target_height, target_width])
        image_stand = tf.image.per_image_standardization(image_resized)

        image_flip_l_r = tf.image.random_flip_left_right(image_stand)
        image_flip_u_d = tf.image.random_flip_up_down(image_flip_l_r)

        int_pick = tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32)
        image_rot = tf.image.rot90(image_flip_u_d, k=int_pick)

        return image_rot, label

    return _image_read_augment


def sequential_conv_model(input_shape, n_outputs, lr):
    model = tf.keras.Sequential()
    lay = tf.keras.layers

    model.add(lay.Conv2D(
        filters=10,
        kernel_size=3,
        strides=1,
        padding='same',
        activation='relu',
        input_shape=input_shape,
    ))
    model.add(lay.MaxPool2D(pool_size=(2, 2)))
    model.add(lay.BatchNormalization())
    model.add(lay.Conv2D(
        filters=20,
        kernel_size=3,
        strides=2,
        padding='same',
        activation='relu',
    ))
    model.add(lay.MaxPool2D(pool_size=(2, 2)))
    model.add(lay.BatchNormalization())
    model.add(lay.Conv2D(
        filters=30,
        kernel_size=3,
        strides=2,
        padding='same',
        activation='relu',
    ))
    model.add(lay.MaxPool2D(pool_size=(2, 2)))
    model.add(lay.BatchNormalization())
    model.add(lay.Flatten())
    model.add(lay.Dropout(0.20))
    model.add(lay.Dense(units=30, activation='relu'))
    model.add(lay.Dense(units=n_outputs, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.binary_crossentropy,
        metrics=['accuracy']
    )

    return model


def plot_history(save_dir, history_dict, save=True):

    acc_keys = [key for key in history_dict.keys() if 'acc' in key]
    for key in acc_keys:
        plt.plot(history_dict[key])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(acc_keys, loc='upper left')
    if save:
        plt.savefig(join(save_dir, 'accuracy_history.pdf'), bbox_inches='tight')
    plt.close()

    loss_keys = [key for key in history_dict.keys() if 'loss' in key]
    for key in loss_keys:
        plt.plot(history_dict[key])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loss_keys, loc='upper left')
    if save:
        plt.savefig(join(save_dir, 'loss_history.pdf'), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    data_dir = 'nailgun'
    classes = [
        'good',
        'bad'
    ]
    classes_ints = {class_name: i for i, class_name in enumerate(classes)}

    # Training variables
    BATCH_SIZE = 32
    EPOCHS = 200
    LEARNING_RATE = 0.001
    IMG_HEIGHT = 200
    IMG_WIDTH = 200

    # Reading in all file paths and labels and converting to numpy arrays
    files, labels = get_files_labels(data_dir, classes)
    files, labels = np.array(files), np.array(labels)

    # Convert labels to integers
    labels = np.array([classes_ints[class_name] for class_name in labels]).reshape((-1, 1)).astype(np.float32)

    # Split data into train and test
    files_train, files_test, labels_train, labels_test = train_test_split(
        files,
        labels,
        test_size=0.20,
        shuffle=True,
        stratify=labels,
        random_state=42
    )

    # Further split data for validation
    files_train, files_val, labels_train, labels_val = train_test_split(
        files_train,
        labels_train,
        test_size=0.20,
        shuffle=True,
        stratify=labels_train,
        random_state=1
    )

    # Create datasets with batch iteration and infinite looping (epochs specified in model.fit method).
    train_data = tf.data.Dataset.from_tensor_slices((files_train, labels_train))
    train_data = train_data.map(_get_read_func_augment(IMG_HEIGHT, IMG_WIDTH)).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((files_val, labels_val))
    val_data = val_data.map(_get_read_func(IMG_HEIGHT, IMG_WIDTH)).batch(BATCH_SIZE).repeat()

    test_data = tf.data.Dataset.from_tensor_slices((files_test, labels_test))
    test_data = test_data.map(_get_read_func(IMG_HEIGHT, IMG_WIDTH)).batch(BATCH_SIZE).repeat()

    model = sequential_conv_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),
        n_outputs=1,
        lr=LEARNING_RATE
    )

    # Generate unique model name and directory
    now = datetime.datetime.today()
    now_string = '{}-{}-{}_{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    model_name = '_'.join(['nailgun', now_string])
    base_dir = 'trained_models'
    model_dir = join(base_dir, model_name)
    if not isdir(model_dir):
        os.makedirs(model_dir)

    # Print and save model_summary
    model.summary()
    with open(join(model_dir, '{}_summary.txt'.format(model_name)), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    callbacks = [
        tf.keras.callbacks.CSVLogger(join(model_dir, '{}.csv'.format(model_name))),
        tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=25),
        tf.keras.callbacks.TensorBoard(join(model_dir, 'tensorboard_logs')),
        tf.keras.callbacks.ModelCheckpoint(join(model_dir, 'checkpoint_{epoch:02d}.h5'), save_best_only=True)
    ]

    history = model.fit(
        train_data,
        epochs=EPOCHS,
        steps_per_epoch=len(files_train) // BATCH_SIZE,
        validation_data=val_data,
        validation_steps=len(files_val) // BATCH_SIZE,
        callbacks=callbacks
    )

    plot_history(model_dir, history.history)

    model.save(join(model_dir, '{}_model.h5'.format(model_name)))

    # Generate and save predictions
    predictions_proba = model.predict(val_data, steps=len(files_val) // BATCH_SIZE)
    predictions = (predictions_proba > 0.5).astype(np.float)
    np.savetxt(join(model_dir, '{}_predictions_proba.txt'.format(model_name)), predictions_proba)
    np.savetxt(join(model_dir, '{}_predictions.txt'.format(model_name)), predictions)

    # Calculate confusion matrix
    cm = confusion_matrix(labels_val, predictions)
    print(cm)
    np.savetxt(join(model_dir, '{}_confusion_matrix.txt'.format(model_name)), cm)

    # tensorboard --logdir=/Users/Adem/PycharmProjects/Coding-Challenge/trained_models/nailgun_2019-1-20_14-0-1/tensorboard_logs --host localhost --port 8088

