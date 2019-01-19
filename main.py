import numpy as np
from os import listdir
from os.path import join, isfile
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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

        # plt.imshow(np.squeeze(image_resized.numpy()), cmap='gray')

        return image_resized, label

    return _image_read


def sequential_conv_model(input_shape, n_outputs):
    model = tf.keras.Sequential()
    lay = tf.keras.layers

    model.add(lay.InputLayer(input_shape=input_shape))
    model.add(lay.Conv2D(
        filters=10,
        kernel_size=3,
        strides=2,
        padding='same',
        activation='relu'
    ))
    model.add(lay.MaxPool2D(pool_size=(2, 2)))
    model.add(lay.Conv2D(
        filters=20,
        kernel_size=3,
        strides=2,
        padding='same',
        activation='relu'
    ))
    model.add(lay.MaxPool2D(pool_size=(2, 2)))
    model.add(lay.Conv2D(
        filters=20,
        kernel_size=3,
        strides=2,
        padding='same',
        activation='relu'
    ))
    model.add(lay.MaxPool2D(pool_size=(2, 2)))
    model.add(lay.Flatten())
    model.add(lay.Dense(units=30, activation='relu'))
    model.add(lay.Dense(units=n_outputs, activation='softmax'))

    model.compile(
        optimizer=tf.train.AdamOptimizer(LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':

    data_dir = 'nailgun'
    classes = [
        'good',
        'bad'
    ]
    classes_ints = {class_name: i for i, class_name in enumerate(classes)}

    # Training variables
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.01
    IMG_HEIGHT = 200
    IMG_WIDTH = 200

    # Reading in all file paths and labels and converting to numpy arrays
    files, labels = get_files_labels(data_dir, classes)
    files, labels = np.array(files), np.array(labels)

    # Convert labels to integers
    labels = np.array([classes_ints[class_name] for class_name in labels]).reshape((-1, 1))

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

    # # Convert labels to one hot encoding
    # labels_train = to_categorical(labels_train)
    # labels_val = to_categorical(labels_val)
    # labels_test = to_categorical(labels_test)

    # Create placeholders for the dataset inputs for less memory usage during a session.
    files_train_placeholder = tf.placeholder(files.dtype, files_train.shape)
    labels_train_placeholder = tf.placeholder(labels.dtype, labels_train.shape)

    files_val_placeholder = tf.placeholder(files.dtype, files_val.shape)
    labels_val_placeholder = tf.placeholder(labels.dtype, labels_val.shape)

    files_test_placeholder = tf.placeholder(files.dtype, files_test.shape)
    labels_test_placeholder = tf.placeholder(labels.dtype, labels_test.shape)

    # Create datasets with batch iteration and infinite looping (epochs specified in model.fit method).
    train_data = tf.data.Dataset.from_tensor_slices((files_train_placeholder, labels_train_placeholder))
    train_data = train_data.map(_get_read_func(IMG_HEIGHT, IMG_WIDTH)).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((files_val_placeholder, labels_val_placeholder))
    val_data = val_data.map(_get_read_func(IMG_HEIGHT, IMG_WIDTH)).batch(BATCH_SIZE).repeat()

    test_data = tf.data.Dataset.from_tensor_slices((files_test_placeholder, labels_test_placeholder))
    test_data = test_data.map(_get_read_func(IMG_HEIGHT, IMG_WIDTH)).batch(BATCH_SIZE).repeat()

    # Create iterators for every dataset (necessary for datasets with placeholders)
    train_iterator = train_data.make_initializable_iterator()
    val_iterator = val_data.make_initializable_iterator()
    test_iterator = test_data.make_initializable_iterator()

    with tf.Session() as sess:
        tf.keras.backend.set_session(sess)

        # Initialize train and validation iterators.
        sess.run(train_iterator.initializer, feed_dict={files_train_placeholder: files_train,
                                                        labels_train_placeholder: labels_train})
        sess.run(val_iterator.initializer, feed_dict={files_val_placeholder: files_val,
                                                      labels_val_placeholder: labels_val})

        file_instance, label_instance = sess.run(train_iterator.get_next())

        plt.imshow(np.squeeze(file_instance[2]), cmap='gray')

        model = sequential_conv_model(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),
            n_outputs=1
        )

        model.summary()

        model.fit(
            train_iterator,
            epochs=EPOCHS,
            steps_per_epoch=len(files_train) // BATCH_SIZE,
            validation_data=val_iterator,
            validation_steps=len(files_val) // BATCH_SIZE
        )

        #todo Weitere Metriken und confusion matrix einführen
        #todo Dokumentation von Metriken und loss (numerisch und visualisiert)
        #todo Regularisierung (early stopping, Batch normalization, Dropout)
        #todo Modell speichern
        #todo Testing Skript (model.evaluate(Dataset) und model.predict)

        #todo cross validation?

