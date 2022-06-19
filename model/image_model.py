# gets a list of the images as well as the prices
# view one of the images
import numpy as np
from PIL import Image
import requests
from io import BytesIO

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, AveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K


# mix the data
def get_img(data):
    img_list = []
    price_list = []
    data_dict = ()
    for i in range(500):  # len(Airbnb_data['price'])):
        try:
            if i % 100 == 0:
                # gets 6,000 images
                print(int((i / 500) * 100), '% done')  # len(data['price']))*100)
            response = requests.get(data['thumbnail_url'][i])
            img = Image.open(BytesIO(response.content)).resize([224, 224])

            img = np.array(img) / 255.0  # makes imputs [0,1]
            img_list.append(img)
            price_list.append(data.price[i])
        except (KeyError or OSError):
            pass
    return img_list, price_list


def create_cnn_model(train_X, train_y, test_X, test_y):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, activation='relu', padding='SAME', input_shape=(224, 224, 3)))
    model.add(MaxPool2D(pool_size=(3, 3), strides=2, padding='VALID'))
    model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='SAME'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=1, padding='VALID'))
    model.add(Conv2D(128, kernel_size=3, strides=2, activation='relu', padding='SAME'))
    model.add(Dropout(.5))
    model.add(Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='SAME'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(31, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    # trains the model
    history = model.fit(train_X, train_y, epochs=5, validation_split=0.2, batch_size=50)
    pred = model.predict(test_X)
    print(np.sqrt(np.mean((pred - test_y) ** 2)))


def create_InceptionV3_model(train_X, train_y, test_X, test_y):
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = Dropout(.5)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(.5)(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='selu')(x)
    x = Dropout(.5)(x)
    # and output layer
    predictions = Dense(1)(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

    # train the model on the new data for a few epochs
    model.fit(train_X[:200], train_y[:200], steps_per_epoch=10, epochs=3, validation_split=0.2, validation_steps=10)

    pred = model.predict(test_X)
    print(np.sqrt(np.mean((pred - test_y) ** 2)))

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
      print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
      layer.trainable = False
    for layer in model.layers[249:]:
      layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.00001, momentum=0.99), loss='mse', metrics=['mse', 'mae'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit(train_X[:200], train_y[:200], steps_per_epoch=10, epochs=3, validation_split=0.2, validation_steps=10)

    pred = model.predict(test_X)
    print(np.sqrt(np.mean((pred - test_y) ** 2)))
    return model

def run(data):
    Airbnb_data = data.sample(frac=1)

    X, y = get_img(Airbnb_data)
    # split up the training set and test set
    train_X = np.asarray(X[:350])
    train_y = y[:350]
    test_X = np.asarray(X[350:])
    test_y = y[350:]
    # create_cnn_model(train_X, train_y, test_X, test_y)




    model = create_InceptionV3_model(train_X, train_y, test_X, test_y)
    return model
    # for i, layer in enumerate(base_model.layers):
    #   print(i, layer.name)
    #
    # # we chose to train the top 2 inception blocks, i.e. we will freeze
    # # the first 249 layers and unfreeze the rest:
    # for layer in model.layers[:249]:
    #   layer.trainable = False
    # for layer in model.layers[249:]:
    #   layer.trainable = True
    #
    # # we need to recompile the model for these modifications to take effect
    # # we use SGD with a low learning rate
    # from keras.optimizers import SGD
    # model.compile(optimizer=SGD(lr=0.00001, momentum=0.99), loss='mse', metrics=['mse', 'mae'])
    #
    # # we train our model again (this time fine-tuning the top 2 inception blocks
    # # alongside the top Dense layers
    # model.fit(train_X[:200], train_y[:200], steps_per_epoch=10, epochs=3, validation_split=0.2, validation_steps=10)
