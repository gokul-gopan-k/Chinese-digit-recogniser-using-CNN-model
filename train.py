import tensorflow as tf
from data_prepare import get_data
from utility import plot_samples

def create_model_train():
    "Function to create a LSTM model and training"
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(15, activation='softmax')
    ])


    model.compile(optimizer = "adam",
                loss = "sparse_categorical_crossentropy",
                metrics=["accuracy"])
    
    
    x_train,y_train_encoded = get_data(mode = "train")
    x_train,y_train_encoded = get_data(mode = "validation")
    
    history = model.fit(x_train,y_train_encoded,
                        validation_data=(x_train,y_train_encoded)
                        epochs=CONFIG.EPOCHS,batch_size = CONFIG.BATCH_SIZE
                        )
    #plot training loss and accuracy
    plot_train_result(history)
    
    return model
