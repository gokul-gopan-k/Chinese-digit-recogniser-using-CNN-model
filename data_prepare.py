import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utility import plot_samples

def parse_data_from_input_csv(filename):
    "Function extracts data from csv. Function called inside get_data function"
    
    with open(filename) as file:

        reader = csv.reader(file, delimiter=",")
        imgs = []
        labels = []
        next(reader, None)

        for row in reader:
            label = row[-2:-1]
            data = row[:-2]
            img = np.array(data).reshape((64, -1))

            imgs.append(img)
            labels.append(label)

    images = np.array(imgs).astype(float)
    labels = np.array(labels).astype(float)

    return images, labels

def get_data(mode):
    "Function get data from csv file and return processed inputs"
    "Return values as per input is train or validation or test""
    
    data_images, data_labels = parse_data_from_input_csv(CONFIG.data_path_csv)
    
    #train, valid and test split
    x_train, x_test, y_train, y_test = train_test_split(data_images, data_labels, test_size=1 - CONFIG.train_ratio)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=CONFIG.test_ratio/(CONFIG.test_ratio + CONFIG.validation_ratio)) 

    #encode train labels
    en_t = LabelEncoder()
    en_t.fit(y_train)
    y_train_encoded = en_t.transform(y_train)
    
    #encode validation labels 
    en_v= LabelEncoder()
    en_v.fit(y_val)
    y_val_encoded = en_v.transform(y_val)

    plot_samples(x_train, y_train_encoded)
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    
    
    if mode == "train":
        return x_train, y_train_encoded
    elif mode == "Validation":
        return x_val,y_val_encoded
    else:
        return x_test,y_test
