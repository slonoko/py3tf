import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import feature_column as fc

numeric_column = fc.numeric_column 
categorical_column_with_vocabulary_list = fc.categorical_column_with_vocabulary_list
gpus = tf.config.experimental.list_physical_devices("GPU")

if len(gpus) > 0:
    print("Using a GPU ...")
    tf.config.experimental.set_memory_growth(gpus[0], True)

featcols = [
    tf.feature_column.numeric_column("area"),
    tf.feature_column.categorical_column_with_vocabulary_list(
        "type", ["bungalow", "apartment"]
    )
]


def train_input_fn():
    features = {
        "area": [1000, 2000, 4000, 1000, 2000, 4000],
        "type": [
            "bungalow",
            "bungalow",
            "house",
            "apartment",
            "apartment",
            "apartment",
        ],
    }
    labels = [500, 1000, 1500, 700, 1300, 1900]
    return features, labels


model = tf.estimator.LinearRegressor(feature_columns=featcols)
model.train(input_fn=train_input_fn, steps=200)


def predict_input_fn():
    features = {"area": [1500, 1800], "type": ["house", "apt"]}
    return features


predictions = model.predict(predict_input_fn)
print(next(predictions))
print(next(predictions))
