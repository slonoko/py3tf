import random
import string
from tensorflow.keras.models import load_model
import cherrypy
import numpy as np


class MyWebService(object):
    @cherrypy.expose
    def index(self):
        return open("./canvas.html")

    @cherrypy.expose
    def predict(self, nb_field=""):
        model = load_model('mnist_model.h5')
        arr_test = nb_field.split(',')
        __input = np.array(arr_test, dtype=float)
        __input = np.expand_dims(__input, axis=0)
        y_test_pred = model.predict_classes(__input, verbose=0)

        return str(y_test_pred)

if __name__ == '__main__':
    conf = {
        '/': {
            'tools.sessions.on': True
        }
    }
    cherrypy.quickstart(MyWebService(), '/', conf)
