
from keras import backend as K

def Jacard_coefficient(y_true, y_pred):
    Ty= K.flatten(y_true)
    Py = K.flatten(y_pred)
    intersection = K.sum(Ty * Py)
    return (intersection + 1.0) / (K.sum(Ty) + K.sum(Py) - intersection + 1.0)