import numpy as np
import operator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def degreeChoice(x,y,degree):

    polynomial_features= PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)

    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    salida_axis = operator.itemgetter(0)
    salida_zip = sorted(zip(x,y_poly_pred), key=salida_axis)
    x_p, y_poly_pred_P = zip(*salida_zip)

    return rmse, x_p, y_poly_pred_P
