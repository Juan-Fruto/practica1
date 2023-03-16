from converter import separator
from quadraticError import degreeChoice
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import operator

examples = [{'type': 'message', 'pattern': None, 'channel': 'canal-10', 'data': '{"id": "temp", "x": [0.8, 2, 2.8, 3.8, 4.5], "y": [1.4, 3.7, 5.5, 7.7, 9.6]}'}]

#1. Obtención del conjunto de datos
data = separator(examples[0]['data'])
print(data)

x = np.array(data['x'])
y = np.array(data['y'])
print(x, y)

#2. Visualización inicial de los datos
plt.plot( x, y, 'ro')
plt.savefig('punto1.png')
plt.show()

#3. Preparación de los datos
x= x[:, np.newaxis]
y= y[:, np.newaxis]

model1 = LinearRegression()
model1.fit(x,y)

#4. Primera prueba (aplicar un modelo de regresión lineal con sklearn)
y_predict = model1.predict(x)

#5. Graficar la regresión obtenida con los datos obtenidos
plt.scatter(x, y)
plt.plot(x, y_predict, color='y')
plt.savefig('punto2.png')
plt.show()

#6. Regresión Polinomial
"""Haga una lectura y un resumen claro del proceso de regresión polinomial,
enumerando las partes más influyentes de este proceso."""

polynomial_features= PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(x)

model = LinearRegression()
model.fit(x_poly,y)
y_predict = model.predict(x_poly)

plt.scatter(x, y)
plt.plot(x, y_predict, color='b')
plt.savefig('punto6.png')
plt.show()

#7. Ajuste excesivo e insuficiente
def printQE(degree):
    rmselist = np.zeros(100)
    x_p_list = [None]*100
    y_poly_pred_P_list=[None]*100

    for i in np.arange(1, degree):
        rmselist[i-1] ,x_p_list[i-1],y_poly_pred_P_list[i-1]= degreeChoice(x,y,i)

    print(np.mean(rmselist))

    plt.plot(np.arange(1, 101), rmselist, color='r')
    plt.savefig(f"degree{degree}.png")
    plt.show()
printQE(4)

"""Que se observa y que se puede concluir ???"""

"""al tener un promedio cercano al cero se puede decir que la linea creada por el modelo de regresión
polinomica es bastante acorde a los datos ya existentes
"""

#8. Visualización de las regresiones lineales, grado 2, 4, 16, 32 y 64
printQE(4)
printQE(8)
printQE(16)
printQE(32)
printQE(64)

