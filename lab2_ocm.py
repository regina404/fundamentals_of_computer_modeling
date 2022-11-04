import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from IPython.display import display


def identification():
    y = np.array([6.0, 8.5, 11, 14.5, 18.5, 23, 28.5, 35, 42, 50.5, 60, 70.5, 82.5, 96, 110.5, 127])
    x = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6])
    z = np.array([5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13])

    ln_x_vals = np.log(x)
    ln_y_vals = np.log(y)
    ln_z_vals = np.log(z)

    ln_x = ln_x_vals.reshape(-1, 1)
    ln_y = ln_y_vals.reshape(-1, 1)
    ln_z = ln_z_vals.reshape(-1, 1)

    features = np.column_stack((ln_x, ln_z))

    model = pl.Pipeline(
        [('pf', sp.PolynomialFeatures(degree=1, include_bias=False)),
         # Генерує всі можливі полінміальні комбінації features степеня <= 1
         ('lr', lm.LinearRegression())]  # Лінійна регресія
    )
    model.fit(features, ln_y)

    ln_y_pred = model.predict(features).reshape(1, -1)[0]  # Значення log(y) які передбачила модель

    y_pred = np.exp(ln_y_pred)  # Значення y які передбачила модель

    intercept = model['lr'].intercept_[0]  # коефіцієнт b0
    coef = model['lr'].coef_[0]  # коефіцієнти b1 та b2

    fig = plt.figure(figsize=plt.figaspect(0.4))

    # графік лінійної функції
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    ax.set_title("y = ln(b0) + b1*ln(x) + b2*ln(z)")

    ax.plot3D(ln_x_vals, ln_z_vals, ln_y_pred, 'gray')  # функція, отримана з моделі
    ax.scatter3D(ln_x_vals, ln_z_vals, ln_y_vals, c='r');  # логарифми точок, які були дані в умові

    # графік початкової функції
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    ax.set_title("y = b0 * (x^b1) * (z^b2)")

    ax.plot3D(x, z, y_pred, 'blue')  # функція, отримана з моделі
    ax.scatter3D(x, z, y, c='r')  # точки, які були дані в умові

    plt.show()

    log_vals_df = pd.DataFrame([ln_x_vals, ln_z_vals, ln_y_vals, ln_y_pred],
                               index=['ln x', 'ln z', 'ln y', 'ln_y_pred'])
    vals_df = pd.DataFrame([x, z, y, y_pred],
                           index=['x', 'z', 'y', 'y_pred'])

    r2 = r2_score(y, y_pred)  # коефіцієнт детермінації

    return coef, intercept, log_vals_df, vals_df, r2


coef, intercept, log_vals_df, vals_df, r2 = identification()

display(log_vals_df)
display(vals_df)

print('Оцінка R2:', r2)

print("\nКоефіцієнти:")
print(f"b0= e^{intercept}")
print("b1=", coef[0])
print("b2=", coef[1])
