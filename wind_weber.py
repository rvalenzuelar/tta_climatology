'''

Raul Valenzuela
raul.valenzuela@colorado.edu

Standard deviation of wind direction
based on:

Weber 1991, Journal of Applied Meteorology


'''

import numpy as np


def sin(x):
    return np.sin(np.radians(~np.isnan(x)))


def cos(x):
    return np.cos(np.radians(~np.isnan(x)))


def var(x):
    return np.nanvar(x)


def mean(x):
    return np.nanmean(x)


def angular_mean(u, v):
    return 270 - (np.arctan2(mean(v), mean(u)) * 180 / np.pi)


def x_mean(u, v):
    th_bar = angular_mean(u, v)
    return mean(u) * cos(th_bar) + mean(v) * sin(th_bar)


def x_var(u, v):
    th_bar = angular_mean(u, v)
    a = var(u) * cos(th_bar) ** 2 - var(v) * sin(th_bar) ** 2
    b = cos(th_bar) ** 2 - sin(th_bar) ** 2
    return a / b


def y_var(u, v):
    th_bar = angular_mean(u, v)
    a = var(v) * cos(th_bar) ** 2 - var(u) * sin(th_bar) ** 2
    b = cos(th_bar) ** 2 - sin(th_bar) ** 2
    return a / b


def a_fun(delta):
    a = 90 + 172.24 * delta + 32.427 * delta ** 2
    b = 1 + 1.6067 * delta + 0.25135 * delta ** 2
    return a / b


def b_fun(delta):
    a = 3366.68 - 1934.33 * delta - 316.342 * delta ** 2
    b = 75.0227 * delta + 1.69087 * delta ** 2
    return a / b


def c_fun(delta):
    return -0.0113449 + 0.57699 * delta


def d_fun(delta):
    a = 188.451 - 49.908 * delta - 1.42925 * delta ** 2
    b = 377.822 * delta + 43.8346 * delta ** 2
    return a / b


def e_fun(delta):
    a = 0.64449 \
        - 0.293313 * delta \
        + 0.329844 * delta ** 2 \
        - 0.172855 * delta ** 3 \
        + 0.0440614 * delta ** 4 \
        - 5.8839e-3 * delta ** 5 \
        + 3.96817e-4 * delta ** 6 \
        - 1.06722e-5 * delta ** 7

    return a


def deltaf(u, v):
    return np.sqrt(x_var(u, v)) / np.sqrt(y_var(u, v))


def gammaf(u, v):
    return x_mean(u, v) / np.sqrt(x_var(u, v))


def angular_stddev1(u, v):

    '''
    Equation 37
    '''

    gamma = gammaf(u, v)
    delta = deltaf(u, v)

    if x_var(u, v) < 0:
        print('Warning: variance of X < 0 -> {}'.format(x_var(u,v)))

    if y_var(u, v) < 0:
        print('Warning: variance of Y < 0 -> {}'.format(y_var(u,v)))

    if gamma < 0 or gamma > 30:
        print('Warning: gamma out of range [0,30] -> {}'.format(gamma))

    if delta < 0.1 or delta > 10:
        print('Warning: delta out of range [0.1,10] -> {}'.format(delta))

    delta_onedeg = 63.7469*gamma**-1.037659

    if delta > delta_onedeg:
        return 0
    else:
        a = a_fun(delta) + b_fun(delta) * gamma \
            + c_fun(delta) * gamma ** 2

        b = 1 + d_fun(delta) * gamma \
            + e_fun(delta) * gamma ** 2

        return a / b

def persistance(u,v):

    vector_mean = np.sqrt(mean(u)**2+mean(v)**2)
    scalar_mean = mean(np.sqrt(u**2+v**2))

    return vector_mean/scalar_mean

def angular_stddev2(u, v):

    '''
    Equation 41
    '''

    delta = np.sqrt(x_var(u, v)) / np.sqrt(y_var(u, v))
    P = persistance(u,v)

    A = (-0.262964 + 50.21289*delta + 124.9263*delta**2)/ \
        (-6.414684e-3 + 0.6601226*delta + delta**2)

    B = (0.6787763 + 0.8351036*delta + 0.311525*delta**2)/ \
        (1.189552 + 1.230715*delta + delta**2)

    return A*(1-P)**B


def angular_stddev3(u, v):

    '''
    Equation 42
    '''

    P = persistance(u,v)

    return 105.75*(1-P)**0.5337