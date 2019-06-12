from scipy.misc import factorial
import numpy as np
import torch

def base(m, n):
    x = np.linspace(-(m // 2)/(m / 2), (m // 2)/(m / 2) - (1 - m % 2)*2/m , num = m)
    y = np.linspace(-(n // 2)/(n / 2), (n // 2)/(n / 2) - (1 - n % 2)*2/n , num = n)
    xv, yv = np.meshgrid(y, x)
    angle = np.arctan2(yv, xv)
    rad = np.sqrt(xv**2 + yv**2)
    rad[m//2][n//2] = rad[m//2][n//2 - 1]
    log_rad = np.log2(rad)
    return log_rad, angle


def rotate(x, k):
    """rotate based on orientation"""
    k = k % 4
    a, b = x.unbind(-1)
    if k == 1:
        a, b = -b, a
    elif k == 2:
        a, b = -a, -b
    elif k == 3:
        a, b = b, -a
    return torch.stack((a, b), -1)

def fft(x):
    *_, h, w, c = x.size()
    xdft = torch.fft(x, 2)
    xdft = torch.roll(xdft, (h//2, w//2), (-3, -2))
    return xdft

def rfft(x):
    *_, h, w = x.size()
    xdft = torch.rfft(x, 2, onesided=False)
    xdft = torch.roll(xdft, (h//2, w//2), (-3, -2))
    return xdft

def ifft(xdft):
    *_, h, w, c = xdft.size()
    x = torch.roll(xdft, (h-h//2, w-w//2), (-3, -2))
    x = torch.ifft(x, 2)
    return x

def irfft(xdft):
    *_, h, w, c = xdft.size()
    x = torch.roll(xdft, (h-h//2, w-w//2), (-3, -2))
    x = torch.irfft(x, 2, onesided=False)
    return x

def pointOp(im, Y, X):
    out = np.interp(im.flatten(), X, Y)
    return np.reshape(out, im.shape)

def rcosFn(width, position):
    N = 256
    X = np.pi * np.array(range(-N-1, 2)) / 2 / N
    Y = np.cos(X) ** 2
    Y[0] = Y[1]
    Y[N+2] = Y[N+1]
    X = position + 2*width/np.pi*(X + np.pi/4)
    return X, Y

def cresize(x, size):
    pass

def resize(x, size):
    pass
