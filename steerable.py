import numpy as np
import torch
import util


class Steerable(object):
    def __init__(self, size, height=5, nbands=4, scale_factor=2., device=None):
        self.height = height
        self.sizes = np.ceil(np.array([
            (M*scale_factor**(-i), N*scale_factor**(-i))
            for i in range(self.height)])).astype(np.int)
        self.scale_factor = scale_factor
        self.device = device or torch.device
        self.build_mask()

    def tensor(self, i):
        return torch.tensor(i[None,:,:,None], dtype=float, device=self.device)

    def build_mask(self):
        M, N = self.sizes[0]
        log_rad, angle = util.base(M, N)
        Xrcos, Yrcos = util.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(1-Yrcos*Yrcos)

        self.lo0mask = self.tensor(util.pointOp(log_rad, YIrcos, Xrcos))
        self.hi0mask = self.tensor(util.pointOp(log_rad, Yrcos, Xrcos))

        self.himasks = []
        self.lomasks = []
        self.anglemasks = []

        lutsize = 1024
        Xcosn = np.pi * np.arange(-(2*lutsize+1), (lutsize+2)) / lutsize
        order = self.nbands - 1
        const = np.power(2, 2*order) * np.square(util.factorial(order)) / (self.nbands * util.factorial(2*order))
        alpha = (Xcosn + np.pi) % (2*np.pi) - np.pi
        Ycosn = 2*np.sqrt(const)*np.power(np.cos(Xcosn), order)*(np.abs(alpha)<np.pi/2)
        for size in self.sizes[1:]:
            Xrcos = Xrcos - np.log2(self.scale_factor)
            himasks.append(self.tensor(log_rad, Yrcos, Xrcos))
            anglemasks = [self.tensor(util.pointOp(
                angle, Ycosn, Xcosn+np.pi*b/self.nbands))
                          for b in self.nbands]
            self.anglemasks.append(anglemasks)
            angle = util.resize(angle, size)
            log_rad = util.resize(log_rad, size)
            self.lomasks.append(self.tensor(util.pointOp(log_rad, YIrcos, Xrcos)))

    def decompose(self, im):
        M, N = self.sizes[0]
        imdft = util.rfft(im)
        hiband = util.irfft(imdft * self.hi0mask)
        order = self.nbands - 1

        imdft = imdft * self.lo0mask
        orients = []
        for hi, angs, lo, sz in zip(
            self.himasks, self.anglemasks, self.lomasks, self.size[1:]):

            band = util.rotate(imdft, -order) * hi
            orients.append([util.ifft(band*ang for ang in angs)])
            imdft = util.cresize(imdft, sz) * lo
        loband = util.irfft(imdft)
        return hiband, loband, orients

    def compose(self, hiband, loband, orients):
        dft = util.irfft(loband)
        order = self.nbands - 1
        for hi, angs, lo, sz, os in reversed(list(zip(
            self.himasks, self.anglemasks, self.lomasks, self.size[:-1], orients))):

            dft = util.cresize(dft*lo, sz)
            dft += sum(util.rotate(hi*ang*util.fft(o), order)
                       for o, ang in zip(orients, angs))
        dft = dft * self.lo0mask + util.irfft(hiband) * self.hi0mask
        return util.irfft(dft)
