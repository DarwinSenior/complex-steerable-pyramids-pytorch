import numpy as np
import torch
import util


class Steerable(object):
    def __init__(self, size, height=5, nbands=4, scale_factor=2., device=None):
        self.height = height
        self.nbands = nbands
        self.sizes = np.ceil(np.array([
            (size[0]*scale_factor**(-i), size[1]*scale_factor**(-i))
            for i in range(self.height)])).astype(np.int)
        self.scale_factor = scale_factor
        self.device = device or torch.device('cpu')
        self.build_mask()

    def tensor(self, i):
        return torch.tensor(i[None,:,:,None], dtype=torch.float, device=self.device)

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
        Ycosn = np.sqrt(2*const)*np.power(np.cos(Xcosn), order)*(np.abs(alpha)<np.pi/2)
        YIrcos = np.sqrt(np.abs(1-Yrcos*Yrcos))
        for size in self.sizes[1:]:
            Xrcos = Xrcos - np.log2(self.scale_factor)
            self.himasks.append(self.tensor(util.pointOp(log_rad, Yrcos, Xrcos)))
            anglemasks = [self.tensor(util.pointOp(
                angle, Ycosn, Xcosn+np.pi*b/self.nbands))
                          for b in range(self.nbands)]
            self.anglemasks.append(anglemasks)
            angle = util.downsample(angle, size)
            log_rad = util.downsample(log_rad, size)
            self.lomasks.append(self.tensor(util.pointOp(log_rad, YIrcos, Xrcos)))

    def decompose(self, im):
        M, N = self.sizes[0]
        imdft = util.rfft(im)
        hiband = util.irfft(imdft * self.hi0mask)
        order = self.nbands - 1

        imdft = imdft * self.lo0mask
        orients = []
        for hi, angs, lo, sz in zip(
            self.himasks, self.anglemasks, self.lomasks, self.sizes[1:]):

            band = util.rotate(imdft, -order) * hi
            orients.append([util.ifft(band*ang) for ang in angs])
            imdft = util.resize(imdft, sz) * lo
        loband = util.irfft(imdft)
        return hiband, loband, orients

    def compose(self, hiband, loband, orients):
        dft = util.rfft(loband)
        order = self.nbands - 1
        for hi, angs, lo, sz, os in reversed(list(zip(
            self.himasks, self.anglemasks, self.lomasks, self.sizes[:-1], orients))):

            dft = util.resize(dft*lo, sz)
            dft += util.rotate(sum(hi*ang*util.fft(o)
                       for o, ang in zip(os, angs)), order)
        dft = dft * self.lo0mask + util.rfft(hiband) * self.hi0mask
        return util.irfft(dft)

def test():
    from skimage import io, transform
    import matplotlib.pyplot as plt
    im = io.imread('assets/lena.jpg', as_gray=True)
    im = torch.tensor(im).float().unsqueeze(0)
    pry = Steerable(im.size()[-2:], height=5)
    hiband, loband, orients = pry.decompose(im)
    im_rec = pry.compose(hiband, loband, orients)
    import ipdb; ipdb.set_trace()
    plt.figure()
    plt.subplot(221)
    plt.title('original')
    plt.imshow(im.squeeze())
    plt.subplot(222)
    plt.title('reconstructed')
    plt.imshow(im_rec.squeeze())
    plt.subplot(223)
    plt.title('difference')
    plt.imshow((im-im_rec).squeeze())
    plt.show()



if __name__ == "__main__":
    test()
