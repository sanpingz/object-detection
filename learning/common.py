#!/usr/bin/env python
__author__ = 'Calvin'

import os, time, glob
from os.path import join
import cv2
import numpy as np
from numpy.linalg import norm
import yaml, re

def get_images(path):
    ls = []
    if os.path.isdir(path) and os.listdir(path):
        ls = os.listdir(path)
    else:
        print 'No images found at %s' % path
        exit()
    return ls and map(lambda x: join(path, x), ls)


def get_mat(image, size=None):
    im = cv2.imread(image, 0)
    if size:
        im = cv2.resize(im, size, interpolation=cv2.INTER_CUBIC)
    return im


def timer(func):
    def newFunc(*args, **args2):
        start = time.time()
        back = func(*args, **args2)
        print "Timer: %.4fs" % (time.time()-start)
        return back
    return newFunc


def resize_scale(src_dir, dst_dir, size, fmt, name, start=1):
    if os.path.isdir(src_dir) and os.path.isdir(dst_dir) and size and fmt and name:
        src = glob.glob(join(src_dir, r'*.'+fmt))
        no = start
        for img in src:
            im = cv2.imread(img)
            rs = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR)
            fm = '%s\%s_%04d.%s' % (dst_dir, name, no, fmt)
            no += 1
            cv2.imwrite(fm, rs)


def cut_image(src, dst, x, y, (w, h)):
    im = cv2.imread(src)
    if im.shape[0]<w+x or im.shape[1]<h+y:
        r = max(float(w+x)/im.shape[0], float(h+y)/im.shape[1])
        im = cv2.resize(im, (int(im.shape[0]*r+0.5), int(im.shape[1]*r+0.5)), interpolation=cv2.INTER_AREA)
    rs = im[x:w+x, y:h+y]
    cv2.imwrite(dst,rs)


class RectSelector:
    def __init__(self, win, callback, scale=None):
        self.win = win
        self.callback = callback
        cv2.setMouseCallback(win, self.on_mouse)
        self.drag_start = None
        self.drag_rect = None
        self.scale = scale
    def on_mouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                if isinstance(self.scale, tuple):
                    y = (x-xo)*self.scale[1]/self.scale[0] + yo
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)
    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True
    @property
    def dragging(self):
        return self.drag_rect is not None


class StatModel(object):
    """parent class - starting point to add abstraction"""
    def load(self, fn):
        if os.path.isfile(fn):
            print 'loading "%s" ...' % fn
            self.model.load(fn)
            print '"%s" loaded successfully' % fn
        else:
            print '"%s" loading failed' % fn
            exit()
    def save(self, fn):
        try:
            self.model.save(fn)
            print '"%s" saved successfully' % fn
        except Exception:
            print '"%s" saving failed' % fn


class Feature(object):
    def process(self): pass


class SVM(StatModel):
    """wrapper for OpenCV SVM algorithm"""
    def __init__(self, **params):
        self.model = cv2.SVM()
        self.params = params
    def train(self, samples, responses):
        """setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_RBF,
                       svm_type = cv2.SVM_C_SVC,
                       C = 1
        )"""
        #print map((lambda (key, val): '%s: %s' % (key, val)), self.params.items())
        timer = time.time()
        self.model.train(samples, responses, params=self.params)
        self.timer = time.time() - timer
    def predict(self, samples):
        return self.model.predict_all(samples).ravel()
        #return np.float32([self.model.predict(s) for s in samples])


class HOG(Feature):
    """_winSize, _blockSize, _blockStride, _cellSize, _nbins
    cv2.HOGDescriptor((128,64), (16,16), (8,8), (8,8), 9)
    hog_num: ((64-16)/8+1)*((128-16)/8+1)*9*4"""
    def __init__(self, **params):
        self.hog = cv2.HOGDescriptor(**params)
        self.num = params and ((params['_winSize'][0]-params['_blockSize'][0])/params['_blockStride'][0]+1)* \
                              ((params['_winSize'][1]-params['_blockSize'][1])/params['_blockStride'][1]+1)* \
                              params['_blockSize'][0]*params['_blockSize'][1]/(params['_cellSize'][0]*params['_cellSize'][1])*params['_nbins']
    def process(self, samples, size=None):
        res = np.empty((len(samples), self.num), np.float32)
        num = 0
        for img in samples:
            im = get_mat(img, size=size)
            rs = self.hog.compute(im)
            res[num, :] = rs.reshape(1,-1)
            num += 1
        return res


def responses(pos, neg):
    pos_num = len(pos)
    neg_num = len(neg)
    total = pos_num + neg_num
    labels = np.ones(total, np.int32)
    labels[pos_num:] = np.zeros(neg_num)
    return labels


def preprocess(pos, neg, feature):
    samples = feature.process(pos+neg)
    labels = responses(pos, neg)
    shuffle = np.random.permutation(len(samples))
    return samples[shuffle], labels[shuffle]


def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


def cross_validate(model_class, params, samples, labels, kfold = 3, pool = None):
    n = len(samples)
    folds = np.array_split(np.arange(n), kfold)
    def f(i):
        model = model_class(**params)
        test_idx = folds[i]
        train_idx = list(folds)
        train_idx.pop(i)
        train_idx = np.hstack(train_idx)
        train_samples, train_labels = samples[train_idx], labels[train_idx]
        test_samples, test_labels = samples[test_idx], labels[test_idx]
        model.train(train_samples, train_labels)
        resp = model.predict(test_samples)
        score = (resp != test_labels).mean()
        print ".",
        return score
    if pool is None:
        scores = map(f, xrange(kfold))
    else:
        scores = pool.map(f, xrange(kfold))
    return np.mean(scores)
