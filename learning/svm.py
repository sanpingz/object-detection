#!/usr/bin/env python
"""Using SVM + HOG train and test pedestrians, cars"""
__author__ = 'Calvin'

from common import *
from samples import *
from multiprocessing.pool import ThreadPool

@timer
def train(model, feature, fn, pos, neg):
    print 'Training ...'
    model.train(*preprocess(pos, neg, feature))
    model.save(fn)
    print '''Training finished
    Positive number: %s
    Negative number: %s
    Training time: %.4fs''' % (len(pos), len(neg), model.timer)

@timer
def test(model, feature, fn, pos=None, neg=None):
    model.load(fn)
    if not (pos and neg):
        print 'Samples are required'
        exit()
    if pos:
        pos_mat = model.predict(feature.process(pos))
        pos_num = len(pos)
        pos_suc = pos_mat.sum(axis=0)
        pos_res = pos_suc/pos_num*100.0
        print '''Positive testing
        Sample number: %d
        Success: %d
        Testing success rate: %.4f%%''' % (pos_num, pos_suc, pos_res)
    if neg:
        neg_mat = model.predict(feature.process(neg))
        neg_num = len(neg)
        neg_suc = neg_num - neg_mat.sum(axis=0)
        neg_res = neg_suc/neg_num*100.0
        print '''Negative testing
        Sample number: %d
        Success: %d
        Testing success rate: %.4f%%''' % (neg_num, neg_suc, neg_res)


def execute(func, *args):
    func(*args)


class Best_Params(object):
    """ params: pos, neg, feature """
    def __init__(self, *params):
        self._samples, self._labels = self.preprocess(*params)

    def preprocess(self, *params):
        return preprocess(*params)

    def get_dataset(self):
        return self._samples, self._labels

    def run_jobs(self, f, jobs):
        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        return pool.imap_unordered(f, jobs)

    @timer
    def adjust_SVM(self, fn):
        Cs = np.logspace(0, 10, 15, base=2)
        gammas = np.logspace(-7, 4, 15, base=2)
        scores = np.zeros((len(Cs), len(gammas)))
        scores[:] = np.nan

        print 'adjusting SVM (may take a long time) ...'
        def f(job):
            i, j = job
            samples, labels = self.get_dataset()
            params = dict(C = Cs[i], gamma=gammas[j])
            score = cross_validate(SVM, params, samples, labels)
            return i, j, score

        ires = self.run_jobs(f, np.ndindex(*scores.shape))
        for count, (i, j, score) in enumerate(ires):
            scores[i, j] = score
            print '%d / %d (best error: %.2f %%, last: %.2f %%)' % (count+1, scores.size, np.nanmin(scores)*100, score*100)
        print scores

        print 'writing score table to "%s"' % fn
        np.savez(fn, scores=scores, Cs=Cs, gammas=gammas)

        i, j = np.unravel_index(scores.argmin(), scores.shape)
        best_params = dict(C = Cs[i], gamma=gammas[j])
        print 'best params:', best_params
        print 'best error: %.2f %%' % (scores.min()*100)
        return best_params

if __name__ == '__main__':
    print __doc__
    ped_pos = (
        get_images(PEDESTRIANS[0]),
        get_images(PEDESTRIANS[2]),
    )
    ped_neg = (
        get_images(PEDESTRIANS[1]),
        get_images(PEDESTRIANS[3]),
    )
    car_pos = (
        get_images(CARS[0]),
        get_images(CARS[2])
    )
    car_neg = (
        get_images(CARS[1]),
        get_images(CARS[3])
    )
    fn = [ PEDESTRIANS_FN, CARS_FN ]
    svm = SVM( kernel_type = cv2.SVM_LINEAR,    # cv2.SVM_RBF cv2.SVM_POLY cv2.SVM_SIGMOID
               svm_type = cv2.SVM_C_SVC,
               C = 4.416358,
               gamma = 0.0078125
    )
    hog = HOG( _winSize = (128,128),    # (128, 128), (64, 128)
               _blockSize = (16,16),
               _blockStride = (8,8),
               _cellSize = (8,8),
               _nbins = 9
    )

    # train(model, feature, fn, pos, neg)
    # test(model, feature, fn, pos=None, neg=None)

    # execute(train, svm, hog, fn[0], ped_pos[0], ped_neg[0])
    # execute(test, SVM(), hog, fn[0], ped_pos[1], ped_neg[1])

    execute(train, svm, hog, fn[1], car_pos[0], car_neg[0])
    execute(test, SVM(), hog, fn[1], car_pos[1], car_neg[1])

    # best = Best_Params(car_pos[0], car_neg[0], hog)
    # best.adjust_SVM(CAR_SCORES_FN)

    cv2.waitKey()
    cv2.destroyAllWindows()