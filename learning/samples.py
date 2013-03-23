#!/usr/bin/env python
__author__ = 'Calvin'

from os.path import join


BASE = r'..\datasets'
CARS_BASE = join(BASE, 'cars')
PEDESTRIANS_BASE = join(BASE, 'pedestrians')


CARS = map(lambda x: join(CARS_BASE, x),
    [
        r'CZ\positive-128x128',
        r'CZ\negative-128x128',
        r'CBCL-positive-128x128',
        r'CZ\negative-web-128x128',
        r'CZ\positive-web-192x128'
    ]
)

PEDESTRIANS = map(lambda x: join(PEDESTRIANS_BASE, x),
    [
        r'CVC-02-Classification\train\positive-rescaled-64x128',
        r'CVC-02-Classification\train\negative-road-rescaled-64x128',
        r'CVC-02-Classification\test\positive-rescaled-64x128',
        r'CVC-02-Classification\test\negative-road-rescaled-64x128'
    ]
)

CARS_FN = r'cars.yml'
PEDESTRIANS_FN = r'pedestrians.yml'
SVM_SCORES_FN = r'svm_scores.npz'