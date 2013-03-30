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
        r'CZ\negative-web-128x128'
    ]
)

PEDESTRIANS = map(lambda x: join(PEDESTRIANS_BASE, x),
    [
        r'CVC-02-Classification\train\positive-rescaled-64x128',
        r'CVC-02-Classification\train\negative-road-rescaled-64x128',
        r'CVC-02-Classification\train\negative-sliding-rescaled-64x128',
        r'CVC-02-Classification\train\negative-false-64x128',
        r'CVC-02-Classification\test\positive-rescaled-64x128',
        r'CVC-02-Classification\test\negative-road-rescaled-64x128',
        r'CVC-02-Classification\test\negative-sliding-rescaled-64x128',
        r'INRIAPerson\positive-64x128',
        r'INRIAPerson\negative-64x128',
    ]
)

CARS_FN = r'cars.yml'
PEDESTRIANS_FN = r'pedestrians.yml'
PED_SCORES_FN = r'ped_scores.npz'
CAR_SCORES_FN = r'car_scores.npz'