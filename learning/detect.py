#!/usr/bin/env python
__author__ = 'Calvin'

import numpy as np
import cv2

from samples import *
from common import *

help_message = '''
USAGE: peopledetect.py <image_names> ...

Press any key to continue, ESC to stop.
'''

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


if __name__ == '__main__':
    import sys
    from glob import glob
    import itertools as it

    print help_message

    hog = cv2.HOGDescriptor()
    cz = np.load('cz_detector.npy')
    detector = cv2.HOGDescriptor_getDefaultPeopleDetector(), cz

    hog.setSVMDetector( detector[1] )
    tmp = r'temp\frame_ped.png', r'temp\frame_no.png'
    sys.argv = ['detect',tmp[0], tmp[1]]

    for fn in it.chain(*map(glob, sys.argv[1:])):
        print fn, ' - ',
        try:
            img = cv2.imread(fn)
        except:
            print 'loading error'
            continue

        found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
        draw_detections(img, found)
        draw_detections(img, found_filtered, 3)
        print '%d (%d) found' % (len(found_filtered), len(found))
        cv2.imshow('img', img)
        ch = 0xFF & cv2.waitKey()
        if ch == 27:
            break
    cv2.destroyAllWindows()