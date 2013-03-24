#!/usr/bin/env python
__author__ = 'Calvin'

from common import *
from random import randint

def cuts(src_dir, size):
    for img in os.listdir(src_dir):
        name = join(src_dir, img)
        cut_image(name, name, 0, 0, size)


def rand_cut(src_dir, dst_dir, name, fmt):
    no = 1
    for img in os.listdir(src_dir):
        x1 = randint(0,106)
        fo = join(src_dir, img)
        fn = '%s\%s_%04d.%s' % (dst_dir, name, no, fmt)
        cut_image(fo, fn, 0, x1, (128,64))
        no += 1
        if x1+64 < 106:
            x2 = randint(x1+65, 106)
            fn = '%s\%s_%04d.%s' % (dst_dir, name, no, fmt)
            cut_image(fo, fn, 0, x2, (128,64))
            no += 1


class FastCut():
    def __init__(self, samples, scale, dst=None, name='frame'):
        self.samples = samples
        self.name = name
        cv2.namedWindow(self.name)
        self.selector = RectSelector(self.name, self.on_rect, scale=scale)
        self.paused = False
        self.dst = dst
        self.num = 0
        self.size = scale
        self.img = None
    def on_rect(self, rect):
        self.rect = rect
        self.cut()
        self.paused = False
    def cut(self):
        dst_name = self.img
        self.num += 1
        if self.dst and os.path.isdir(self.dst):
            dst_name = '%s\%s_%04d.%s' % (self.dst, self.name, self.num, self.img.split('.')[-1])
        im = cv2.imread(self.img)
        r = self.rect
        im = im[r[1]:r[3], r[0]:r[2]]
        rs = cv2.resize(im, self.size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst_name, rs)
        print '%d > scope: %s, %s' % (self.num, self.rect, self.img)
    def run(self):
        it = iter(self.samples)
        while True:
            if not self.paused:
                self.paused = True
                try:
                    self.img = it.next()
                except StopIteration:
                    print 'Finished'
                    exit()
            vis = cv2.imread(self.img)
            self.selector.draw(vis)
            cv2.imshow(self.name, vis)
            ch = cv2.waitKey(10)
            if ch == ord(' '):
                self.paused = False
                continue
            if ch == 27:
                break

def yaml_adapter(fn):
    with open(fn) as f:
        f.seek(11)
        cnt = f.read()
        cnt = re.sub(':\S', lambda m: ': '+m.group(0)[-1], cnt, count=3)
        rn = fn.split('.')[0]+'.yaml'
        tmp = open(rn, 'w')
        try:
            tmp.write(cnt)
        finally:
            tmp.close()
    return rn
        # lines = lines.replace(':', ': ')
        # print 'replace finished'
        # dataMap = yaml.load(lines)
        # print dataMap.get('my_svm').get('sv_total')

def get_array(fn):
    with open(fn) as f:
        ym = yaml.load_all(f.read())
        print 'load finished'
        for key in ym:
            print ym


class Array_parser(object):
    """Get support_vectors, rho, alpha from SVM train result file"""
    def __init__(self):
        self.keywords = 'var_count', 'sv_total', 'support_vectors', 'rho', 'alpha'
        self.var_count = self.sv_total = self.rho = 0
        self.support_vectors = []
        self.alpha = []
        self.pattern = re.compile(r'[\+|\-]?\d+[\d\.\-\+e]+')
        self.line = None
        self.sv_num = self.sv_dm = 0
    def parse_int(self):
        if self.line.startswith(self.keywords[0]):
            self.var_count = int(self.line.split(':')[-1].strip())
        elif self.line.startswith(self.keywords[1]):
            self.sv_total = int(self.line.split(':')[-1].strip())
        elif self.line.startswith(self.keywords[3]):
            self.rho = float(self.line.split(':')[-1].strip())
    def parse_array(self):
        ls = self.pattern.findall(self.line)
        if ls:
            self.support_vectors += map((lambda x: float(x)), ls)
            self.sv_num -= len(ls)
        return True
    def parse_alpha(self):
        ls = self.pattern.findall(self.line)
        if ls:
            self.alpha += map((lambda x: float(x)), ls)
            self.sv_dm -= len(ls)
        return True
    @staticmethod
    def run(fn):
        parser = Array_parser()
        print parser.__doc__
        flag = False
        with open(fn) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = parser.line = line.strip()
                if line.startswith(parser.keywords[0]) or \
                        line.startswith(parser.keywords[1]) or \
                        line.startswith(parser.keywords[3]):
                    parser.parse_int()
                elif line.startswith(parser.keywords[2]):
                    parser.sv_num = parser.sv_total*parser.var_count
                elif parser.sv_num:
                    parser.parse_array()
                elif line.startswith(parser.keywords[4]):
                    parser.sv_dm = parser.sv_total
                    parser.parse_alpha()
                elif parser.sv_dm:
                    parser.parse_alpha()
        parser.support_vectors = np.array(parser.support_vectors).reshape((parser.sv_total, -1))
        parser.alpha = np.array(parser.alpha)
        res = parser.var_count, parser.sv_total, \
              parser.support_vectors,\
              parser.rho, \
              parser.alpha
        return dict(zip(parser.keywords, res))

def save_detector(fn, dn='cz_detector'):
    sv = Array_parser.run(fn)
    rs = np.dot(sv.get('alpha'),sv.get('support_vectors'))
    rs = np.append(rs, np.array([sv.get('rho')]), 0).reshape((-1, 1))
    np.save(dn, rs)
    return dn

if __name__ == '__main__':
    src_dir = r'E:\FavoriteVideo\Images\Original'
    dst_dir = r'E:\FavoriteVideo\Images\org-800x600'
    size = (800, 600)
    fmt = 'jpg'
    name = 'ssd'
    #resize_scale(src_dir, dst_dir, size, fmt, name, start=1)

    #cut_image('street_0001.jpg', 'street_0002.jpg', 0 , 0, (128,128))

    src_dir = r'C:\Users\Calvin\PycharmProjects\machine\datasets\cars\test\positive-128x128'
    #cuts(src_dir, (128,128))

    src_dir = r'C:\Users\Calvin\PycharmProjects\machine\datasets\pedestrians\CVC-Virtual-Pedestrian\negative-171x128'
    dst_dir = r'C:\Users\Calvin\PycharmProjects\machine\datasets\pedestrians\CVC-Virtual-Pedestrian\negative-64x128'
    name = 'background'
    fmt = 'png'
    #rand_cut(src_dir, dst_dir, name, fmt)

    src = get_images(r'E:\FavoriteVideo\Images\org-800x600')
    dst = r'C:\Users\Calvin\PycharmProjects\machine\datasets\cars\CZ\postive'
    #FastCut(src, (128,128), dst=dst).run()

    #print yaml_adapter('pedestrians.yml')
    fn = 'pedestrians.yml'
    #get_array(fn)

    #print save_detector(fn)


    cv2.waitKey()