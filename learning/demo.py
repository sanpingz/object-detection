#!/usr/bin/env python
__author__ = 'Calvin'

from common import *

def cuts(src_dir, size):
    for img in os.listdir(src_dir):
        name = join(src_dir, img)
        cut_image(name, name, 0, 0, size)


def rand_cut(src_dir, dst_dir, name, fmt, scale, size=10):
    no = 0
    def rant_pair(scale, num=size):
        x0 = np.random.randint(scale[1], size=num)
        y0 = np.random.randint(scale[0], size=num)
        return list(zip(x0,y0))
    def minus(t1,t2):
        return abs(t1[0]-t2[0]), abs(t1[1]-t2[1])
    for img in os.listdir(src_dir):
        fo = join(src_dir, img)
        no += 1
        for x1,y1 in rant_pair(minus(cv2.imread(fo,0).shape, scale), num=size):
            fn = '%s\%s_%04d.%s' % (dst_dir, name, no, fmt)
            cut_image(fo, fn, x1, y1, scale)
            no += 1


def remap(src_dir, dst_dir, shape, name, fmt):
    no = 1
    map_x = np.zeros(shape, np.float32)
    map_y = np.zeros(shape, np.float32)
    for x in range(shape[1]):
        for y in range(shape[0]):
            map_x[y,x] = shape[1]-1-x
            map_y[y,x] = y
    for img in os.listdir(src_dir):
        fo = join(src_dir, img)
        fn = '%s\%s_%04d.mirror.%s' % (dst_dir, name, no, fmt)
        no += 1
        im = cv2.remap(cv2.imread(fo), map_x, map_y, interpolation=cv2.INTER_AREA)
        cv2.imwrite(fn, im)


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
              parser.support_vectors, \
              parser.rho, \
              parser.alpha
        return dict(zip(parser.keywords, res))


def save_detector(fn, dn='cz_detector'):
    sv = Array_parser.run(fn)
    rs = np.dot(sv.get('alpha'),sv.get('support_vectors'))
    rs = np.append(rs, np.array([sv.get('rho')]), 0).reshape((-1, 1))
    np.save(dn, rs)
    return dn

def find_eggs(src_dir, size):
    def is_equal(t1,t2):
        if t1[0] == t2[0] and t1[1] == t2[1]:
            return True
    for img in os.listdir(src_dir):
        name = join(src_dir, img)
        if not is_equal(size, cv2.imread(name,0).shape):
            print name


def fixed_cut(src, dst, o, size):
    for img in os.listdir(src):
        fs = join(src, img)
        fd = join(dst, img)
        im = cv2.imread(fs)
        cv2.imwrite(fd, im[o[0]:o[0]+size[0], o[1]:o[1]+size[1]])


if __name__ == '__main__':
    # src_dir = r'E:\FavoriteVideo\Images\Original'
    # dst_dir = r'E:\FavoriteVideo\Images\org-800x600'
    # size = (800, 600)
    # fmt = 'jpg'
    # name = 'ssd'
    #resize_scale(src_dir, dst_dir, size, fmt, name, start=1)

    #cut_image('street_0001.jpg', 'street_0002.jpg', 0 , 0, (128,128))

    # src_dir = r'C:\Users\Calvin\PycharmProjects\machine\datasets\cars\test\positive-128x128'
    #cuts(src_dir, (128,128))

    # src_dir = r'C:\Users\Calvin\PycharmProjects\machine\datasets\pedestrians\INRIAPerson\negative'
    # dst_dir = r'C:\Users\Calvin\PycharmProjects\machine\datasets\pedestrians\INRIAPerson\negative-96x160'
    # name = 'neg'
    # fmt = 'png'
    # rand_cut(src_dir, dst_dir, name, fmt, (160,96))

    # src = get_images(r'E:\FavoriteVideo\Images\org-800x600')
    # dst = r'C:\Users\Calvin\PycharmProjects\machine\datasets\cars\CZ\postive'
    # FastCut(src, (128,128), dst=dst).run()

    #print yaml_adapter('pedestrians.yml')
    fn = 'pedestrians.yml'
    #get_array(fn)

    # print save_detector(fn)

    fc = 'cars.yml'
    img = [
        r'temp\frame_ped.png',
        r'temp\frame_no.png',
        r'temp\frame_no1.png',
        r'temp\ssd_0282.jpg',
        r'temp\ssd_2217.jpg',
        r'temp\person_236_min.png',
        r'temp\person_236.png',
        r'temp\person_265.png',
        r'temp\201939.png',
        r'temp\21376.jpg',
        r'temp\frame_ped_min.png'
    ]
    car = [
        r'temp\frame0000.png',
        r'temp\frame0073.png',
        r'temp\frame0041.png'
    ]
    svm = SVM()
    svm.load(fn)
    hog = HOG(_winSize=(64,128))
    im = cv2.imread(img[-3])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # print Detector(svm, hog).detect(gray, debug=True)
    founds = Detector(svm, hog).detectMultiScale(gray, debug=False, fit=True, win_stride=(16,16),scale=0.95)
    # founds = Detector(svm, hog).detectMultiScale(gray, debug=False, fit=True, resize=(0.72,0.72), group_threshold=0.5)
    Detector.draw_rectangle(im, founds, thickness=2)
    # print '%d found' % len(founds)
    cv2.putText(im, '%d found'%len(founds), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 250), thickness = 1)
    cv2.imshow('demo', im)

    # src_dir=r'C:\Users\Calvin\PycharmProjects\machine\datasets\cars\CZ\positive-128x128'
    # dst_dir=r'C:\Users\Calvin\PycharmProjects\machine\datasets\cars\CZ\positive-128x128'
    # remap(src_dir, dst_dir, (128,128), 'frame', 'jpg')\

    # find_eggs(r'C:\Users\Calvin\PycharmProjects\machine\datasets\pedestrians\INRIAPerson\positive-64x128', (128, 64))

    # src_dir = r'C:\Users\Calvin\PycharmProjects\machine\datasets\pedestrians\INRIAPerson\negative-96x160'
    # dst_dir = r'C:\Users\Calvin\PycharmProjects\machine\datasets\pedestrians\INRIAPerson\negative-64x128'
    # fixed_cut(src_dir, dst_dir, (16,16), (128,64))

    cv2.waitKey()