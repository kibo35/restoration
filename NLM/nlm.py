#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

parser = argparse.ArgumentParser(description='Non-local means')
parser.add_argument('input', help='input filename')
parser.add_argument('--smoothing_strength', '-s', default=1, type=float,
                    help='smoothing strength a.k.a parameter h of non-local means.')
parser.add_argument('--search_window_size', '-w', default=21, type=int,
                    help='search window size')
parser.add_argument('--patch_size', '-p', default=7, type=int,
                     help='patch size')
args = parser.parse_args()

r   = 64

input   = args.input
param_h = args.smoothing_strength
search_size = args.search_window_size
patch_size  = args.patch_size
print   'param_h', param_h, 'search', search_size, 'patch', patch_size

img = cv2.imread(input)
h, w    = img.shape[:2]
img = cv2.resize(img, (w / 4, h / 4))
h, w    = img.shape[:2]
cv2.imwrite('result/resize.png', img)

print   'NLM...'
dst = cv2.fastNlMeansDenoisingColored(img, param_h, param_h, patch_size, search_size)
cv2.imwrite('result/result.png', dst)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

plt.figure()
plt.suptitle('Example of non-local means')

plt.subplot(221)
plt.imshow(img)
plt.axis('off')
plt.title('Noisy input')

plt.subplot(222)
plt.imshow(dst)
plt.axis('off')
plt.title('NL-means, h = {}'.format(param_h))

plt.subplot(223)
plt.imshow(img[h/2 - r:h/2 + r, w/2 - r: w/2 + r])
plt.axis('off')

plt.subplot(224)
plt.imshow(dst[h/2 - r:h/2 + r, w/2 - r: w/2 + r])
plt.axis('off')

plt.subplots_adjust(0, 0, 1, 0.9, 0, 0.1)
plt.savefig('result/fig.png')

plt.show()
