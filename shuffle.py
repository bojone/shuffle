#! -*- coding: utf-8 -*-
# 有限内存下，全局打乱大文件
# 多个文件合并打乱，然后在分割为多个文件
# 简介：https://kexue.fm/archives/8662

import glob, time
import numpy as np
from bert4keras.snippets import parallel_apply
from tqdm import tqdm

start_time = time.time()
batch_size = 100000

#
# =========== 局部打乱 ===========
#

jsons = glob.glob('corpus/*.json')


def generator():
    batch, k = [], 0
    for j in tqdm(jsons, ncols=0, desc='Local Shuffling'):
        with open(j) as f:
            for l in f:
                batch.append(l)
                if len(batch) == batch_size:
                    yield batch, k
                    batch = []
                    k += 1
    if batch:
        yield batch, k


def local_shuf(batch_k):
    batch, k = batch_k
    np.random.shuffle(batch)
    with open('corpus_local_shuf/%05d.json' % k, 'w') as f:
        for text in batch:
            f.write(text)


parallel_apply(
    func=local_shuf, iterable=generator(), workers=5, max_queue_size=10
)

#
# =========== 全局打乱 ===========
#

jsons = glob.glob('corpus_local_shuf/*.json')
opens = [open(j) for j in jsons]

n, k = 0, 0
F = open('corpus_shuf/%05d.json' % k, 'w')
for i in tqdm(range(batch_size), ncols=0, desc='Global Shuffling'):
    orders = np.random.permutation(len(opens))
    for i in orders:
        text = opens[i].readline()
        if text:
            n += 1
            F.write(text)
            if n == batch_size:
                n = 0
                k += 1
                F = open('corpus_shuf/%05d.json' % k, 'w')

if n == 0:
    os.remove('corpus_shuf/%05d.json' % k)

#
# =========== 计算时间 ===========
#

end_time = time.time()
print(u'总耗时：%s秒' % (end_time - start_time))
