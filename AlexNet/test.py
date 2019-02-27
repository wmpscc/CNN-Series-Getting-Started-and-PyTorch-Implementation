# encoding=utf-8
import threading
import numpy as np
import tensorflow as tf


# 创建一个函数实现多线程，参数为Coordinater和线程号
def func(coord, t_id):
    count = 0


    for i in range(500):
        count += 1
        print('thread ID:', t_id, 'count =', count)
        if (count == 500):  # 计到5时请求终止
            coord.request_stop()


coord = tf.train.Coordinator()
threads = [threading.Thread(target=func, args=(coord, i)) for i in range(4)]
# 开始所有线程
for t in threads:
    t.start()
coord.join(threads)  # 等待所有线程结束
