#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 2017/12/29 14:05
#You can access a specific env via url:
#  http://localhost.com:8097/env/main
import visdom
import  numpy as np
vis=visdom.Visdom()
vis.text("hello world")
vis.image(np.ones((3,20,10)))


if __name__ == '__main__':
    pass