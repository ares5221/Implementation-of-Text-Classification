#!/usr/bin/env python
# _*_ coding:utf-8 _*_

def foo():
    with open('test.txt', 'a', encoding='utf-8') as f:
        f.write('this is Outer text ' + '\n')
        # f.close()
        for i in range(2):
            sample()


def sample():
    with open('test.txt', 'a', encoding='utf-8') as f:
        f.write('this is Inner text ' + '\n')


if __name__ == "__main__":
    foo()
