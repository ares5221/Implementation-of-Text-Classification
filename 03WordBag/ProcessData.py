#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os


def process_data(filename):
    '''
    处理dataset数据,将数据按标签分为pos，neg
    filename:train, test表示处理的是dataset中两个文件夹下的数据
    :return: labels,texts
    '''
    db_dir = r'G:\tf-start\Implementation-of-Text-Classification\dataset'
    train_dir = os.path.join(db_dir, filename)

    labels = []
    texts = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname), 'r', encoding='UTF-8')
                tmp = f.read()
                # print(len(tmp), type(tmp))
                # if len(tmp) > 200:
                #     print(fname)
                #     tmp = tmp[:200]
                # print('#############',tmp)
                texts.append(tmp)
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    print(len(labels), len(texts))
    # for tt in texts:
    #     print('@@@@@@@@@@@@@@@', tt)
    return labels, texts


if __name__ == '__main__':
    filename = 'train'
    process_data(filename)