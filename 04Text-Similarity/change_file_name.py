#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os


def change_file_name():
    '''
    输入待处理文件夹路径
    将该文件夹下文档统一修改name格式如：282 育人案例那时花开.txt，序号自动递增
    :return:
    '''
    documentsPath = os.path.abspath('../dataset/DocumentsForChangeName/')
    print('待修改文件名的文档路径：', documentsPath)
    startIndex = 552  # 当前路径下索引号开始位置
    filename_list = os.listdir(documentsPath)
    print(filename_list)
    # filename_list.sort(key=lambda x:(x[:-4]))
    # print(filename_list.sort())
    n = 0
    for fname in filename_list:
        if fname[-4:] == '.txt':
            print(fname)
            # if fname[:3].isdigit():
            #     print(fname)
            oldname = documentsPath + filename_list[n]
            # print(filename_list[n][5:])
            indexstr = startIndex + n
            if indexstr < 10:
                indexstr = '000' + str(indexstr)
            elif indexstr >= 10 and indexstr < 100:
                indexstr = '00' + str(indexstr)
            elif indexstr >= 100 and indexstr < 1000:
                indexstr = '0' + str(indexstr)
            else:
                indexstr = str(indexstr)
            newname = documentsPath + indexstr + ' ' + filename_list[n][10:]

            os.rename(oldname, newname)
            print(oldname, '======>', newname)
            n += 1


if __name__ == '__main__':
    change_file_name()
