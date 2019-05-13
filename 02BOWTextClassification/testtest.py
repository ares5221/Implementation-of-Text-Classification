#!/usr/bin/env python
# _*_ coding:utf-8 _*_
# filename = 'stopword.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径
# result = []
# with open(filename, 'r',encoding='utf-8') as f:
#     for line in f:
#         print(line)
#         # result.append(list(line.strip('\n').split(',')))
f = open("word_dic.txt", "r", encoding='utf-8')  # 设置文件对象
str = f.read()  # 将txt文件的所有内容读入到字符串str中
print(str)
print(str[1:-1])
str = str[1:-1]
# print(str.split(','))
for i in str.split(','):
    # print(i)
    i= i.lstrip().lstrip('\'').rstrip('\'')
    print(i)
f.close()  # 将文件关闭
