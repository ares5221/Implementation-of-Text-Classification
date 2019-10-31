#!/usr/bin/env python
# _*_ coding:utf-8 _*_
#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from docx import Document
import re

'''解析pdf转换后的docx文件，将其中每个案例存放在单独的txt文件中'''
file_path = os.path.abspath('./')


def Translate(path):
    txt = [[] for i in range(80)]
    for f in os.listdir(path):
        if (f[0] == '~' or f[0] == '.'):
            continue
        filepath = os.path.join(path,f)#path + '/' + f
        if filepath[-5:] == '.docx':
            pattern_rule = re.compile("案例[\d]+")
            document = Document(filepath)  # 打开docx文件
            title_index = 0
            buttle = False
            for paragraph in document.paragraphs:
                # print(paragraph.text)     # 打印各段落内容文本
                linetxt = paragraph.text
                linetxt= linetxt.replace(' ','')

                startFlag = pattern_rule.findall(linetxt)
                if startFlag:
                    title_index +=1
                    buttle = True
                if buttle:
                    txt[title_index].append(linetxt)

    for cc in range(len(txt)):
        # print(cc, txt[cc])
        if txt[cc]:
            ss = txt[cc][0].replace('\t','')
            savename = ss+'.txt'
            for cont in txt[cc]:
                with open(savename, 'a', encoding='utf-8') as ff:
                    ff.write(cont)
                    ff.write('\n')



if __name__ == '__main__':
    Translate(file_path)
    print('文件夹中文件转换ok')
