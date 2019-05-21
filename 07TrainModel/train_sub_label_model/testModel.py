#!/usr/bin/env python
# _*_ coding:utf-8 _*_







def execute():
    sourcepath = os.path.abspath('../06ReadLabelSaveTxt/')
    real_file = 'label_text.csv'
    dataPath = os.path.join(sourcepath, real_file)
    print('ss', dataPath)
    content, train_label = read_data(dataPath)
    print(train_label)
    clean_label(train_label)
    print('ssss', train_label)
    train_data = peredata(content)
    build_model(train_data, train_label)


# START
if __name__ == '__main__':
    execute()