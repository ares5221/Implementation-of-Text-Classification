#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import urllib.request
import urllib
import requests
import json

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def get_response(msg):
    api = 'http://openapi.tuling123.com/openapi/api/v2'
    dat = {
        "perception": {
            "inputText": {
                "text": msg
            },
            "inputImage": {
                "url": "imageUrl"
            },
            "selfInfo": {
                "location": {
                    "city": "北京",
                    "province": "北京",
                    "street": "信息路"
                }
            }
        },
        "userInfo": {
            "apiKey": '676240625655401396553c7b1f77e799',
            "userId": '136772'
        }
    }
    dat = json.dumps(dat)
    r = requests.post(api, data=dat).json()
    print(r)
    mesage = r['results'][0]['values']['text']
    print(r['results'][0]['values']['text'])
    return mesage
                  
if __name__=='__main__':

    q = 'nn'
    get_response(q)


    if not is_Chinese(q):
        print(q, is_Chinese(q))
        url = 'http://www.tuling123.com/openapi/api'
        apikey = '676240625655401396553c7b1f77e799'
        test_data = {'key': apikey, 'info': "你好", 'userid': '136772'}
        # test_data_urlencode = urllib.urlencode(test_data)
        data = bytes(urllib.parse.urlencode(test_data), encoding='utf-8')

        res_data = urllib.request.urlopen(url=url, data=data)
        res = res_data.read().decode('utf8')
        print(json.loads(res)['text'])

    
    
