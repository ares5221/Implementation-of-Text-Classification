#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import requests

def download_file(url):
    local_filename = url.split('/')[-1]
    print(local_filename)
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                print(chunk)
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename


if __name__ == '__mmain__':
    url = 'http://rasinsrv07.cstcis.cti.depaul.edu/CSC455/Assignment5.txt'
    download_file(url)
    r = requests.get(url)
    print(r.text)