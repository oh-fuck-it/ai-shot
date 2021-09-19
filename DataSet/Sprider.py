from typing import List

import requests
from bs4 import BeautifulSoup

url = 'https://www.dpchallenge.com/image.php?IMAGE_ID='
path = './photonet_dataset.txt'


def load_data() -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
        ids = [line.split(" ")[1] for line in lines]
        f.close()
    return ids


def spider_data(ids: List[str]):
    headers = {
        'Host': 'www.dpchallenge.com',
        'sec-ch-ua': '"Google Chrome";v="93", " Not;A Brand";v="99", "Chromium";v="93"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36',
    }
    for id in ids:
        data = requests.get(url + id,headers=headers).content
        soup = BeautifulSoup(data, 'html.parser')
        img = soup.find_all('img', class_='page-image')
        print(img)


if __name__ == '__main__':
    spider_data(load_data())
