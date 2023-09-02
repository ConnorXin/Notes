# 一个爬虫小程序

```python
from urllib.request import urlopen

url = 'http://www.baidu.com'
resp = urlopen(url)

resp.read()  # 从网址读取内容
'''
但是 resp.read() 中的内容都是字节 需要进行转换
'''

resp.read().decode('utf-8')  # 内容可以出现中文  但是一堆字符串依旧看不懂

with open('mybaidu.html', mode = 'w') as f:
    f.write(resp.read().decode('utf-8'))  # 读取到网页源代码
    
print('over')
```

# Web 请求全过程

1. 服务器渲染
   在服务器那边直接把数据和 html 整合在一起，统一返回给浏览器
   在网页源代码是能够找得到网页内容
2. 客户端渲染
   第一次请求只要一个 html 骨架，第二次请求拿到数据，进行数据展示
   在网页源代码中看不到数据，即看不到网页内容
3. 熟练使用浏览器抓包工具
   检查 (F12)

# HTTP 协议

协议
就是两个计算机之间为了能够流畅的进行沟通而设置的一个君子协定，常见的协议有 TCP/IP.SOAP 协议，HTTP 协议，SMTP 协议等等

HTTP 协议
Hyper Text Transfer Protocol (超文本传输协议) 的缩写，用于从万维网 (WWW: World Wide Web) 服务器传输文本到本地浏览器的传送协议，直白点就是浏览器和服务器之间的数据交互遵守的协议

## 请求方式

**GET**

显示请求；一般是查询某些东西

**POST**

隐示提交；对服务器的数据进行一点修改

# Requests 模块

```python
pip install requests
```

```python
# -*- coding: utf-8 -*-
'''
example 1
'''
import requests


if __name__ == '__main__':
    
    url = 'https://www.baidu.com/s?wd=%E5%88%98%E9%9B%A8%E6%98%95&rsv_spt=1&rsv_iqid=0x82794a2b0003dc97&issp=1&f=8&rsv_bp=1&rsv_idx=2&ie=utf-8&tn=baiduhome_pg&rsv_enter=1&rsv_dl=ib&rsv_sug3=5&rsv_sug1=5&rsv_sug7=101&rsv_sug2=0&rsv_btype=i&inputT=761&rsv_sug4=2137'

    # 模拟正常浏览器对网址进行访问
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
    }
    
    resp = requests.get(url, headers = headers)  # 在地址栏输入的网址链接一定是 get 方式提交 因此使用 get 请求
    print(resp)  # <Response [200]>
    print(resp.text)  # 拿到页面源代码
    
    resp.close()  # 最后一定要关掉 resp
```

```python
# -*- coding: utf-8 -*-
'''
example 2
'''

import requests


if __name__ == '__main__':
    '''
    百度翻译 网站
    '''
    url = 'https://fanyi.baidu.com/sug'

    # 模拟正常浏览器对网址进行访问
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
    # }
    word = input('please input words you want to translate\n')
    # 我们要发送的数据
    dict_data = {
        'kw': word
    }

    # 发送 post 请求 发送的数据必须放在字典中 通过 data 参数进行传递
    resp = requests.post(url, data = dict_data)

    print(resp.json())  # 将服务器返回的内容直接处理成 json()  => dict
    
    resp.close()  # 最后一定要关掉 resp
    

'''
result:
please input words you want to translate
dog
{'errno': 0, 'data': [{'k': 'dog', 'v': 'n. 狗; 蹩脚货; 丑女人; 卑鄙小人 v. 困扰; 跟踪'}, {'k': 'DOG', 'v': 'abbr. Data Output Gate 数据输出门'}, {'k': 'doge', 'v': 'n. 共和国总督'}, {'k': 'dogm', 'v': 'abbr. dogmatic 教条的; 独断的; dogmatism 教条主义; dogmatist'}, {'k': 'Dogo', 'v': '[地名] [马里、尼日尔、乍得] 多戈; [地名] [韩国] 道高'}]}

'''
```

```python
# -*- coding: utf-8 -*-
'''
example 3
'''

import requests


if __name__ == '__main__':
    '''
    https://movie.douban.com/typerank?type_name=%E5%8A%A8%E4%BD%9C&type=5&interval_id=100:90&action=
    该网址是属于第二次请求服务器才有数据
    因此在网页源代码中并没有数据
    抓包工具中有 XHR
    第二次请求一般是 XHR
    
    从 XHR 中找到想要的数据的 url 即网页中的影片信息
    
    url: https://movie.douban.com/j/chart/top_list?type=5&interval_id=100%3A90&action=&start=0&limit=20
    # 以上 url 中 ? 前面的是 url; ? 后面的是参数
    在抓包工具中的 Payload 中有参数
    '''

    url = 'https://movie.douban.com/j/chart/top_list'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
    }

    # 重新封装参数
    params = {
        'type': '5',
        'interval_id': '100:90',
        'action': '',
        'start': 0,
        'limit': 20
    }

    # 发送 get 请求
    resp = requests.get(url = url, params = params, headers = headers)

    print(resp.request.url)  # 查看重新封装后的 url 结果
    print(resp.text)  # 发现什么都没输出 什么东西都没有; 加入 User_Agent 之后能够成功拿到数据
    print(resp.json())  # 转换成 json 可读性较强
    
    resp.close()  # 最后一定要关掉 resp
```

# 数据解析

目前掌握了抓取整个网页的基本技能，但是大多数情况不需要整个网页内容，只需要一小部分

爬虫中，提供三种解析方式

1. `re` 解析
2. `bs4` 解析
3. `xpath` 解析

# Re 解析

## 正则表达式

正则表达式  Regular Expression
一种使用表达式的方式对字符串进行匹配的语法规则

我们抓取到的网页源代码本质上就是一个超长字符串，想从里面提取内容，用正则表达式再合适不过了

正则的语法
使用元字符进行排列组合用来匹配字符串，在线测试正则表达式

[oschina]: https://tool.oschina.net/regex/	"oschina"

**元字符**
具有固定含义的特殊符号

常用元字符

```python
.		匹配除换行符以外的任意字符
\w		匹配字母或数字或下划线
\s		匹配任意的空白符
\d		匹配数字
\n		匹配一个换行符
\t		匹配一个制表符

^		匹配字符串的开始
$		匹配字符串的结尾

\W		匹配非字母或数字或下划线
\D		匹配非数字
\S		匹配非空白符
a|b		匹配字符 a 或字符 b
()		匹配括号内的表达式 也表示一个组
[...]	匹配字符组中的字符
[^...]	匹配除了字符组中字符的所有字符
```

**量词**
控制前面的元字符出现的次数

```python
*		重复零次或更多次
+		重复一次或更多次
?		重复零次或一次
{n}		重复 n 次
{n,}	重复 n 次或更多次
{n,m}	重复 n 到 m 次
```

**贪婪匹配 / 惰性匹配**

```python
.*		贪婪匹配
.*?		惰性匹配
```

## python 正则

```python
# -*- coding: utf-8 -*-
import re

if __name__ == '__main__':

    '''
    findall: 匹配字符串中所有符合正则的内容
    但是使用不多 效率并不高
    '''
    li = re.findall(r'\d+', '我的电话是10086 / 10000')  # 前面加 r 之后  '\d+' 下面就没有波浪线了
    print(li)

    
    print('-' * 100)
    '''
    finditer: 匹配字符串中所有的内容 返回的是迭代器
    '''
    it = re.finditer(r'\d+', '我的电话是10086 / 10000')
    print(it)  # <callable_iterator object at 0x000001634023D910>
    # 从迭代器中拿东西
    for i in it:
        print(i)  # <re.Match object; span=(5, 10), match='10086'>
        print(i.group())  # 通过 group() 拿到我们想要的内容

        
    print('-' * 100)
    '''
    seach 返回的结果是 match 对象  需要使用 group()
    但是只能拿到第一个匹配的数据
    找到一个结果就返回
    '''
    s = re.search(r'\d+', '我的电话是10086 / 10000')
    print(s)  # <re.Match object; span=(5, 10), match='10086'>
    print(s.group())  # 10086

    
    print('-' * 100)
    '''
    match 从头开始匹配 开头不匹配就报错
    '''
    # s = re.match(r'\d+', '我的电话是10086 / 10000')
    # print(s.group())  # AttributeError: 'NoneType' object has no attribute 'group'; 出现此错误说明 .group 前面是空
    s = re.match(r'\d+', '10086 / 10000')
    print(s.group())  # 10086
    # s = re.match(r'\d+', 'co10086 / 10000')
    # print(s.group())  # 同样出现上面错误

    
    print('-' * 100)
    '''
    预加载正则表达式
    '''
    obj = re.compile(r'\d+')
    ret = obj.finditer('我的电话是10086 / 10000')
    print(ret)  # <callable_iterator object at 0x000001634023D6A0>

    
    print('-' * 100)
    '''
    一段 html
    <div class='xin'><span id='1'>刘雨昕</span></div>
    <div class='jay'><span id='2'>周杰伦</span></div>
    <div class='jolin'><span id='3'>郭麒麟</span></div>
    <div class='sylar'><span id='4'>范思哲</span></div>
    <div class='tory'><span id='5'>胡歌</span></div>
    
    (?P<group name>.*?): 在 .*? 的匹配内容前面加上 ?P<nama> 同时外面加上小括号加组；group name 表示给组起一个名字
    其他我们想要的数据同理
    '''
    htmlC = '''
    <div class='xin'><span id='1'>刘雨昕</span></div>
    <div class='jay'><span id='2'>周杰伦</span></div>
    <div class='jolin'><span id='3'>郭麒麟</span></div>
    <div class='sylar'><span id='4'>范思哲</span></div>
    <div class='tory'><span id='5'>胡歌</span></div>
    '''
    obj = re.compile(r'div class=\'.*?\'><span id=\'.*?\'>(?P<name>.*?)</span></div>', re.S)  # re.S: 让.能够匹配换行符
    result = obj.finditer(htmlC)
    for it in result:
        # print(it.group())  # 不符合我们想要的数据
        print(it.group('name'))  # 我们想要的数据
```

# 案例: 豆瓣电影 TOP250 排行

```python
# -*- coding: utf-8 -*-
'''
movies_doubanTOP250.py
豆瓣电影 TOP250  https://movie.douban.com/top250
网页源代码能够找到电影相关信息 说明是服务器渲染的数据

思路
1 拿到网页源代码  通过 requests 模块
2 提取有效信息  通过 re 模块
'''

import numpy as np
import pandas as pd
import requests
import re


if __name__ == '__main__':

    url = 'https://movie.douban.com/top250'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
    }
    respond = requests.get(url, headers = headers)
    page_content = respond.text  # 发现为空 必须要有 User-Agent


    '''
    想要排行中的 电影名字 年份 评分 评价数
    '''
    movies = pd.DataFrame(columns=['name', 'year', 'rating', 'ratingNum'])
    # 解析数据
    obj = re.compile(r'<li>.*?<div class="item">.*?<span class="title">(?P<movie_name>.*?)</span>'
                     r'.*?<p class="">.*?<br>(?P<movie_year>.*?)&nbsp;/&nbsp'
                     r'.*?<span class="rating_num" property="v:average">(?P<movie_rate>.*?)</span>'
                     r'.*?<span>(?P<movie_rateNum>.*?)人评价</span>', re.S)
    movie_info = obj.finditer(page_content)
    for it in movie_info:

        name = np.array(it.group('movie_name'))  # 成功拿到数据
        year = np.array(it.group('movie_year').strip())  # 年份前面有好多空白 使用 strip 处理
        rating = np.array(it.group('movie_rate'))
        ratingNum = np.array(it.group('movie_rateNum'))

        movies = movies.append({
            'name': name, 'year': year,
            'rating': rating, 'ratingNum': ratingNum
        }, ignore_index=True)



    respond.close()


    movies.to_csv('movies_doubanTOP250.csv', encoding = 'utf-8')

    print(movies)
```

# 案例: 电影天堂

```python
# -*- coding: utf-8 -*-
'''
movies_dytt.py
电影天堂  https://www.dytt89.com/
获取2023年必看热片片名与下载地址

思路
1 定位到2023必看热片
2 从2023必看热片中提取到子页面的链接地址
3 请求子页面的链接地址 拿到下载地址
'''

import numpy as np
import pandas as pd
import requests
import re


if __name__ == '__main__':

    url = 'https://www.dytt89.com/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
    }
    respond = requests.get(url = url, headers = headers)  # verify = False  意思是去掉安全验证
    respond.encoding = 'gb2312'  # 指定字符集
    page_content = respond.text  # 指定字符集之后正常显示中文

    # print(page_content)  # 乱码，requests 里面默认解码是 utf-8；网站编码若不是 utf-8，则出错


    # 使用 "2023必看热片" 进行定位
    # 解析数据
    obj1 = re.compile(r'2023必看热片.*?<ul>(?P<moive_li>.*?)</ul>', re.S)
    obj2 = re.compile(r'a href=\'(?P<moive_url>.*?)\' title=', re.S)
    obj3 = re.compile(r'◎译　　名(?P<translate_name>.*?)<br />◎片　　名(?P<name>.*?)<br />.*?'
                      r'<td style="WORD-WRAP: break-word" bgcolor="#fdfddf"><a href="(?P<download>.*?)">magnet:?', re.S)
    movieLi = obj1.finditer(page_content)
    for it in movieLi:
        movie_2023 = it.group('moive_li')

        # 提取 movie_2023 子页面中的 a 标签里面的 herf 超链接  即子页面的链接
        childUrl = obj2.finditer(movie_2023)
        child_urlList = np.array([])
        for itt in childUrl:
            # movie_url =
            # 拼接子页面的 url
            child_urlList = np.append(child_urlList, url + itt.group('moive_url').strip('/'))

            # print(movie_2023)
        # print(child_urlList)

    # 从子页面解析数据
    translate_name = np.array([])
    movie_name = np.array([])
    movie_download = np.array([])
    movies_download = pd.DataFrame(columns = ['Translate Name', 'Movie Name', 'Download'])
    for ur in child_urlList:
        child_respond =requests.get(url = ur, headers = headers)
        child_respond.encoding = 'gb2312'
        child_content = child_respond.text
        child_contents = obj3.search(child_content)
        translate_name = child_contents.group('translate_name').strip()
        movie_name = child_contents.group('name').strip()
        movie_download = child_contents.group('download').strip()

        movies_download = movies_download.append({
            'Translate Name': translate_name,
            'Movie Name': movie_name,
            'Download': movie_download
        }, ignore_index=True)

    # print(translate_name)
    # print(movie_name)
    print(movies_download)

    respond.close()
    child_respond.close()


    movies_download.to_csv('dyttMovie_2023.csv', encoding = 'utf-8')
```

# bs4 解析

使用 bs4 中的 BeautifulSoup 通过 html 标签对数据的抓取

## 案例: 爬取农贸产品的价格

```python
# -*- coding: utf-8 -*-
# @time    : 2023/3/27 20:35
# @author  : w-xin
# @file    : bs4_base.py
# @software: PyCharm

"""
北京新发地 爬取农贸产品的价格
但是现在的北京新发地网址已经不是 GET 请求
如下代码仅供 bs4.BeautifulSoup 学习
"""

import time
import requests
from bs4 import BeautifulSoup


if __name__ == '__main__':

    url = 'http://www.xinfadi.com.cn/index.html'
    respond = requests.get(url)

    # 解析数据
    # 把页面源代码交给 BeautifulSoup 进行处理，生成 bs 对象
    page = BeautifulSoup(respond.text, 'html.parser')  # html.parser 指定 html 解析器

    # 从 bs 对象中查找数据
    '''
    find(标签, 属性=值)
    find_all(标签, 属性=值)
    '''
    table = page.find('table', class_ = 'hq_table')  # class 是关键字 因此在后面添加下划线
    table = page.find('table', attrs = {'class': 'hq_table'})  # 与上一行一个意思不同写法 可以避免 class

    # 拿到所有数据行
    trs = table.find_all('td')[1: ]
    for tr in trs:
        tds = tr.find_all('td')
        name = tds[0].text
        low = tds[1].text
        avg = tds[2].text
        high = tds[3].text
        scale = tds[4].text
        kind = tds[5].text
        date = tds[6].text

    time.sleep(1)
```

## 案例: 安居客房源数据

```python
# -*- coding: utf-8 -*-
# @time    : 2023/3/28 8:32
# @author  : w-xin
# @file    : bs4_select.py
# @software: PyCharm

# 全局取消证书验证
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import requests
import bs4
import pandas as pd
import time


if __name__ == '__main__':


    # 模拟浏览器头部
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'
    }


    df = pd.DataFrame(columns=['小区名称', '户型', '详细地址', '面积', '价格', '楼层', '朝向', '租用类型'])

    page = 1
    item_num, max_num = 1, 3000  # 爬取3000条数据
    while True:
        # 获取其中一页的源代码
        url = 'https://sz.zu.anjuke.com/fangyuan/p{}/'.format(page)
        response = requests.get(url, headers=header)
        if response.status_code != 200:
            print('终止页为', page)
            break
        response.encoding = 'utf-8'
        html = response.text
        soup = bs4.BeautifulSoup(html, 'lxml')
        # div_li_1 = soup.select('div[class="zu-info"]')
        # div_li_2 = soup.select('div[class="zu-side"]')
        div_li = soup.select('div[class="zu-itemmod"]')
        print(div_li)

        for div in div_li:
            # 爬取楼层
            flo = div.select('div[class="zu-info"] > p[class="details-item tag"]')[0].text
            flo = flo.split('|')[2]
            flo_ind = flo.find(')')
            flo = flo[: flo_ind + 1]
            # 爬取小区名称
            name = div.select('div[class="zu-info"] > address[class="details-item"] > a[target="_blank"]')[0].text
            # 爬取户型
            sha_1 = div.select('div[class="zu-info"] > p[class="details-item tag"] > b[class="strongbox"]')[0].text
            sha_2 = div.select('div[class="zu-info"] > p[class="details-item tag"] > b[class="strongbox"]')[1].text
            hou_shape = sha_1 + '室' + sha_2 + '厅'
            # 爬取地址
            address_ = div.select('div[class="zu-info"] > address[class="details-item"]')[0].text
            address_ = address_.replace(' ', '')
            address_ = address_.strip()
            for i in range(1, 50):
                if len(address_[0: i]) == len(name):
                    address_ = address_[i:]
                    address_ = address_.strip()
                else:
                    continue
            # 抓取面积
            area = div.select('div[class="zu-info"] > p[class="details-item tag"] > b[class="strongbox"]')[
                       2].text + '平米'
            # 爬取朝向
            direct = div.select('div[class="zu-info"] > p[class="details-item bot-tag"] > span[class="cls-2"]')[0].text
            # 爬取租用类型
            hire_type = div.select('div[class="zu-info"] > p[class="details-item bot-tag"] > span[class="cls-1"]')[
                0].text
            # 抓取价格
            # print(div.select('p > strong > b[class="strongbox"]')[0].text)
            price = div.select('div[class="zu-side"] > p')[0].text

            df = df.append({
                '小区名称': name, '户型': hou_shape,
                '详细地址': address_, '面积': area,
                '价格': price, '朝向': direct,
                '租用类型': hire_type, '楼层': flo
            }, ignore_index=True)
            item_num += 1
            if item_num > max_num:
                break

        if item_num > max_num:
            print('爬取完毕')
            break
        print('当前爬取页为：', page)
        page = page + 1
        time.sleep(1)
```

## 案例: 优美图库下载图片

```python
# -*- coding: utf-8 -*-
# @time    : 2023/3/27 21:28
# @author  : w-xin
# @file    : img_reptile.py
# @software: PyCharm

"""
优美图库爬取图片
https://www.umei.cc/weimeitupian/yijingtupian/

思路
页面源代码中提取子页面链接
即需要找到源代码中的 href 标签
"""
import time

import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':

    url = 'https://www.umei.cc/weimeitupian/yijingtupian/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
    }

    respnd = requests.get(url = url, headers = headers)
    respnd.encoding = 'utf-8'
    html_content = respnd.text

    soup = BeautifulSoup(html_content, 'lxml')
    img_aLi = soup.select('div[class="item_list infinite_scroll"] > div[class="item masonry_brick"] > '
                          'div[class="item_b clearfix"] > div[class="title"] > span > a')

    # 从标签内获取属性的值
    fa_url = 'https://www.umei.cc/'
    for li in img_aLi:
        href = li.get('href')  # 拿到子页面链接
        child_url = fa_url + href.strip('/')
        # print(child_url)

        child_respond = requests.get(url = child_url, headers = headers)
        child_respond.encoding = 'utf-8'
        child_content = child_respond.text

        # 从子页面拿到图片下载路径
        # 通过 BeautifulSoup 拿到图片下载地址
        child_soup = BeautifulSoup(child_content, 'lxml')
        child_url = child_soup.find('div', class_ = "big-pic").find('a').find('img').get('src')
        # print(child_url)

        # 下载图片
        img_respond = requests.get(child_url)
        # 从响应里面拿到图片
        img_respond.content  # 这里拿到的是字节  要把字节写到文件里面去
        # 文件命名  建立文件夹存储
        img_name = child_url.split('/')[-1]
        with open('./img/' + img_name, mode = 'wb') as f:
            f.write(img_respond.content)  # 图片内容写入文件

        print(f'{img_name} over')
        time.sleep(1)

    print('all over!')

    respnd.close()
    child_respond.close()

    # print(href)
```

# xpath 解析

xpath 是在 XML 文档中搜索内容的一门语言；html 是 xml 的一个子集

在 Python 中想用 xpath 需要安装 lxml 模块

## 简单 xpath

```python
# -*- coding: utf-8 -*-
# @time    : 2023/3/29 23:33
# @author  : w-xin
# @file    : xpath_basic.py
# @software: PyCharm
from lxml import etree

if __name__ == '__main__':

    xml = '''
    <book>
    <id>l</id>
    <name>野花遍地香</name>
    <price>1.23</price>
    <nick>土豆洋芋</nick>
    <author>
        <nick id="10086">刘雨昕</nick>
        <nick id="10000">胡歌</nick>
        <nick class="xin">XINLIU</nick>
        <nick class="wangyang">王阳</nick>
        <div>
            <nick>刘雨昕！1</nick>
        </div>
        <span>
            <nick>刘雨昕！2</nick>
        </span>
        
    </author>

    <partner>
        <nick id="ppc">胖胖陈</nick>
        <nick id="ppbc">胖胖不陈</nick>
    </partner>
    </book>'''


    tree = etree.XML(xml)
    result = tree.xpath('/book')  # / 表示层级关系  第一个 / 是根节点
    # print(result)  # [<Element book at 0x26e04d5ebc0>]

    result = tree.xpath('/book/name')
    # print(result)  # [<Element name at 0x255eb6a8080>]
    # 想看标签内容
    result = tree.xpath('/book/name/text()')  # text() 表示拿到标签里面的文本
    # print(result)  # ['野花遍地香']


    # 拿取 nick 标签内容
    result = tree.xpath('/book/author/nick/text()')
    # print(result)  # ['刘雨昕', '胡歌', 'XINLIU', '王阳']

    result = tree.xpath('/book/author/div/nick/text()')
    # print(result)  # ['刘雨昕！']

    # 把 author 里面的内容一次性输出
    result = tree.xpath('/book/author//nick/text()')  # // 表示把 author 这个父节点里面的所有子节点内容都拿到
    # print(result)  # ['刘雨昕', '胡歌', 'XINLIU', '王阳', '刘雨昕！']  ['刘雨昕', '胡歌', 'XINLIU', '王阳', '刘雨昕！1', '刘雨昕！2', '刘雨昕！3']

    # 修改 div 为 span
    # 需要把刘雨昕！1  刘雨昕！2 一次拿到
    result = tree.xpath('/book/author/*/nick/text()')  # * 表示任意节点
    print(result)  # ['刘雨昕！1', '刘雨昕！2']
```

## 案例: 猪八戒网站

```python
# -*- coding: utf-8 -*-
# @time    : 2023/3/30 14:21
# @author  : w-xin
# @file    : zbjWebsite.py
# @software: PyCharm

"""
猪八戒网站
https://beijing.zbj.com/
"""


import time
import requests
from lxml import etree

if __name__ == '__main__':

    url = 'https://beijing.zbj.com/search/service/?kw=saas&r=2'
    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'
    }

    html_respond = requests.get(url = url, headers = header)
    html_content = html_respond.text
    # print(html_content)

    # xpath 解析
    tree = etree.HTML(html_content)
    bricks = tree.xpath('//*[@id="__layout"]/div/div[3]/div/div[4]/div/div[2]/div[1]/div')  # 全部服务商
    # print(brick)

    # 遍历每一个服务商
    for brick in bricks:
        server_name = brick.xpath('.//div/a/div[2]/div[1]/div/text()')
        server_price = brick.xpath('.//div/div[2]/div[1]/span/text()')
        server_intro = brick.xpath('.//div/div[2]/div[2]/a/text()')


        # print(server_name)
        print(server_price)
        # print(server_intro)
        time.sleep(1)
    html_respond.close()
```

# Requests 模块进阶

在之前的爬虫程序中已经使用过 headers 了，headers 为 HTTP 协议中的请求头，一般存放一些和请求内容无关的数据，有时也会存放一些安全验证信息，比如常见的 User-Agent, token, cookie 等

通过 requests 发送的请求，我们可以把请求头信息放在 headers 中，也可以单独进行存放，最终由 requests 自动帮我们拼接成完整的 http 请求头

## 处理 cookie

网页进行登录

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-04-03%20230551.png)

首先进行登录，得到 cookie
带着 cookie 去请求到书架 url
拿到书架上的内容

## 案例: 17k 小说网

```python
# -*- coding: utf-8 -*-
# @time    : 2023/4/3 23:10
# @author  : w-xin
# @file    : GetNovels.py
# @software: PyCharm

"""
登录 -> 得到 cookie
带着 cookie 去请求到书架 url -> 书架上的内容
必须把以上两个操作连起来

我们可以使用 session 进行请求  -> session 可以认为是一连串的请求；在这个过程中的 cookie 不会丢失
"""
import requests

if __name__ == '__main__':
    # 准备 session
    session = requests.session()

    # 登录
    url = 'https://passport.17k.com/ck/user/login'
    data = {
        'loginName': '13823273489',
        'password': '!@#$1234qwer'
    }
    respond = session.post(url = url, data = data)
    # print(respond.text)
    # print(respond.cookies)  # 查看 cookie

    # 抓取书架的数据
    # 找到隐藏起来的书架 url 数据地址
    book_url = 'https://user.17k.com/ck/author/shelf?page=1&appKey=2406394919'
    book_resp = session.get(url = book_url)
    print(book_resp.json())

    # 不使用 session 使用 requsets 也行 但是麻烦
    # book_resp = requests.get(url = book_url, headers = {
    #     'Cookie': 'GUID=038a8c04-dd9c-407a-a910-77af03873ef3; Hm_lvt_9793f42b498361373512340937deb2a0=1680535053,1680682714; c_channel=0; c_csc=0; __bid_n=187508248aa907f7624207; FPTOKEN=ymAwckJoUxYtm1nG+XpXyFKASi8YRVNjT3DoCVQxSQqXfbqfxxWxaGgC7h3WwoNwceUqpHZJqy3lFSafkMJZopJu+wQOTn9eFDOp29fuao2oQSXkhIQwkLkuPTzUnrcnCV5FDCxRlA7HKJEix+pmT0v0z0GpzTv0mBQiJNVSsjEk2vbl1Diuvaoi1wgmbe9dkAqOoInGIYVKeLwbG8Cyr3KO+pnbhYpMMzbBBaHqv7GDBHe0YnkDTdZFt+FV3pKp/Wi7tpCBC5ScXSXqBz4ar86ANsWkZ7DXFln6ivDQiOJUXfcd/UxiJvBQL5rJ8DZxlI4aU72Jilv0QpVryY0T5s7oQsrCOYWIsnV3vJ+JTE7HDnNc+T72KaBbeYMI2GWbEj4oEpvmdDppPKm2z5UvWw==|RonY/4VooJvfVgim2uRQ3tEY2w+r2ogRfBGZQbYX7TA=|10|b1131f64752b7cd681f340bf58390276; accessToken=avatarUrl%3Dhttps%253A%252F%252Fcdn.static.17k.com%252Fuser%252Favatar%252F08%252F28%252F54%252F100135428.jpg-88x88%253Fv%253D1680683866000%26id%3D100135428%26nickname%3D%25E4%25B9%25A6%25E5%258F%258B0S9f98xF6%26e%3D1696236097%26s%3D83050cf413460f93; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%22100135428%22%2C%22%24device_id%22%3A%2218747b1e20e669-01f03861f8b97d-26031851-921600-18747b1e20f107e%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E5%BC%95%E8%8D%90%E6%B5%81%E9%87%8F%22%2C%22%24latest_referrer%22%3A%22https%3A%2F%2Fgraph.qq.com%2F%22%2C%22%24latest_referrer_host%22%3A%22graph.qq.com%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC%22%7D%2C%22first_id%22%3A%22038a8c04-dd9c-407a-a910-77af03873ef3%22%7D; Hm_lpvt_9793f42b498361373512340937deb2a0=1680686033'
    # })
    # print(book_resp.text)


    session.close()
    respond.close()
    book_resp.close()
```

## 案例: 爬取梨视频

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-04-05%20195017.png)

srcurl 为视频相关的链接地址，但是实际上这个链接什么都没有，为404 Not Found

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-04-05%20195333.png)

对比两个网址

```python
# -*- coding: utf-8 -*-
# @time    : 2023/4/5 19:42
# @author  : w-xin
# @file    : Pearvideo.py
# @software: PyCharm

"""
梨视频视频爬取     https://pearvideo.com/

在开发者工具中能够找到 video 标签的视频链接
但是在网页源代码并没有 video 标签  说明此视频有可能是后期通过 js 二次加载进去的
开发者工具中的源代码与页面源代码是有偏差的
"""
import requests

if __name__ == '__main__':

    original_url = 'https://pearvideo.com/video_1693830'

    contId = original_url.split('_')[1]

    # 通过这个 url 拿到视频相关的 json
    # 把 json 里面的 srcurl 并且需要把里面的部分链接替换掉得到视频链接
    video_status = f'https://pearvideo.com/videoStatus.jsp?contId={contId}&mrd=0.7503395563056279'

    '''
    Referer: https://pearvideo.com/video_1693830
    Header 下面有个 Referer 为防盗链
    防盗链: 溯源，当前链接的上一级链接
    header 中加上防盗链就能拿到 srcurl 的 json
    '''

    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36',
        'Referer': 'https://pearvideo.com/video_1693830'
    }

    # 通过 requests 拿到 video_status 中的 json
    respond = requests.get(url = video_status, headers = header)
    srcurl = respond.json()['videoInfo']['videos']['srcUrl']
    systemTime = respond.json()['systemTime']
    # print(systemTime)

    # srcurl: https://video.pearvideo.com/mp4/adshort/20200826/1680695370568-15348904_adpkg-ad_hd.mp4
    # videourl: https://video.pearvideo.com/mp4/adshort/20200826/cont-1693830-15348904_adpkg-ad_hd.mp4
    # 对 srcurl 进行替换 拿到真实的视频链接

    video_utl = srcurl.replace(systemTime, 'cont-' + contId)
    # print(video_utl)


    # 下载视频
    with open('./pearvidel.mp4', mode = 'wb') as f:
        f.write(requests.get(video_utl).content)

    print('download over!')

    respond.close()
```

## 代理

通过第三方的一个机器去去发送请求

```python
# 免费代理 ip  zdaye.com/FreePlist.html
# 找到一个代理 ip
# 218.60.8.83:3129
proxies = {
    'https': 'https://218.60.8.83:3129'
}

respond = requests.get(url = 'https://www.baidu.com', proxies = proxies)

print(respond.text)
```

# 综合训练: 网易云音乐评论

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-04-05%20212242.png)

点击 Request call stack 最上面的那个

点进去之后的代码是经过了压缩 经过了加密的代码；在 Send 上打一个断点；重新刷新
看右边栏的 Local；在这个断点之前的代码的所有变量都在 Local 中
点开 request；
看 url 是否是自己需要的，点击蓝色键进行下一次拦截

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-04-05%20213147.png)

点击回到函数 v~ 中

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-04-05%20214058.png)

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-04-05%20220226.png)

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-04-05%20220554.png)

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-04-05%20220756.png)

# 爬虫效率

## 线程 / 进程

**进程**

是资源单位，每一个进程至少要有一个线程

**多线程**

线程是执行单位

启动每一个程序默认都会有一个主线程

```python
# -*- coding: utf-8 -*-
# @time    : 2023/4/6 23:48
# @author  : w-xin
# @file    : what_thread.py
# @software: PyCharm


"""
以下程序并不是多线程的程序
是单线程程序

func 函数执行完成之后执行 func 下面的循环
"""

def func():
    '''
    示例程序
    :return:
    '''
    for i in range(1000):
        print('func', i)


if __name__ == '__main__':

    func()
    for i in range(1000):
        print('main', i)
```

```python
# -*- coding: utf-8 -*-
# @time    : 2023/6/19 10:36
# @author  : w-xin
# @file    : multiple_thread.py
# @software: PyCharm


"""
python Thread
"""
from threading import Thread

def func():
    '''
    example code
    :return:
    '''

    for i in range(1000):
        print("func", i)


if __name__ == '__main__':

    t = Thread(target = func)
    t.start()  # 多线程状态为开始工作状态，具体的执行时间由 CPU 决定

    for i in range(1000):
        print('main', i)
```

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-06-19%20104423.png)

```python
"""
python Thread
"""
from threading import Thread

def func(name):
    '''
    example code
    :param name: 区别名称
    :return:
    '''

    for i in range(1000):
        print(name, i)


if __name__ == '__main__':

    # 创建多个多线程
    t1 = Thread(target = func, args = ('xin',))  # 传参进行区别 且必须是元组
    t1.start()  # 多线程状态为开始工作状态，具体的执行时间由 CPU 决定

    t2 = Thread(target = func, args = ('um',))
    t2.start()
```



```python
'''
第二种写法
'''
class MyThread(Thread):  # 继承 Thread

    def run(self):
        for i in range(1000):
            print('子线程', i)


if __name__ == '__main__':

    t = MyThread()
    t.start()

    for i in range(1000):
        print('主线程', i)
```

**多进程**

```python
# -*- coding: utf-8 -*-
# @time    : 2023/6/19 10:51
# @author  : w-xin
# @file    : multiple_process.py
# @software: PyCharm


"""
multiple process
"""
from multiprocessing import Process


# def function():
#     for i in range(1000):
#         print("子进程", i)
#
#
# if __name__ == '__main__':
#
#     p = Process(target = function)
#     p.start()
#
#     for i in range(1000):
#         print("主进程", i)



class MyProcess(Process):  # 继承 Process

    def run(self):
        for i in range(1000):
            print('子进程', i)


if __name__ == '__main__':

    p = MyProcess()
    p.start()

    for i in range(1000):
        print('主进程', i)
```

## 线程池 / 进程池

**线程池**

一次性开辟一些线程，用户直接给线程池提交任务，具体哪条线程执行不用管

```python
# -*- coding: utf-8 -*-
# @time    : 2023/6/19 13:04
# @author  : w-xin
# @file    : ThreadProcess_Pool.py
# @software: PyCharm

"""
Thread & Process Pool
"""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def fn(name):


    for i in range(1000):
        print(name, i)


if __name__ == '__main__':

    # 创建线程池
    with ThreadPoolExecutor(50) as t:  # 创建50个线程的线程池
        for i in range(100):  # 100个任务交给线程池
            t.submit(fn, name = f'线程{i}')  # 向线程池提交任务

    # 等待线程池中的任务全部执行完毕，才继续执行
    print('333')

    # 改成进程即把名称改了
```

**案例**

```python
# -*- coding: utf-8 -*-
# @time    : 2023/6/19 13:14
# @author  : w-xin
# @file    : example_withThreadProcessPool.py
# @software: PyCharm

"""
北京新发地 网站

1 如何提取单个页面数据
2 线程池 多个页面同时抓取
"""
import requests
import csv
from lxml import etree
from concurrent.futures import ThreadPoolExecutor


def download_one_page(url):
    '''
    提取一个页面的数据
    :param url: 页面网址
    :return:
    '''
    resp = requests.get(url)
    resp.encoding = 'utf-8'
    html = etree.HTML(resp.text)
    table = html.xpath('//*[@id="table"]')[0]
    trs = table.xpath('./tr')[1:-1]  # 剔除表头
    # trs = table.xpath('./tr[position()>1]')

    # 遍历每个 tr
    for tr in trs:
        txt = tr.xpath('./td/text()')
        # 对数据做简单的处理 例如 / \\
        txt = (item.strip() for item in txt)  # 迭代器
        # 数据写入
        csvwriter.writerow(txt)
        # print(list(txt))
    print(f'{url} complete.')
    # print(trs)


if __name__ == '__main__':

    f = open('foreign exchange data.csv', mode = 'w', encoding = 'gbk')
    csvwriter = csv.writer(f)
    # for i in range(1, 14870):   # page 14870  效率很低
    #     download_one_page(f'http://www.xinfadi.com.cn/marketanalysis/0/list/{i}.shtml')
    # download_one_page('http://www.waihuipaijia.cn/helandun.htm')

    contries = ['aomenyuan', 'helandun', 'hanguoyuan', 'feilvbinbisuo', 'xinxilanyuan', 'ruidiankelang',
           'yidalilila', 'danmaikelang', 'fenlanmake', 'faguofalang', 'deguomake']
    # 使用线程池
    with ThreadPoolExecutor(5) as t:  # 同时50个页面进行下载
        for i in range(10):
            t.submit(download_one_page(f'http://www.waihuipaijia.cn/{contries[i]}.htm'))

    print('All Complete.')
```

## 协程  多任务异步操作

**入门**

```python
# -*- coding: utf-8 -*-
# @time    : 2023/6/19 16:52
# @author  : w-xin
# @file    : coroutine_1.py
# @software: PyCharm

"""
协程入门

input() 程序也是处于阻塞状态
requests.get(bilibili) 在网络请求返回数据之前 程序也是处于阻塞状态的
一般情况下 当程序处于 IO 操作的时候 线程都会处于阻塞状态

协程  当程序遇见了 IO 操作时 可以选择性的切换到其他任务上
在微观上是一个任务一个任务进行切换 切换条件一般是 IO 操作
在宏观上 我们能看到的其实是多个任务一起执行
多任务异步操作 都是在单线程的条件下
"""
import time


def func():
    '''
    example
    :return:
    '''
    print('我爱黎明')
    time.sleep(3)  # 让当前的线程处于阻塞状态 CPU 是不为我工作的
    # 在单线程里面出现该语句时效率是很低的 因为在这三秒钟什么都没干
    print('我真的爱黎明')


if __name__ == '__main__':

    func()
```

**协程基础代码**

```python
# -*- coding: utf-8 -*-
# @time    : 2023/6/19 22:26
# @author  : w-xin
# @file    : coroutine_2.py
# @software: PyCharm

"""
coroutine_2
"""
import asyncio
import time


async def func():
    '''
    加一个前缀 async 变成异步执行
    变成异步协程函数 此时函数执行得到的是一个协程对象
    :return:
    '''
    print('Hello World')


# async def func1():
#     '''
#     function 1
#     :return:
#     '''
#     print('function one')
#     # time.sleep(3)
#     await asyncio.sleep(3)
#     print('function one executing...')
#
#
# async def func2():
#     '''
#     function 2
#     :return:
#     '''
#     print('function two')
#     # time.sleep(2)  # time.sleep(2) 是同步操作 出现同步操作时 异步就中断了
#     await asyncio.sleep(3)  # 异步操作代码 await 是挂起的意思
#     print('function two executing...')
#
#
# async def func3():
#     '''
#     function 3
#     :return:
#     '''
#     print('function three')
#     # time.sleep(5)
#     await asyncio.sleep(5)
#     print('function three executing...')
#
#
# if __name__ == '__main__':
#
#     f = func()
#     # 借助 asyncio 模块去运行协程函数
#     # asyncio.run(f)  # 由于是单个任务 能够运行但不见得高效
#
#     f1 = func1()
#     f2 = func2()
#     f3 = func3()
#
#     # 将任务都放进一个列表里面
#     tasks = [f1, f2, f3]  # python 3.8 以后的版本需要手动将 tasks 里面的东西包装成 Tasks 对象
#     tasks = [asyncio.create_task(f1), asyncio.create_task(f2), asuncio.create_task(f3)]
#
#     t1 = time.time()
#     # 使用协程一次性启动多个任务
#     asyncio.run(asyncio.wait(tasks))
#     t2 = time.time()
#     print(t2 - t1)

    '''
    但是以上代码一般不这么写 会导致 main 主线程任务居多
    
    可以改写成以下代码
    '''


async def func1():
    '''
    function 1
    :return:
    '''
    print('function one')
    # time.sleep(3)
    await asyncio.sleep(3)
    print('function one executing...')


async def func2():
    '''
    function 2
    :return:
    '''
    print('function two')
    # time.sleep(2)  # time.sleep(2) 是同步操作 出现同步操作时 异步就中断了
    await asyncio.sleep(3)  # 异步操作代码 await 是挂起的意思
    print('function two executing...')


async def func3():
    '''
    function 3
    :return:
    '''
    print('function three')
    # time.sleep(5)
    await asyncio.sleep(5)
    print('function three executing...')


async def main():
    '''
    在 main 主线程前写个 main 函数
    在这个函数里面希望将上方三个任务同时跑起来
    :return:
    '''
    # write method one
    # f1 = func1()
    # await f1  # await 一般放在协程对象前面
    # write methos two  recommodate
    tasks = [asyncio.create_task(func1()), asyncio.create_task(func2()), asyncio.create_task(func3())]
    await asyncio.wait(tasks)


if __name__ == '__main__':

    t1 = time.time()
    # 直接调用 main 函数
    asyncio.run(main())
    t2 = time.time()
    print(t2 - t1)

    # await f1  # await 在这里用不行 必须要有 async 前缀的函数里面
```

**进程爬虫模板**

```python
# -*- coding: utf-8 -*-
# @time    : 2023/6/19 23:00
# @author  : w-xin
# @file    : reptile_coroutine.py
# @software: PyCharm

"""
coroutine for reptile
could serve as moudle to use
"""
import asyncio


async def download(url):
    '''
    download module
    :param url: website url
    :return:
    '''
    print('download start')
    await asyncio.sleep(2)  # 模拟网络请求  requests.get()
    print('download complete')


async def main():
    '''
    main function
    :return:
    '''
    urls = ['url1', 'url2', 'url3']

    tasks = []
    for url in urls:
        d = asyncio.create_task(download(url))
        tasks.append(d)

    await asyncio.wait(tasks)

if __name__ == '__main__':

        asyncio.run(main())
```

**将爬虫程序中的同步代码换成异步代码**

```python
# -*- coding: utf-8 -*-
# @time    : 2023/6/20 10:47
# @author  : w-xin
# @file    : CoroutineToReptile.py.py
# @software: PyCharm

"""
coroutine adapt to reptile
use aiohttp module
"""
import asyncio
import aiohttp


async def aiodownload(url):
    '''
    发送请求 对 url 地址内容进行获取下载
    保存到文件
    发送请求的 Requests 需要替换成 aiohttp 中的代码进行协程操作
    :param url: 链接地址
    :return:
    '''
    file_name = url.rsplit('/', 1)[1]  # 右边切1次取第1个
    async with aiohttp.ClientSession() as session:  # aiohttp.ClientSession 相当于 requests
        # 使用 with 可以在 with 执行完成之后自动关闭 与文件操作相同
        async with session.get(url) as resp:  # 对 url 发出请求
            # resp.content.read()  # == resp.content()  读取文本的话就是 resp.text(); 原来是 resp.text
            with open(f'umIMG/{file_name}', mode = 'wb') as f:  # 文件读写也是 IO 操作 可以进行异步 另一个模块 aiofiles 进行异步
                f.write(await resp.content.read())  # 异步操作需要加 await

    print(file_name, 'complete!')



async def main():
    '''
    主函数中对多个 url 进行循环下载
    :return:
    '''
    tasks = [asyncio.create_task(aiodownload(url)) for url in urls]
    await asyncio.wait(tasks)



if __name__ == '__main__':

    urls = [
        'http://kr.shanghai-jiuxin.com/file/bizhi/20220929/ug153nfffzy.jpg',
        'http://kr.shanghai-jiuxin.com/file/bizhi/20220929/kqc3vgnbnct.jpg',
        'http://kr.shanghai-jiuxin.com/file/bizhi/20220929/j3kpwqx3ply.jpg'
    ]

    '''
    asyncio+aiohttp 出现 Exception ignored：RuntimeError('Event loop is closed'):
    像 aiohttp 这类第三方协程库都是依赖于标准库 asyncio 的，而 asyncio 对 Windows 的支持本来就不好。
    Python3.8 后默认 Windows 系统上的事件循环采用 ProactorEventLoop （仅用于 Windows ）这篇文档描述了
    其在 Windows 下的缺陷：https://docs.python.org/zh-cn/3/library/asyncio-platforms.html#windows 
    引发异常的函数是 _ProactorBasePipeTransport.__del__ ，所以 aiohttp 铁定使用了 _ProactorBasePipeTransport，
    并且在程序退出释放内存时自动调用了其__del__ 方法
    
    解决方法:
    1 不要使用 run 函数
    既然 _ProactorBasePipeTransport 会在程序结束后自动关闭事件循环，那就不要用 run 函数了，用官网的例子，使用 loop 
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    
    2 替换事件循环
    在调用 run 函数前，替换默认的 ProactorEventLoop 为 SelectorEventLoop
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
    但是 SelectorEventLoop 是有一些缺点的，比如不支持子进程等
    '''
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
```

## 协程案例

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-06-20%20192029.png)

```python
# -*- coding: utf-8 -*-
# @time    : 2023/6/20 19:22
# @author  : w-xin
# @file    : Case_BaiDuNovel.py
# @software: PyCharm

"""
coroutine to get Baidu Novel
the first url: https://dushu.baidu.com/api/pc/getCatalog?data={"book_id":"4306063500"}   %22 就是 "  得到所有章节的内容(名称 cid)
the second url: https://dushu.baidu.com/api/pc/getChapterContent?data={"book_id":"4306063500","cid":"4306063500|1569782244","need_bookinfo":1}  ==>  得到小说内容

操作流程
1 同步操作 访问 getCatalog 即 the first url  拿到所有章节的名称和 cid
2 异步操作 访问 getChapterContent 下载所有的文章内容
"""
import asyncio
import json

import aiofiles as aiofiles
import aiohttp
import requests


async def aiodownload(cid, book_id, title):
    '''
    下载小说内容
    :param cid: 章节 cid
    :param book_id: 书的 id
    :param title: 章节名
    :return:
    '''
    data = {
        "book_id": book_id,
        "cid": f"{book_id}|{cid}",
        "need_bookinfo": 1
    }
    # 需要把 data 变成 json 字符串
    data = json.dumps(data)
    url = f'https://dushu.baidu.com/api/pc/getChapterContent?data={data}'

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            # 读取 json 字典
            contentJson = await resp.json()
            # 从 json 中获取内容
            # data ==> novel ==> content
            content = contentJson['data']['novel']['content']
            # with open(f'XiYouJi/{title}.txt', mode = 'w', encoding = 'utf-8') as f:
            #     f.write(content)
            # 异步读写文件
            async with aiofiles.open(f'XiYouJi/{title}.txt', mode = 'w', encoding = 'utf-8') as f:
                await f.write(content)


    print(f'{title} download complete!')


async def getCatalog(url):
    '''
    获取小说名称和章节 cid
    :param url: 小说章节 url
    :return:
    '''
    resp = requests.get(url)
    # 使用 json 拿到 cid
    resp_jsonDict = resp.json()
    # data ==> novel ==> items
    tasks = []
    for item in resp_jsonDict['data']['novel']['items']:  # item 对应每个章节名称和 cid
        title = item['title']
        cid = item['cid']
        # 每一个 url 就是一个异步任务
        # 从这里开始准备异步
        tasks.append(asyncio.create_task(aiodownload(cid = cid, book_id = book_id, title = title)))

    await asyncio.wait(tasks)
    # print(resp_jsonDict)




if __name__ == '__main__':

    book_id = '4306063500'
    first_url = 'https://dushu.baidu.com/api/pc/getCatalog?data={"book_id":' + book_id + '}'

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(getCatalog(first_url))
```

# 视频网站工作原理

```html
<video src="kdsz.mp4"></video>
```

一般不会将视频设计成以上 html 代码

```python
# -*- coding: utf-8 -*-
# @time    : 2023/6/20 23:08
# @author  : w-xin
# @file    : 1_video_website_principle.py
# @software: PyCharm

"""
how to get a video
一般的视频网站的工作原理
1 用户上传 4k 的视频 --> 转码(对视频做不同清晰度的处理)  --> 切片处理(把单个文件进行拆分)
2 用户在拉动进度条的时候
  ========================若用户在此处开始播 那么只需要加载前一个或一部分的视频切片即可====================== (进度条)
3 拆分的时候视频必须按顺序播放 就需要一个文件记录 -视频播放顺序 -视频存放的路径
4 一般情况下会把拆分好的视频放在 M3U 文件里面
  M3U 文件经过 utf-8 编码 就变成 M3U8 文件

想要抓取一个视频 基本的流程
1 找到 m3u8 (通过各种手段)
2 通过 m3u8 下载到 ts 文件
3 可以通过各种手段 (不仅是编程手段) 把 ts 文件合并为一个 mp4 文件
"""
```

![](./%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-06-20%20233001.png)

