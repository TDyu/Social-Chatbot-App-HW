#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Crawler for PPT WorkinChina.
有二功能
1. 搜集多頁的文章資訊, 並且可選是否輸出 csv
2: 搜尋結果, 並且可選是否輸出 csv
各功能簡略抓取過程步驟請看其負責函數裡面的註解
每一步驟詳細說明, 請再跳到負責執行當步驟的函數裡頭看其註解
"""
import re
import os
import time

import requests
import urllib.parse
from requests_html import HTML  # 整合 lxml 與 PyQuery 的套件
import pandas as pd


def get_response_text(url, params_dict=None):
    """取得網頁內容
    使用 "GET" 方法, 取得網頁內容 response
    且為防止某些"第一次"訪問時會有"確認年齡是否滿十八歲"的測試, 需要把回答滿 18 的 cookies 記錄下來
    接續便能以此 cookies 訪問網頁, 假裝已經通過滿十八歲測試
    最後回傳其網頁內容的 text

    params_dict: 可以選擇要不要帶參數
    """
    # 第一次訪問, 記錄下回答滿 18 的 cookies 為 {'over18': '1'}
    response = requests.get(url)
    # 接下來帶這個 cookie 進行訪問, 便躲過年齡測試
    if params_dict:
        response = requests.get(url, cookies={'over18': '1'}, params=params_dict)
    else:
        response = requests.get(url, cookies={'over18': '1'})
    return response.text


def parse_article_elements(response_text):
    """解析當前 response text, 取得所有文章標題訊息的 requests_html.Element 包裝物件
    首先藉由瀏覽器的開發人員工具對 PTT 網頁原始碼做觀察之後, 發現文章的標題訊息會放在 class="r-ent" 的 div 標籤裡
    利用requests_html.HTML 對 response text 做出解析, 包裝成 HTML() 物件
    再使用 find & CSS selector 操作 HTML() 物件, 並指定尋找目標為 class="r-ent" 的 div 標籤
    每一個目標都會被包裝成 requests_html.Element 物件（內含文章訊息, 將在下一步做處理）, 找尋過後會是一個 list
    最後回傳這個包含所有文章標題訊息元素物件的 list
    """
    html = HTML(html=response_text)
    article_elements_list = html.find('div.r-ent')
    return article_elements_list


def parse_article_information(article_element):
    """解析此筆 requests_html.Element 物件, 包裝其 text 結果成 dict
    首先藉由瀏覽器的開發人員工具對 PTT 網頁做觀察
    發現 推文數 (push)、標題 (title)、作者 (author)、發文日期 (date) 和文章網址 (link) 分別在 class="r-ent" 的 div 標籤下層的哪個標籤
    再使用 find & CSS selector, 解析此筆 requests_html.Element 物件
    將這五個資訊轉成 text, 包裝成一個 dict 回傳

    但觀察後發現若該頁有文章被刪除時, "本文已被刪除"這個元素的原始碼長的和一般文章標題訊息不一樣
    被刪除的文章僅保留推文數、標題名稱、發文日期
    所以要對此做例外處理
    """
    # 不管有沒有被刪除, 都保有推文數、標題名稱、發文日期的資訊
    article_information_dict = {
        'title': article_element.find('div.title', first=True).text,
        'push': article_element.find('div.nrec', first=True).text,
        'date': article_element.find('div.date', first=True).text,
    }

    try:
        # 嘗試取得作者、文章網址的資訊, 並且添加到 dict
        # 若文章未被刪除便能獲取
        article_information_dict['author'] = article_element.find(
            'div.author', first=True).text
        article_information_dict['link'] = 'https://www.ptt.cc/' + article_element.find('div.title > a', first=True).attrs['href']
    except AttributeError:
        # 對被刪除的文章做處理
        if '(本文已被刪除)' in article_information_dict['title']:
            # 因為被刪除的文和文章的標題都可能含有'(本文已被刪除)'
            # 但是真正被刪除的文章在原始碼中有其特殊結構
            # 所以利用正則式再進行確認
            match_author = re.search(
                '\[(\w*)\]', article_information_dict['title'])
            if match_author:
                article_information_dict['author'] = match_author.group(1)
        elif re.search('已被\w*刪除', article_information_dict['title']):
            # e.g., "(已被cappa刪除) <edisonchu> op"
            match_author = re.search('\<(\w*)\>', article_information_dict['title'])
            if match_author:
                article_information_dict['author'] = match_author.group(1)
    
    return article_information_dict


def get_next_page_url(start_url):
    """取得前往下一頁的 url
    觀察原始碼, 發覺前往下一頁的連結放在 <div class='action-bar> 的 <a class='btn wide'> 裡
    """
    response_text = get_response_text(start_url)
    html = HTML(html=response_text)
    controls = html.find('.action-bar a.btn.wide')
    link = controls[1].attrs.get('href')
    return urllib.parse.urljoin('https://www.ptt.cc/', link)


def get_many_next_page_urls(url, pages=3):
    """取得多頁（預設為 3 頁）的 link, 裝成 list 回傳
    """
    next_urls = []

    for _ in range(pages):
        url = get_next_page_url(url)
        next_urls.append(url)

    return next_urls


def get_many_page_article_infomations(urls):
    """整合多頁的文章標題訊息物件
    """
    article_elements_list = []
    for url in urls:
        response_text = get_response_text(url)
        article_elements_list += parse_article_elements(response_text)
    return article_elements_list


def get_article_information_dict_list(article_elements_list, output_path=None):
    """解析每一個文章標題訊息物件, 取得其文章資訊的 dict, 整合成一個 dataframe, 並輸出 csv（若需要）
    """
    # 先創建一個空的 dataframe
    df = pd.DataFrame(columns=['title', 'push', 'date', 'author', 'link'])

    # 迭代每個文章標題訊息物件, 取得其文章資訊, 增添新一行到 dataframe 中
    index = 0
    for article_element in article_elements_list:
        article_information_dict = parse_article_information(article_element)
        df.loc[index] = article_information_dict
        index += 1
    
    # 若需保存 cvs 檔案
    if output_path:
        # 拆分欲保存檔案的目錄路徑與檔案名稱
        # 若非同資料夾下保存, 需先檢查目錄是否存在, 若不存在則先創建
        folder_path = os.path.split(output_path)[0]
        if folder_path != './' and folder_path != '/' and folder_path != '.':
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        df.to_csv(output_path, encoding='utf-8')

    return df


def article_informations(start_url, pages=3, output_path=None):
    """功能1: 搜集多頁(預設 3 頁)的文章資訊, 並且可選是否輸出 csv
    """
    # 1. 躲過年齡測驗, 取得網頁內容 text
    response_text = get_response_text(start_url)

    # 2. 取得多個下一頁的連接
    next_urls = get_many_next_page_urls(start_url, pages=pages)

    # 3. 解析每一頁的 response text, 取得所有文章標題訊息的 requests_html.Element 包裝物件
    article_elements_list = []
    # 解析當前頁面
    article_elements_list += parse_article_elements(response_text)
    # 解析其他多個下一頁
    article_elements_list += get_many_page_article_infomations(next_urls)

    # 4. 解析每一個文章標題訊息物件, 取得其文章資訊的 dict, 整合成一個 dataframe, 並輸出 csv（若需要）
    if article_informations_output_path:
        return get_article_information_dict_list(
            article_elements_list, output_path=output_path)
    else:
        return get_article_information_dict_list(article_elements_list)


def search_result(search_endpoint_url, keyword, assigned_page=None, is_same_article=False, is_author=False, is_recommend_number=False, output_path=None):
    """功能2: 搜尋結果, 並且可選是否輸出 csv
    觀察出搜尋的 api 開頭為 https://www.ptt.cc/bbs/WorkinChina/search
    並且用以篩選的搜尋結果的後綴參數狀況如下:
        搜尋關鍵字: 'q': 搜尋關鍵字
        搜尋相同文章: 將"thread:"加到搜尋關鍵字前
        搜尋相同作者文章: "author:"+作者名字 作為搜尋關鍵字
        搜尋推文數大於多少: "recommend:"+希望搜尋到的最低推文數 作為搜尋關鍵字
        指定搜尋那一頁: 'page': 指定的頁碼

    其他獲取與保存辦法與功能1一樣
    """
    params = {'q': ''}
    if assigned_page:
        params['page'] = int(assigned_page)

    # 1. 組合正確搜尋字串
    if is_author:
        params['q'] = 'author:' + str(keyword)
    elif is_recommend_number:
        params['q'] = 'recommend:' + str(keyword)
    elif is_same_article:
        params['q'] = 'thread:' + str(keyword)
    else:
        params['q'] = str(keyword)
    print(params)
    
    # 2. 躲過年齡測試, 取得搜尋結果頁面 response
    response_text = get_response_text(search_endpoint_url, params_dict=params)

    # 3. 解析此頁的 response text, 取得所有文章標題訊息的 requests_html.Element 包裝物件
    article_elements_list = parse_article_elements(response_text)

    # 4. 解析每一個文章標題訊息物件, 取得其文章資訊的 dict, 整合成一個 dataframe, 並輸出 csv（若需要）
    if output_path:
        return get_article_information_dict_list(
            article_elements_list, output_path=output_path)
    else:
        return get_article_information_dict_list(article_elements_list)
        


if __name__ == "__main__":
    # 從"最新"的頁面開始
    START_URL = 'https://www.ptt.cc/bbs/WorkinChina/index.html'
    # 觀察出搜尋的 api 開頭
    SEARCH_ENDPOINT_URL = 'https://www.ptt.cc/bbs/WorkinChina/search'

    # 各個保存路徑
    article_informations_output_path = './article_informations.csv'
    search_article_information_output_path = './search_article_informations.csv'

    # # 功能1: 搜集多頁的文章資訊, 並且可選是否輸出 csv
    # start = time.time()
    # result_pd = article_informations(START_URL, pages=10,
    #                      output_path=article_informations_output_path)
    # print(result_pd)
    # print('搜集多頁的文章資訊 - Speed time: %f seconds' % (time.time() - start))

    # 功能2: 搜尋結果, 並且可選是否輸出 csv
    start = time.time()
    result_pd = search_result(SEARCH_ENDPOINT_URL, '廣州', assigned_page=2, is_same_article=False,
                  is_author=False, is_recommend_number=False, output_path=search_article_information_output_path)
    # result_pd = search_result(SEARCH_ENDPOINT_URL, '上海', assigned_page=2, is_same_article=False,
    #               is_author=False, is_recommend_number=False, output_path=search_article_information_output_path)
    # result_pd = search_result(SEARCH_ENDPOINT_URL, '北京', assigned_page=2, is_same_article=False,
    #               is_author=False, is_recommend_number=False, output_path=search_article_information_output_path)
    print(result_pd)
    print('搜尋結果 - Speed time: %f seconds' % (time.time() - start))
