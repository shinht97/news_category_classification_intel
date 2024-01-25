# 일반적인 웹 페이지는 주소가 랜덤함
# 직접 접속하여 존재하는 데이터를 얻는 방식

# selenium을 이용하여 웹페이지에 접속
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException

import pandas as pd
import re
import time
import datetime

category = ["Politics", "Economics", "Social", "Culture", "World", "IT"]
pages = [105, 105, 105, 81, 105, 81]

options = ChromeOptions()
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
options.add_argument("user_agent=" + user_agent)
options.add_argument("lang=ko_KR")

service = ChromeService(executable_path = ChromeDriverManager().install())

# 웹 페이지 객체를 생성
driver = webdriver.Chrome(service = service, options = options)

df_titles = pd.DataFrame()

for l in range(1, 6):
    section_url = "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}".format(l)
    titles = []
    for k in range(1, pages[l]):
        url = section_url + "#&date=%2000:00:00&page={}".format(k)
        try:
            driver.get(url) #
            time.sleep(0.5)  # 딜레이 함수

        except:
            print(f"driver.get {l}, {k}")

        for i in range(1, 5):
            for j in range(1, 6):
                try:
                    title = driver.find_element("xpath",
                                                '//*[@id="section_body"]/ul[{0}]/li[{1}]/dl/dt[2]/a'.format(i, j)).text
                # 특정 element(xpath)를 찾음
                    title = re.compile("[^가-힣]").sub(" ", title)

                    titles.append(title)
                except:
                    print(f"find element {l}, {k}, {i}, {j}")

        if k % 5 == 0:
            print(f"working at section {l} page {k}")
            df_section_title = pd.DataFrame(titles, columns=["titles"])
            df_section_title["category"] = category[l]
            # print(df_section_title)
            df_section_title.to_csv("../crawling_data/data_{0}_{1}.csv".format(l, k),
                                    index = False, encoding = "utf-8")
            titles = [] # 데이터 중복 방지를 위해 titles를 비움
            # df_titles = pd.concat([df_titles, df_section_title],
            #                       axis="rows", ignore_index=True)

driver.close()  # 브라우저 창을 닫음

