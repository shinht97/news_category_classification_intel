from bs4 import BeautifulSoup  # pip install bs4
import requests  # pip install requests
                 # 프로그램을 클라이언트로 사용하기 위한 패키지
import re  # python에서 정규식 표현식 관련 하여 사용 하기 위한 패키지
import pandas as pd
import datetime

category = ["Politics", "Economics", "Social", "Culture", "World", "IT"]

# url = "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100#&date=%2000:00:00&page=1"

# headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
# 브라우저가 요청하는 것 처럼 만들어 주기 위해 추가

# resp = requests.get(url, headers = headers)  # 클라이언트처럼 동작하여 서버에 요청

# print(resp)
# print(resp.text)

# soup = BeautifulSoup(resp.text, "html.parser")  # html 문서 형태로 바꿔줌

# print(soup)

# title_tags = soup.select(".sh_text_headline")  # (.) : class 라는 의미

# print(title_tags)
# print(len(title_tags))
# print(type(title_tags[0]))

# titles = []
# 
# for title_tag in title_tags:
#     titles.append(re.compile("[^가-힣|a-z|A-Z]").sub(" ", title_tag.text))  # "[^가-힣|a-z|A-Z]" : 한글, 알파벳 대소문자를 제외(^) : 정규 표현식
#                                                                                 # sub : 타겟 문자열에서 지정 문자를 빼고 지정 문자로 변환
# 
# print(titles)

df_titles = pd.DataFrame()  # 빈 데이터 프레임을 생성
re_title = re.compile("[^가-힣|a-z|A-Z]")  # 사용할 정규 표현식을 지정

headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}  # 클라이언트 헤더 지정

for i in range(6):
    url = "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=10{}#&date=%2000:00:00&page=1".format(i)
    resp = requests.get(url, headers = headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    title_tags = soup.select(".sh_text_headline")

    titles = []

    for title_tag in title_tags:
        titles.append(re_title.sub(" ", title_tag.text))

    df_section_titles = pd.DataFrame(titles, columns = ["titles"])
    df_section_titles["category"] = category[i]

    df_titles = pd.concat([df_titles, df_section_titles], 
                          axis = "rows", ignore_index = True)  # ignore_index : 기존 인덱스 무시
    
print(df_titles.head())
print(df_titles.info())
print(df_titles["category"].value_counts())

df_titles.to_csv(
    "../crawling_data/naver_headline_news_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d")),
    # datetime.datetime.now() : 현재 시간(ns 단위)
    # strftime(%Y%m%d) : 지정 문자열로 시간을 표시, %Y : 년(20xx), %m : 월, %d : 일
    index = False
)
