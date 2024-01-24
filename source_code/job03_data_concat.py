import pandas as pd
import glob
import datetime

data_path = glob.glob("../crawling_data/*")

df = pd.DataFrame()

for path in data_path:
    df_temp = pd.read_csv(path)
    df_temp.dropna(inplace = True)
    df = pd.concat([df, df_temp], axis = "rows", ignore_index = True)

print(df.head())
print(df["category"].value_counts())
print(df.info())

df.to_csv("../datasets/naver_news_titles_{}.csv".format(datetime.datetime.now().strftime("%Y%m%d")), index = False)