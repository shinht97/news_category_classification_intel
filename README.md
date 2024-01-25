# news_category_classification_intel

네이버 뉴스의 제목을 이용하여 카테고리를 예측하는 프로그램

사용 패키지는 requirements.txt 참고

job01 실행시 오늘자 헤드라인 뉴스를 카테고리별로 클로링, csv 파일 저장

job02 실행시 오늘자 뉴스의 제목을 카테고리별로 크롤링, csv 파일 저장

job03 실행시 csv 파일 합침

job04 실행시 전처리 과정으로, 뉴스 제목의 tokenize, 카테고리의 onehot_encoding 진행

job05 실행시 모델 학습 진행, 제목 최대값 max, 토큰의 수 wordsize가 필요

job06 실행시 뉴스 제목의 카테고리 분류 예측 실행