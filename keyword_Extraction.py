from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
import pandas as pd
import json
import re
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder='templates')

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi'}

# 특수 문자 및 기호 제거
def cleaning(texts, punct, mapping):
    cleaned_texts = []
    for text in texts:
        for p in mapping:
            text = text.replace(p, mapping[p])

        for p in punct:
            text = text.replace(p, '')
        text = re.sub(r'<[^>]+>', '', text)   # Html 태그 제거
        text = re.sub(r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', '', text)   # e-mail 제거
        text = re.sub(r'(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', ' ', text)   # URL 제거
        text = re.sub(r'\s+', ' ', text)       # 여백 제거
        text = re.sub('([ㄱ-ㅎㅏ-ㅣ]+)', '', text)  #  한글 자음, 모음 제거
        text = re.sub('[^\w\s\n]', '', text)   # 특수기호 제거
        cleaned_texts.append(text)

    return cleaned_texts

# 키워드 추출에 필요한 열 추출(text(본문)만 가지고 추출 진행)
def extract(df):
    ex_df = df[['data_id', 'text']]
    return ex_df

# mecab 형태소 분석기 명사 추출 & 불용어 제거
def mecab(ex_df):
    npl = Mecab()

    with open('./watson_ko/stopwords.txt', 'r', encoding='utf-8') as file:
        stopwords = set([word.strip() for word in file.readlines()])

    tokenized_texts = {}
    for d_id in ex_df['data_id'].unique():
        texts = ex_df[ex_df['data_id'] == d_id]['text'].tolist()
        word_tokens = []
        for text in texts:
            tokens = npl.nouns(text)
            filtered_tokens = [token for token in tokens if token not in stopwords]
            word_tokens += filtered_tokens
        tokenized_texts[d_id] = word_tokens
        
    return tokenized_texts


def tfifd(tokenized_texts):
    tfidf_vec = TfidfVectorizer()
    tfidf_df_dict = {}

    # 날짜별로 TF-IDF 계산
    for date, token in tokenized_texts.items():
        documents = tokenized_texts[date]
        document_text = ' '.join(documents)
        # TF-IDF 계산
        tfidf_matrix = tfidf_vec.fit_transform([document_text])

        tfidf_df_dict[date] = pd.DataFrame(tfidf_matrix.toarray().tolist(), columns=tfidf_vec.get_feature_names_out())

    top_15_tfidf_dict = {}

    # 상위 15개 키워드 추출
    for date, df in tfidf_df_dict.items():
        top_15_tfidf_dict[date] = df.apply(lambda row: row.nlargest(15).index.tolist(), axis=1)

    top_15_tfidf_df = pd.DataFrame.from_dict(top_15_tfidf_dict).T

    top_15_tfidf_df.rename(columns = {0:'keyword'}, inplace=True)

    temp_1 =top_15_tfidf_df['keyword'].apply(pd.Series)

    return temp_1

@app.route('/')
def index():
    return render_template('./index.html')

@app.route('/upload', methods=['POST'])
def upload_json_file():
    # 파일을 받는 엔드포인트
    
    # 업로드된 파일들 가져오기
    uploaded_files = request.files.getlist('files')
    
    # 파일을 저장할 디렉토리 경로
    upload_dir = './uploaded'
    os.makedirs(upload_dir, exist_ok=True)
    
    # 각 파일 저장
    for uploaded_file in uploaded_files:
        uploaded_file.save(os.path.join(upload_dir, uploaded_file.filename))
    
    return redirect(url_for('process'))

@app.route('/process')
def process():
    # 파일을 처리하는 엔드포인트
    
    # 디렉토리 내의 파일 목록 가져오기
    directory = './uploaded'
    file_names = os.listdir(directory)
    
    # 모든 JSON 파일의 데이터 프레임을 저장할 빈 리스트 생성
    all_dfs = []

    # 각 파일에 대해 반복하여 데이터 프레임 생성
    for file_name in file_names:
        if file_name.endswith('.json'):
            file_path = os.path.join(directory, file_name)

            # JSON 파일 열기
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # json 객체 중 'data_investing' 만 추출
                data_investing = data['data_investing']
                df = pd.DataFrame(data_investing)

                # json 객체 'data_id' 추가
                df['data_id'] = data['data_id']

                # 리스트에 데이터 프레임 추가
                all_dfs.append(df)

    # 모든 데이터 프레임을 하나의 데이터 프레임으로 연결
    final_df = pd.concat(all_dfs, ignore_index=True)

    final_df['text'] = cleaning(final_df['text'], punct, punct_mapping)
    ex_df = final_df[['data_id', 'text']]
    
    proc_1 = extract(ex_df)
    proc_2 = mecab(proc_1)
    proc_3 = tfifd(proc_2)
    
    return jsonify(proc_3.T.to_dict())


if __name__ == '__main__':
    app.run(debug=True)