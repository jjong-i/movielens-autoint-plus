import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from autoint_plus import AutoIntPlusTF

@st.cache_resource
def load_assets():
    base_path = 'aiffel/autoint/ml-1m/'
    movies = pd.read_csv(base_path + 'movies.dat', sep='::', engine='python', 
                         names=['MovieId', 'Title', 'Genres'], encoding='latin-1')
    encoders = joblib.load('label_encoders.pkl')
    field_dims = np.load('field_dims.npy')
    
    # AutoInt+ 모델 로드
    model = AutoIntPlusTF(field_dims)
    model(np.zeros((1, len(field_dims)), dtype=np.int32)) 
    model.load_weights('autoInt_plus_tf.weights.h5')
    
    return movies, encoders, model

st.set_page_config(page_title="MovieLens 추천 시스템", layout="wide")
st.title("🎬 MovieLens 고도화 추천 시스템 (AutoInt+)")

movies, encoders, model = load_assets()

with st.sidebar:
    user_id = st.number_input("User ID (1-6040)", 1, 6040, 2)
    btn = st.button("고도화된 추천 받기")

if btn:
    user_idx = encoders['UserId'].transform([user_id])[0]
    all_movie_ids = movies['MovieId'].values
    
    # 예측용 데이터 생성
    test_inputs = []
    valid_movies = []
    for m_id in all_movie_ids:
        try:
            m_idx = encoders['MovieId'].transform([m_id])[0]
            test_inputs.append([user_idx, m_idx, 0, 0, 0, 0])
            valid_movies.append(m_id)
        except: continue

    preds = model.predict(np.array(test_inputs), batch_size=512)
    top_10_idx = np.argsort(preds)[-10:][::-1]
    
    st.subheader(f"👤 {user_id}번 유저를 위한 추천 목록")
    res = movies[movies['MovieId'].isin([valid_movies[i] for i in top_10_idx])].copy()
    res['Score'] = [preds[i] for i in top_10_idx]
    st.table(res[['Title', 'Score']])
