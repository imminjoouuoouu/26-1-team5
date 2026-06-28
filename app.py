import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

# 페이지 제목 및 설정
st.set_page_config(page_title="수면 건강 종합 진단", layout="centered")
st.title("🛌 수면 건강 종합 진단 서비스")
st.markdown("---")

@st.cache_resource
def load_and_train():
    if not os.path.exists('Sleep_health_and_lifestyle_dataset.csv'):
        return None, None, None, None
        
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('Normal')
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    bmi_map = {'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Obese': 2}
    df['BMI Category'] = df['BMI Category'].map(bmi_map)
    
    def estimate_heart_rate(sbp, dbp):
        return int(np.clip(0.25 * sbp + 0.35 * dbp + 25, 45, 120))
    if 'Heart Rate' not in df.columns or df['Heart Rate'].isnull().any():
        df['Heart Rate'] = df.apply(lambda r: estimate_heart_rate(r['Systolic_BP'], r['Diastolic_BP']), axis=1)
        
    FEATURE_COLS = ['Gender', 'Age', 'Quality of Sleep', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']
    X = df[FEATURE_COLS]
    y = df['Sleep Disorder']
    
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42)
    model.fit(X_sc, y)
    
    knn_scaler = StandardScaler()
    knn_scaled = knn_scaler.fit_transform(df[FEATURE_COLS])
    knn_model = NearestNeighbors(n_neighbors=8, metric='euclidean')
    knn_model.fit(knn_scaled)
    
    return df, scaler, model, knn_model

df, scaler, model, knn_model = load_and_train()

if df is None:
    st.error("⚠️ 'Sleep_health_and_lifestyle_dataset.csv' 파일을 찾을 수 없습니다. Kaggle 다운로드를 먼저 실행해 주세요.")
    st.stop()

def assign_persona(disorder, stress, quality):
    if disorder == 'Normal':
        if quality >= 7 and stress <= 4:
            return {"type": "🌙 Deep Rest (안정적 수면형)", "risk": "낮음", "desc": "부교감신경이 활성화된 심리적 안정 상태입니다. PSQI 기준 매우 우수한 수면 상태입니다."}
        else:
            return {"type": "🏃 Recovery Booster (활동 회복형)", "risk": "낮음", "desc": "글림프 시스템을 통한 뇌 노폐물 제거가 진행 중인 상태입니다. 수면 질 7점 이상이 목표입니다."}
    elif disorder == 'Insomnia':
        if stress >= 6:
            return {"type": "⚡ Stress Sleeper (스트레스 과부하형)", "risk": "높음", "desc": "코르티솔 과다 분비로 HPA축이 과활성화되어 멜라토닌 분비가 억제된 상태입니다."}
        else:
            return {"type": "🔄 Irregular Rhythm (수면 리듬 불균형형)", "risk": "중간", "desc": "일주기리듬(Circadian Rhythm) 붕괴 상태로, 취침·기상 시각 고정이 최우선입니다."}
    else:
        return {"type": "🫁 Apnea Risk (호흡 불안형)", "risk": "높음", "desc": "수면 중 기도 협착으로 반복적 호흡 정지가 발생, 산소포화도 저하가 심혈관계에 부담을 줍니다."}

def get_knn_targets(user_raw):
    FEATURE_COLS = ['Gender', 'Age', 'Quality of Sleep', 'Stress Level', 'BMI Category', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']
    knn_scaler = StandardScaler()
    knn_scaled = knn_scaler.fit_transform(df[FEATURE_COLS])
    user_df = pd.DataFrame([user_raw])
    user_sc = knn_scaler.transform(user_df)
    _, indices = knn_model.kneighbors(user_sc)
    similar = df.iloc[indices[0]]
    good = similar[similar['Quality of Sleep'] >= 7]
    ref = good if len(good) >= 2 else similar
    return {
        "target_daily_steps": int(ref['Daily Steps'].mean()),
        "target_stress_level": round(ref['Stress Level'].mean(), 1),
        "target_quality": round(ref['Quality of Sleep'].mean(), 1),
    }

def gemini_medical_analysis_mock(persona, user_raw, knn):
    return {
        "medical_summary": f"입력하신 데이터 분석 결과, 현재 {persona['type']} 경향이 뚜렷하게 관찰됩니다. 자율신경계 균형과 호르몬 분비 주기를 정상화하기 위한 라이프스타일 교정이 요구됩니다.",
        "mechanism": "과도한 신체적/심리적 스트레스 유인 혹은 수면 환경 불일치로 인해 생체 시계인 일주기 리듬이 일시적으로 교란된 상태입니다.",
        "recommended_sleep_hours": "7~8시간",
        "missions": [
            "취침 1시간 전 스마트폰 및 블루라이트 완전히 차단하기",
            "실내 온도를 약간 서늘한 18~20°C로 세팅하여 심부체온 낮추기",
            "눕기 전 4-7-8 호흡법을 5분간 실시하여 교감신경 가라앉히기"
        ],
        "warning_signs": "주 3회 이상의 수면 곤란이 2주 이상 지속될 경우, 수면전문의 전문 상담을 권장합니다."
    }

st.header("📋 1단계: 수면 및 라이프스타일 정보 입력")
col1, col2 = st.columns(2)
with col1:
    gender_input = st.selectbox("성별을 선택하세요", ["여성", "남성"])
    gender = 1 if gender_input == "남성" else 0
with col2:
    age = st.slider("나이 (세)", 10, 100, 30)

col3, col4, col5 = st.columns(3)
with col3:
    bmi_input = st.selectbox("BMI 체형 분류", ["정상 체중", "과체중", "비만"])
    bmi_cat = {"정상 체중": 0, "과체중": 1, "비만": 2}[bmi_input]
with col4:
    sbp = st.number_input("수축기 혈압 (최고)", value=120)
with col5:
    dbp = st.number_input("이완기 혈압 (최저)", value=80)

steps = st.number_input("하루 평균 걸음 수 (보)", value=7000, step=500)

st.subheader("📊 주관적 상태 평가")
sleep_score = st.slider("최근 나의 수면의 질 점수 (1~10)", 1, 10, 5)
stress_score = st.slider("최근 나의 스트레스 지수 (1~10)", 1, 10, 5)

hr = int(np.clip(0.25 * sbp + 0.35 * dbp + 25, 45, 120))
st.info(f"💓 혈압 기반 자동 추정 심박수: **{hr} bpm**")
st.markdown("---")

if st.button("🚀 종합 수면 건강 분석하기", type="primary"):
    user_raw = {
        'Gender': gender, 'Age': age, 'Quality of Sleep': sleep_score,
        'Stress Level': stress_score, 'BMI Category': bmi_cat,
        'Heart Rate': hr, 'Daily Steps': steps, 'Systolic_BP': sbp, 'Diastolic_BP': dbp
    }
    user_df = pd.DataFrame([user_raw])
    user_sc = scaler.transform(user_df)
    disorder = model.predict(user_sc)[0]
    proba = model.predict_proba(user_sc)[0]
    proba_dict = {c: f"{round(p*100, 1)}%" for c, p in zip(model.classes_, proba)}
    
    persona = assign_persona(disorder, stress_score, sleep_score)
    knn = get_knn_targets(user_raw)
    gd = gemini_medical_analysis_mock(persona, user_raw, knn)
    
    st.header("🎯 분석 결과 레포트")
    st.subheader(f"👤 배정된 수면 페르소나: **{persona['type']}**")
    st.warning(f"**위험도 레벨:** {persona['risk']} | **상태 설명:** {persona['desc']}")
    
    st.subheader("🩺 AI 수면 장애 위험도 예측 (Random Forest)")
    st.write(f"가장 유력한 진단: **{disorder}**")
    st.table(pd.DataFrame([proba_dict], index=["예측 확률"]))
    
    st.subheader("🎯 데이터 분석 기반 추천 목표 (KNN 유사 그룹 비교)")
    col_a, col_b, col_col = st.columns(3)
    col_a.metric(label="목표 걸음 수", value=f"{knn['target_daily_steps']:,} 보", delta=int(knn['target_daily_steps'] - steps))
    col_b.metric(label="목표 스트레스 지수", value=f"{knn['target_stress_level']}/10", delta=round(knn['target_stress_level'] - stress_score, 1), delta_color="inverse")
    col_col.metric(label="목표 수면의 질", value=f"{knn['target_quality']}/10", delta=round(knn['target_quality'] - sleep_score, 1))

    st.markdown("---")
    st.subheader("🔬 의학 전문가 종합 코멘트 (By Gemini)")
    st.write(gd['medical_summary'])
    st.caption(f"**생리적 메커니즘:** {gd['mechanism']}")
    st.info(f"⏰ **당신에게 권장하는 일일 수면 시간:** {gd['recommended_sleep_hours']}")
    
    st.subheader("✅ 오늘 밤 실천할 행동 미션")
    for idx, mission in enumerate(gd['missions'], 1):
        st.checkbox(f"{idx}. {mission}", key=f"m_{idx}")
        
    st.error(f"⚠️ **전문의 상담 기준:** {gd['warning_signs']}")
