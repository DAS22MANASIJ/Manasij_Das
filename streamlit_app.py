import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
import io
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='CrewSignal',layout='wide')
st.markdown("""<style>body{background:linear-gradient(180deg,#021124,#04293a);color:#e6f0f6} .stButton>button{background:linear-gradient(90deg,#00f5d4,#6a82fb);color:#001219;font-weight:700}</style>""",unsafe_allow_html=True)

def load_dataset(path='Health data.csv'):
    with open(path,'r',encoding='utf-8') as f:
        lines=[ln.rstrip('\n') for ln in f if ln.strip()]
    cleaned=[]
    for ln in lines:
        if '|' in ln:
            parts=ln.split('|',1)
            rest=parts[1]
        else:
            rest=ln
        cleaned.append(rest.strip())
    data='\n'.join(cleaned)
    df=pd.read_csv(io.StringIO(data))
    cols={c:c.strip() for c in df.columns}
    df.rename(columns=cols,inplace=True)
    colmap={}
    for c in df.columns:
        lc=c.lower()
        if 'pulse' in lc:
            colmap[c]='Pulse'
        elif 'temp' in lc or 'temperature' in lc:
            colmap[c]='Temperature'
        elif 'spo2' in lc:
            colmap[c]='SPO2'
        elif 'status' in lc or 'label' in lc:
            colmap[c]='Label'
    df.rename(columns=colmap,inplace=True)
    if 'Label' not in df.columns:
        st.error('Dataset missing Label/Status column')
        st.stop()
    features=[]
    for f in ['Pulse','Temperature','SPO2']:
        if f in df.columns:
            features.append(f)
    if not features:
        st.error('Dataset missing required features')
        st.stop()
    df=df[features+['Label']].dropna()
    df['Label']=df['Label'].astype(int)
    return df,features

df,FEATURES=load_dataset()
X=df[FEATURES].values
y=df['Label'].values
scaler=StandardScaler().fit(X)
Xs=scaler.transform(X)
X_train,X_test,y_train,y_test=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
rf=RandomForestClassifier(n_estimators=30,max_depth=5,random_state=42)
svm=SVC(kernel='rbf',probability=True,C=1.0,random_state=42)
rf.fit(X_train,y_train)
svm.fit(X_train,y_train)

def metrics(model,Xt,yt):
    yp=model.predict(Xt)
    return dict(accuracy=float(accuracy_score(yt,yp)),precision=float(precision_score(yt,yp,average='weighted',zero_division=0)),recall=float(recall_score(yt,yp,average='weighted',zero_division=0)),f1=float(f1_score(yt,yp,average='weighted',zero_division=0)),confusion_matrix=confusion_matrix(yt,yp).tolist())

rf_metrics=metrics(rf,X_test,y_test)
svm_metrics=metrics(svm,X_test,y_test)
rf_cv=float(np.mean(cross_val_score(rf,Xs,y,cv=3)))
svm_cv=float(np.mean(cross_val_score(svm,Xs,y,cv=3)))

st.title('CrewSignal')
col1,col2=st.columns([1,2])
with col1:
    astronaut_id=st.text_input('Astronaut ID')
    astronaut_name=st.text_input('Astronaut Name')
    age=st.number_input('Age',min_value=18,max_value=100,value=30)
    sex=st.text_input('Sex (M/F)')
    pulse=st.number_input('Pulse',min_value=30.0,max_value=200.0,value=70.0)
    temperature=st.number_input('Body Temperature',min_value=30.0,max_value=42.0,step=0.1,value=36.6)
    spo2=st.number_input('SPO2',min_value=50.0,max_value=100.0,value=98.0)
    if st.button('Predict'):
        try:
            features_vals=[]
            for f in FEATURES:
                if f=='Pulse':
                    features_vals.append(float(pulse))
                elif f=='Temperature':
                    features_vals.append(float(temperature))
                elif f=='SPO2':
                    features_vals.append(float(spo2))
            Xu=scaler.transform([features_vals])
            rf_pred=int(rf.predict(Xu)[0])
            svm_pred=int(svm.predict(Xu)[0])
            rf_prob=float(rf.predict_proba(Xu).max())
            svm_prob=float(svm.predict_proba(Xu).max())
            agreement=rf_pred==svm_pred
            overall='Unknown'
            next_step='Monitor'
            if rf_pred==2 and svm_pred==2:
                overall='High'
                next_step='Escalate'
            elif rf_pred==1 or svm_pred==1:
                overall='Moderate'
                next_step='Repeat measurement and log symptoms'
            else:
                overall='Low'
                next_step='Continue monitoring'
            rec=dict(astronaut_id=astronaut_id,astronaut_name=astronaut_name,age=int(age),sex=sex,pulse=float(pulse),temperature=float(temperature),spo2=float(spo2),timestamp=datetime.utcnow().isoformat(),rf_pred=rf_pred,svm_pred=svm_pred)
            if os.path.exists('data_records.json'):
                with open('data_records.json','r') as f:
                    data=json.load(f)
            else:
                data=[]
            data.append(rec)
            with open('data_records.json','w') as f:
                json.dump(data,f)
            st.success('Prediction complete')
            with col2:
                st.subheader('Model Performance')
                st.write('Random Forest')
                st.json({ 'accuracy':rf_metrics['accuracy'],'precision':rf_metrics['precision'],'recall':rf_metrics['recall'],'f1':rf_metrics['f1'],'cv_score':rf_cv })
                st.write('SVM')
                st.json({ 'accuracy':svm_metrics['accuracy'],'precision':svm_metrics['precision'],'recall':svm_metrics['recall'],'f1':svm_metrics['f1'],'cv_score':svm_cv })
                st.subheader('Rule Based')
                st.write({'overall_risk':overall,'recommended_next_step':next_step})
                st.subheader('ML Outputs')
                st.write({'random_forest_predicted_status':rf_pred,'svm_predicted_status':svm_pred,'model_agreement':agreement,'random_forest_probability':rf_prob,'svm_probability':svm_prob})
                if rf_pred==2 and svm_pred==2:
                    if os.path.exists('static/emergency_voice.mp3'):
                        audio_file=open('static/emergency_voice.mp3','rb').read()
                        st.audio(audio_file,format='audio/mp3')
            with st.expander('Charts and Matrices',expanded=True):
                df_records=pd.DataFrame(data)
                if not df_records.empty:
                    fig,ax=plt.subplots()
                    sns.lineplot(data=df_records,x='timestamp',y='pulse',ax=ax)
                    ax.set_title('Pulse Trend')
                    ax.tick_params(axis='x',labelrotation=45)
                    st.pyplot(fig)
                    fig,ax=plt.subplots()
                    sns.lineplot(data=df_records,x='timestamp',y='temperature',ax=ax,color='r')
                    ax.set_title('Temperature Trend')
                    ax.tick_params(axis='x',labelrotation=45)
                    st.pyplot(fig)
                    fig,ax=plt.subplots()
                    sns.lineplot(data=df_records,x='timestamp',y='spo2',ax=ax,color='g')
                    ax.set_title('SPO2 Trend')
                    ax.tick_params(axis='x',labelrotation=45)
                    st.pyplot(fig)
                fig,ax=plt.subplots()
                categories=['Accuracy','Precision','Recall','F1']
                rf_vals=[rf_metrics['accuracy'],rf_metrics['precision'],rf_metrics['recall'],rf_metrics['f1']]
                svm_vals=[svm_metrics['accuracy'],svm_metrics['precision'],svm_metrics['recall'],svm_metrics['f1']]
                x=np.arange(len(categories))
                width=0.35
                ax.bar(x-width/2,rf_vals,width,label='RandomForest',color='#00f5d4')
                ax.bar(x+width/2,svm_vals,width,label='SVM',color='#ff5c7c')
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
                ax.set_ylim(0,1)
                ax.legend()
                st.pyplot(fig)
                cm_rf=rf_metrics['confusion_matrix']
                fig,ax=plt.subplots()
                sns.heatmap(cm_rf,annot=True,fmt='d',cmap='Blues',ax=ax)
                ax.set_title('RF Confusion Matrix')
                st.pyplot(fig)
                cm_svm=svm_metrics['confusion_matrix']
                fig,ax=plt.subplots()
                sns.heatmap(cm_svm,annot=True,fmt='d',cmap='Reds',ax=ax)
                ax.set_title('SVM Confusion Matrix')
                st.pyplot(fig)
        except Exception as e:
            st.error('Error during prediction')
