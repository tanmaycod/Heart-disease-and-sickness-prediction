# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:45:08 2022

@author: tanmay
"""

import numpy as np
import pandas as pd
import base64
import pickle
import streamlit as st
from streamlit_option_menu import option_menu


l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)
    
    
df=pd.read_csv("D:/disease prediction/Heart disease and sickness prediction/saved models/Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)

tr=pd.read_csv("D:/disease prediction/Heart disease and sickness prediction/saved models/Testing.csv")

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)







def randomforest(s1,s2,s3,s4):
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    st.write(accuracy_score(y_test, y_pred))
    # -----------------------------------------------------

    psymptoms = [s1,s2,s3,s4]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    
    if (h=='yes'):
        
        writee('result = '+disease[a])
       
    else:
        writee('not found')
        
        



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
      




OPTIONS = ["Select"]+sorted(l1)


















heart_model = pickle.load(open('D:/disease prediction/Heart disease and sickness prediction/saved models/modelheart.sav','rb'))


def writee(url):
     st.markdown(f'<p style="background-color:#0E1117;color:#FAFAFA;font-size:20px;border-radius:2%; padding: 10px;">{url}</p>', unsafe_allow_html=True)


st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        background-color: #224562;
        color: #224562;
        background-image: url("https://c4.wallpaperflare.com/wallpaper/176/79/767/colorful-minimalism-graphic-design-gradient-wallpaper-preview.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        border:none;
        
    }
    [data-testid="Write"][aria-expanded="true"] > div:first-child {
        text-color: #224562;
        
        
    }
   
    </style>
    """,
    unsafe_allow_html=True,
    )


#sidebar

with st.sidebar:
    

    
    selected = option_menu('Heart Disease and Sickness Prediction',
                           
                           ['Heart Disease Prediction',
                            'Sickness Prediction'],
                           
                           icons = ['activity','person'],
                           
                           
                           default_index = 0)
    


# Heart prediction
if(selected == 'Heart Disease Prediction'):
    
    add_bg_from_local('D:\disease prediction\Heart disease and sickness prediction\g4.jpg')
    
    #title
    st.title('Heart Disease Prediction')    
    
    
    #getting input
    col1,col2,col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age')
    
    with col2:
        sex = st.number_input('Sex')
        
    with col3:
        cp = st.number_input('Chest Pain Type')
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic Results')
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.number_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    
    
    
    #code for prediction
    heart_dia = ''

    #button
    if st.button('Heart Disease Test Result'):
        heart_pred = heart_model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    
        if(heart_pred[0] == 1):
            heart_dia = 'the pearson is having a heart disease'
            
        else:
            heart_dia = 'the person doest not have any heart disease'
        
    st.success(heart_dia)



# sickness prediction
if(selected == 'Sickness Prediction'):
    
    #title
    st.title('Patient Sickness Prediction')    
    
    add_bg_from_local('D:\disease prediction\Heart disease and sickness prediction\g1.jpg')

    
    #input
    Symptom1 = st.selectbox('symptom1',OPTIONS)
    Symptom2 = st.selectbox('symptom2',OPTIONS)
    Symptom3 = st.selectbox('symptom3',OPTIONS)
    Symptom4 = st.selectbox('symptom4',OPTIONS)
    Symptom5 = st.selectbox('symptom5',OPTIONS)

    
    #button
    if st.button('Predict'):
        if(Symptom1 == 'Select' and Symptom2 == 'Select' and Symptom3 == 'Select' and Symptom4 == 'Select' and Symptom5 == 'Select'):
            writee('please select atleast two symptoms')
        
        else:
            if(Symptom1 != 'Select' and Symptom2 == 'Select' and Symptom3 == 'Select' and Symptom4 == 'Select' and Symptom5 == 'Select'):
                writee('please select one more Symptom')
            else:
                randomforest(Symptom1, Symptom2, Symptom3, Symptom4,Symptom5)
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    