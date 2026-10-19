import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn.model_selection import train_test_split


info=pd.read_csv("disease_info.csv")
print(info.head())

data=pd.read_csv("disease.csv")
data=data.drop(columns=["Unnamed: 133"])
print(data.info())
print(data.describe())
print(data.head())

x=data.drop("prognosis",axis=1)
y=data["prognosis"]


with (st.container(border=True)):
   symptom = data.drop("prognosis", axis=1).columns
   m=st.subheader("check your symptoms")
   t=st.text_input("search the symptoms")

   col1, col2, col3 = st.columns(3)
   st.divider()
   p = []

   # Convert search text to lowercase for case-insensitive matching
   search_term = t.lower()

   # column names except prognosis
   for i, sym in enumerate(symptom):
       k = False  # Default state is unchecked/hidden

       if search_term in sym.lower():
           if i % 3 == 0:
               k = col1.checkbox(sym)
           elif i % 3 == 1:
               k = col2.checkbox(sym)
           elif i % 3 == 2:
               k = col3.checkbox(sym)

       p.append(1 if k else 0)




x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

a=RandomForestClassifier(random_state=42)
a.fit(x_train,y_train)
prediction=a.predict([p])
print(prediction[0])

probability=a.predict_proba([p])
print(probability)
g=probability.argmax()

if st.button("Predict", use_container_width=True):
   st.divider()
   with st.container(border=True):
     st.success(f"The predicted disease is {prediction[0]}")
     st.write("Model Accuracy:",a.score(x_test,y_test)*100,"%")
     st.write("Prediction Confidence:",probability[0][g]*100,"%")

     for h in range(len(info)):
         if prediction[0]==info["Disease"][h]:
             st.header("Description:")
             st.write(info["Description"][h])
             st.header("Precautions:")
             st.write(info["Precautions"][h])
             st.header("Prescribed Medicines:")
             st.write(info["Common_Medicines"][h])
             st.header("Specialist:")
             st.write(info["Specialist"][h])

















