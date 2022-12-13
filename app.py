import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

st.title("Iris Flower Classification")
st.text("")
df=pd.read_csv('IRIS.csv')
flower = st.sidebar.multiselect("Filter by Flower", options=df['species'].unique(),default=df['species'][0])
df_filtered = df.query(
        "species == @flower"
    )
st.dataframe(df_filtered.sample(5))
st.text("")
st.text("")
df_chart = df_filtered.sample(50)
st.bar_chart(data=df_chart.iloc[:,:4],use_container_width=True)
x=df.iloc[:,:4]
y=df.iloc[:,4]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=110)
model=LogisticRegression()
model.fit(x_train,y_train)

file = open('model.pkl', 'wb')
pickle.dump(file)
# file = open('model.pkl', 'rb')
# model = pickle.load(file)
s_length = st.slider('Sepal Length', 0, 10, 2, step=1)
s_width = st.slider('Sepal Width', 0, 10, 2, step=1)
p_length = st.slider('Petal Length', 0, 10, 2, step=1)
p_width = st.slider('Petal Width', 0, 10, 2, step=1)
button = st.button('Predict')
if button:
    y_pred=model.predict([[s_length,s_width,p_length,p_width]])
    if y_pred[0] == 'Iris-setosa':
        st.subheader(y_pred[0])
        st.image('images/setosa.jpg',width=300)
    if y_pred[0] == 'Iris-virginica':
        st.subheader(y_pred[0])
        st.image('images/virginica.jpg',width=300)
    if y_pred[0] == 'Iris-versicolor':
        st.subheader(y_pred[0])
        st.image('images/versicolor.jpg',width=300)



