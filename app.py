import streamlit as st
import pickle
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# import matplotlib.pyplot as plt
import plotly.express as px
#%matplotlib inline

df=pd.read_csv("home_price_loc.csv")

st.title("Home price Prediction")
st.dataframe(df)

fig_branch=px.bar(
    df,
    x="location",
    y="price",
    text="location",
    title="location wise price",
    color="location"
)

st.plotly_chart(fig_branch)

le=LabelEncoder()
df["location"] = le.fit_transform(df["location"])
#dependent variable is price
#independent variable is area and location
x=df.drop('price',axis='columns')
y=df.price

st.subheader("Dependent Variables that's in 2 Dimention")
st.dataframe(x)

st.subheader("Independent Variables that's in Series")
st.dataframe(y)



st.sidebar.header("Dataset Information")
st.sidebar.subheader("No# area we have")
st.sidebar.info(df["area"].count())
st.sidebar.subheader("Last Predicted Price")
st.sidebar.info(df['price'].iloc[len(df['price'])-1])


model=linear_model.LinearRegression()
model.fit(x,y)

area=int(st.number_input("Enter area",min_value=500))

loc=st.selectbox("Enter location",le.classes_)

if st.button("Predict"):
    

    if loc:
        are=le.transform([loc])[0]
        predicted_price=model.predict([[area,are]])
        final_price=int(predicted_price[0])
        st.success(f"Your Predicted Price for {loc} Rs.{final_price}/-")
     




    existing_data=pd.read_csv("home_price_loc.csv")
    new_data=pd.DataFrame({"area":[area],"location":[loc],"price":[final_price]})
    updated_data=pd.concat([existing_data,new_data])
    updated_data.to_csv("home_price_loc.csv",index=False)


# st.sidebar.subheader("Last Predicted Price")
# st.sidebar.info(final_price)