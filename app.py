from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import pickle
app=Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/Prediction',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        Credit_Mix = request.form['Credit_Mix']
        Annual_Income=float(request.form['Annual_Income'])
        Outstanding_Debt=float(request.form['Outstanding_Debt'])
        Total_EMI_per_month=float(request.form['Total_EMI_per_month'])
        Interest_Rate=float(request.form['Interest_Rate'])
        Credit_History_Age=float(request.form['Credit_History_Age'])
        
        #Load the model from the file
        model= pickle.load(open('model.pkl','rb'))

         # Load the scaler from the file
        scaler = pickle.load(open('scaler.pkl','rb'))

         # Load the encoder from the file
        encoder = pickle.load(open('encoder.pkl','rb'))
        
        dt = {'Annual_Income':Annual_Income, 'Outstanding_Debt':Outstanding_Debt, 'Total_EMI_per_month':Total_EMI_per_month,
       'Interest_Rate':Interest_Rate, 'Credit_History_Age':Credit_History_Age, 'Credit_Mix':Credit_Mix}
        
        new_data = pd.DataFrame(dt,index=[0])

        # Preprocess the new data (example using MinMaxScaler and OrdinalEncoder)
        new_data[['Annual_Income','Outstanding_Debt','Total_EMI_per_month','Credit_History_Age']] = scaler.transform(new_data[['Annual_Income','Outstanding_Debt','Total_EMI_per_month','Credit_History_Age']])
        new_data['Credit_Mix_Enco'] = encoder.transform(new_data[['Credit_Mix']])

        features = model.feature_names_in_


        result=model.predict(new_data.drop('Credit_Mix',axis=1)[features])[0]
        print(result)
        

    return render_template('prediction.html',result=result)

if __name__=='__main__':
    app.run()