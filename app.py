from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('airline.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    result=''
    return render_template('home.html',**locals())

@app.route('/predict',methods=['POST','GET'])
def predict():
        data1 = request.form['ID']
        data2 = request.form['Gender']
        data3 = request.form['Age']
        data4 = request.form['Customer-Type']
        data5 = request.form['Type-of-travel']
        data6 = request.form['Class']
        data7 = request.form['Flight-Distance']
        data8 = request.form['Departure-Delay']
        data9 = request.form['Arrival-Delay']
        data10 = request.form['Departure-and-Arrival-Time-Convenience']
        data11 = request.form['Ease-of-Online-Booking']
        data12 = request.form['Check-in-Service']
        data13 = request.form['Online-Boarding']
        data14 = request.form['Gate-Location']
        data15 = request.form['On-board-Service']
        data16 = request.form['Seat-Comfort']
        data17 = request.form['Leg-Room-Service']
        data18 = request.form['Cleanliness']
        data19 = request.form['Food-and-Drink']
        data20 = request.form['In-flight-Service']
        data21 = request.form['In-flight-Wifi-Service']
        data22 = request.form['In-flight-Entertainment']
        data23 = request.form['Baggage-Handling']
        arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16, data17, data18, data19, data20, data21, data22, data23]])
        # Make predictions
        pred = model.predict(arr)

        # Map prediction to labels
        if pred[0] == 0:
            prediction_text = 'The Customer is Dissatisfied'
        else:
            prediction_text = 'The Customer is Satisfied'

        # Render template with the prediction text
        # return render_template('home.html', prediction_text=prediction_text)
        return render_template('after.html', prediction_text=prediction_text)
      


if __name__ == "__main__":
    app.run(debug=True)
