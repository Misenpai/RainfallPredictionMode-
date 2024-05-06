from flask import Flask,request,jsonify
import numpy as np
import pickle

modelRainfall = pickle.load(open('rainfall.pkl','rb'))

state_to_int = pickle.load(open("state_to_int.pkl", 'rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return "Hello World"
@app.route('/predict/rainfall',methods=['POST'])



def predictRainfall():
    state = request.form.get("state")
    year = request.form.get("year")
    state_encoded = state_to_int.get(state, None)
    if state_encoded is None:
        return jsonify({'error': 'Invalid state name'}), 400
    
    input_query = np.array([[state_encoded,year]])
    result = modelRainfall.predict(input_query)[0]
    response = {
                'state': state,
        'year': year,
        'predicted_rainfall': {
            'JAN': result[0],
            'FEB': result[1],
            'MAR': result[2],
            'APR': result[3],
            'MAY': result[4],
            'JUN': result[5],
            'JUL': result[6],
            'AUG': result[7],
            'SEP': result[8],
            'OCT': result[9],
            'NOV': result[10],
            'DEC': result[11],
        }
    }
    return jsonify(response)




if __name__ == "__main__":
    app.run(debug=True)