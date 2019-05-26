
from flask import Flask, request, render_template, Response, jsonify
from sklearn.externals import joblib

app = Flask(__name__, template_folder='./templates')

model = joblib.load('./models/CarPriceCategoryPredictor.pkl')

def class_to_range(price_class):
    if price_class == 1:
        return {'max': 2000}
    elif price_class == 2:
        return {'min': 2000, 'max': 3000}
    elif price_class == 3:
        return {'min': 3000, 'max': 4000}
    elif price_class == 4:
        return {'min': 4000, 'max': 5000}
    elif price_class == 5:
        return {'min': 5000, 'max': 10000}
    elif price_class == 6:
        return {'min': 10000, 'max': 15000}
    elif price_class == 7:
        return {'min': 15000, 'max': 20000}
    elif price_class == 8:
        return {'min': 20000, 'max': 25000}
    elif price_class == 9:
        return {'min': 25000, 'max': 30000}
    elif price_class == 10:
        return {'min': 30000, 'max': 40000}
    elif price_class == 11:
        return {'min': 40000, 'max': 50000}
    else:
        return {'min': 50000}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    data = dict(request.form)

    X = [[
        int(data['maker']), 
        int(data['mileage']),
        int(data['year']),
        int(data['engine_displacement']),
        int(data['engine_power']),
        int(data['transmission']),
        int(data['door_count']),
        int(data['seat_count']),
        int(data['fuel_type'])
        ]]

    predicted = model.predict(X)
    result = class_to_range(predicted[0])
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
