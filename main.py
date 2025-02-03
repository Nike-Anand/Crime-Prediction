from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import random

app = Flask(__name__)
CORS(app)

class CrimePredictionModel:
    def __init__(self):
        self.crime_types = [
            'Murder', 'Rape', 'Kidnaping & Abduction', 'Robbery', 'Theft', 
            'Riots', 'Cheating ,Forgery& Fraud', 'Counterfeiting', 
            'Grievous Hurt', 'Assault On Women', 'Death By Negligence', 
            'Accidents', 'Cyber Crime', 'Dowry deaths'
        ]
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.df = None
        self.load_and_train()
    
    def load_and_train(self):
        self.df = pd.read_csv('crime detais.csv')
        self.prepare_data()
        self.train()
    
    def prepare_data(self):
        self.df['Total_Crime'] = self.df[self.crime_types].sum(axis=1)
        self.df['encoded_district'] = self.label_encoder.fit_transform(self.df['District'])
        self.df['prev_year_total'] = self.df.groupby('District')['Total_Crime'].shift(1)
        self.df['crime_growth'] = self.df.groupby('District')['Total_Crime'].pct_change()
        self.df = self.df.fillna(0)
    
    def train(self):
        for crime in self.crime_types + ['Total_Crime']:
            X = self.df[['Year', 'encoded_district', 'prev_year_total', 'crime_growth']]
            y = self.df[crime]
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            self.models[crime] = model
    
    def predict(self, district, year):
        if district not in self.df['District'].unique():
            raise ValueError("Invalid district")
        
        district_encoded = self.label_encoder.transform([district])[0]
        
        # Create a DataFrame with the same feature names as the training data
        X = pd.DataFrame([{
            'Year': year,
            'encoded_district': district_encoded,
            'prev_year_total': 0,  # Assuming no prior data
            'crime_growth': 0      # Assuming no prior data
        }])
        
        predictions = {}
        variation_factor = random.uniform(0.95, 1.05)
        
        for crime in self.crime_types + ['Total_Crime']:
            base_prediction = self.models[crime].predict(X)[0]
            predictions[crime] = round(max(0, base_prediction * variation_factor), 2)
        
        return predictions

    
    def get_historical_data(self, district):
        district_data = self.df[self.df['District'] == district]
        return district_data[['Year'] + self.crime_types + ['Total_Crime']].to_dict('records')

# Initialize model
model = CrimePredictionModel()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        district = data.get('district')
        year = int(data.get('year'))
        
        predictions = model.predict(district, year)
        historical_data = model.get_historical_data(district)
        
        return jsonify({
            'predictions': predictions,
            'historical_data': historical_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/districts', methods=['GET'])
def get_districts():
    districts = sorted(model.df['District'].unique().tolist())
    return jsonify({'districts': districts})

if __name__ == '__main__':
    app.run(debug=True)