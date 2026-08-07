from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and encoders once at startup — not on every request
with open('model/ipl_score_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

le_bat   = encoders['bat_team']
le_bowl  = encoders['bowl_team']
le_venue = encoders['venue']

# These must match exactly what was in your training data
TEAMS = sorted(le_bat.classes_.tolist())
VENUES = sorted(le_venue.classes_.tolist())

FEATURE_COLS = ['bat_team', 'bowl_team', 'venue', 'over',
                'cumulative_runs', 'cumulative_wickets',
                'run_last_5', 'wicket_last_5']

@app.route('/')
def index():
    return render_template('index.html', teams=TEAMS, venues=VENUES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        bat_team        = data['bat_team']
        bowl_team       = data['bowl_team']
        venue           = data['venue']
        over            = int(data['over'])
        current_runs    = int(data['current_runs'])
        current_wickets = int(data['current_wickets'])
        run_last_5      = int(data['run_last_5'])
        wicket_last_5   = int(data['wicket_last_5'])

        bat_enc   = le_bat.transform([bat_team])[0]
        bowl_enc  = le_bowl.transform([bowl_team])[0]
        venue_enc = le_venue.transform([venue])[0]

        input_df = pd.DataFrame(
            [[bat_enc, bowl_enc, venue_enc, over,
              current_runs, current_wickets, run_last_5, wicket_last_5]],
            columns=FEATURE_COLS
        )

        prediction = round(float(model.predict(input_df)[0]))

        return jsonify({'success': True, 'predicted_score': prediction})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)