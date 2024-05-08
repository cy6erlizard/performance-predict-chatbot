from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model and encoder
model = joblib.load('football_prediction_model.pkl')
encoder = joblib.load('team_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    home_team = data.get('home_team')
    away_team = data.get('away_team')

    # Encode team names
    try:
        home_encoded = encoder.transform([home_team])[0]
        away_encoded = encoder.transform([away_team])[0]
    except ValueError:
        return jsonify({'error': 'One of the team names is not recognized. Please enter valid team names.'}), 400

    # Use neutral values for other features
    features = [[home_encoded, away_encoded, 0, 0, 0, 0, 50, 25, 0, 0, 0]]

    # Get probability predictions
    probability = model.predict_proba(features)[0]

    # Return predicted probabilities
    return jsonify({
        'home_team': home_team,
        'away_team': away_team,
        'predicted_prob_home_win': probability[2],
        'predicted_prob_away_win': probability[1],
        'predicted_prob_draw': probability[0]
    })

if __name__ == '__main__':
    app.run(debug=True)
