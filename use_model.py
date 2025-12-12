import torch
import torch.nn as nn
import requests
import random
import numpy as np
from datetime import datetime, timedelta

NASA_API_KEY = "ffpRcNXaeVNObQwp0jdDAlo0Vjs2fhuREG0v8ViD"
NASA_NEO_FEED_URL = "https://api.nasa.gov/neo/rest/v1/feed"

# Old neural network architecture (for backward compatibility)
class AsteroidNetOld(nn.Module):
    def __init__(self, input_size=5):
        super(AsteroidNetOld, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

# Neural network architecture (must match training model)
class AsteroidNet(nn.Module):
    def __init__(self, input_size=5, hidden_dims=[64, 32, 16]):
        super(AsteroidNet, self).__init__()
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Load trained model and normalization parameters (supports old and new models)
def load_model(model_path=r'C:\Users\Yasmin Reyes\Documents\asteroid ai\asteroid_model.pth'):
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Check if this is a new model with config
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        model = AsteroidNet(
            input_size=model_config['input_size'],
            hidden_dims=model_config['hidden_dims']
        )
        print(f"✓ Loading NEW model architecture: {model_config['hidden_dims']}")
    else:
        # Old model format - check the state dict keys
        state_dict_keys = checkpoint['model_state_dict'].keys()
        
        if 'network.0.weight' in state_dict_keys:
            # This is actually a new model but without config saved
            model = AsteroidNet(input_size=5, hidden_dims=[64, 32, 16])
            print(f"✓ Loading NEW model architecture: [64, 32, 16]")
        else:
            # Old model architecture
            model = AsteroidNetOld(input_size=5)
            print(f"✓ Loading OLD model architecture: [32, 16, 8]")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    norm_params = checkpoint['norm_params']
    
    print(f"✓ Model loaded from {model_path}")
    print(f"✓ Normalization parameters loaded")
    return model, norm_params

# Fetch random asteroids from NASA API
def fetch_random_asteroids(days=7, count=5):
    print(f"\nFetching random asteroids from NASA API...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'api_key': NASA_API_KEY
    }
    
    try:
        response = requests.get(NASA_NEO_FEED_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        all_asteroids = []
        for asteroids in data['near_earth_objects'].values():
            all_asteroids.extend(asteroids)
        
        if len(all_asteroids) == 0:
            print("No asteroids found")
            return []
        
        selected = random.sample(all_asteroids, min(count, len(all_asteroids)))
        print(f"✓ Found {len(all_asteroids)} asteroids, selected {len(selected)} for testing")
        
        return selected
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching data: {e}")
        return []

# Parse asteroid data into features (must match training features)
def parse_asteroid_features(asteroid):
    try:
        diameter = asteroid['estimated_diameter']['kilometers']
        approach = asteroid['close_approach_data'][0]
        
        diam_min = diameter['estimated_diameter_min']
        diam_max = diameter['estimated_diameter_max']
        velocity = float(approach['relative_velocity']['kilometers_per_hour'])
        distance = float(approach['miss_distance']['kilometers']) / 1e6
        magnitude = float(asteroid['absolute_magnitude_h'])
        
        # Additional features (must match training)
        diam_avg = (diam_min + diam_max) / 2
        diam_range = diam_max - diam_min
        
        features = [
            diam_min,
            diam_max,
            diam_avg,
            diam_range,
            velocity,
            distance,
            magnitude,
            velocity / distance  # velocity-to-distance ratio
        ]
        
        return features, asteroid
    except (KeyError, IndexError, ValueError, ZeroDivisionError):
        return None, None

# Predict with normalization (uses median normalization like training)
def predict_asteroid(model, features, norm_params):
    mean, std = norm_params
    features_np = np.array([features], dtype=np.float32)
    features_normalized = (features_np - mean) / std
    
    with torch.no_grad():
        tensor = torch.from_numpy(features_normalized)
        prediction = model(tensor)
        hazard_probability = prediction.item()
    
    return hazard_probability

# Display prediction results
def display_prediction(asteroid, features, hazard_prob, threshold=0.4):
    actual_hazard = asteroid['is_potentially_hazardous_asteroid']
    predicted_hazard = hazard_prob > threshold
    match = "✓" if actual_hazard == predicted_hazard else "✗"
    
    # Calculate confidence (distance from decision boundary)
    confidence = abs(hazard_prob - threshold) * 100 / (1 - threshold)
    
    print("\n" + "=" * 70)
    print(f"Asteroid: {asteroid['name']}")
    print("=" * 70)
    
    print(f"Features:")
    print(f"  • Diameter: {features[0]:.4f} - {features[1]:.4f} km (avg: {features[2]:.4f})")
    print(f"  • Velocity: {features[4]:,.0f} km/h")
    print(f"  • Miss Distance: {features[5]:.2f} million km")
    print(f"  • Absolute Magnitude: {features[6]:.2f}")
    print(f"  • Velocity/Distance Ratio: {features[7]:.2f}")
    
    approach = asteroid['close_approach_data'][0]
    print(f"\nClose Approach:")
    print(f"  • Date: {approach['close_approach_date']}")
    print(f"  • Orbiting Body: {approach['orbiting_body']}")
    
    actual_status = "HAZARDOUS" if actual_hazard else "SAFE"
    predicted_status = "HAZARDOUS" if predicted_hazard else "SAFE"
    
    print(f"\nPrediction:")
    print(f"  • Hazard Probability: {hazard_prob * 100:.2f}%")
    print(f"  • Confidence: {confidence:.2f}%")
    print(f"  • Predicted: {predicted_status}")
    print(f"  • Actual: {actual_status}")
    print(f"  • Result: {match} {'CORRECT' if actual_hazard == predicted_hazard else 'INCORRECT'}")

# Test model on multiple random asteroids
def test_random_asteroids(model, norm_params, count=5, threshold=0.4):
    asteroids = fetch_random_asteroids(days=7, count=count)
    
    if not asteroids:
        print("No asteroids available for testing")
        return
    
    print("\n" + "=" * 70)
    print(f"TESTING MODEL ON {len(asteroids)} RANDOM ASTEROIDS")
    print(f"Decision Threshold: {threshold*100:.0f}%")
    print("=" * 70)
    
    correct = 0
    total = 0
    
    for asteroid in asteroids:
        features, parsed_asteroid = parse_asteroid_features(asteroid)
        if features is None:
            continue
        
        hazard_prob = predict_asteroid(model, features, norm_params)
        display_prediction(parsed_asteroid, features, hazard_prob, threshold)
        
        actual = parsed_asteroid['is_potentially_hazardous_asteroid']
        predicted = hazard_prob > threshold
        
        if actual == predicted:
            correct += 1
        total += 1
    
    if total > 0:
        accuracy = (correct / total) * 100
        print("\n" + "=" * 70)
        print(f"SUMMARY")
        print("=" * 70)
        print(f"  Total Tested: {total}")
        print(f"  Correct: {correct}")
        print(f"  Incorrect: {total - correct}")
        print(f"  Accuracy: {accuracy:.2f}%")
        print("=" * 70)

# Predict on custom asteroid features
def predict_custom(model, norm_params, diameter_min, diameter_max, velocity, distance_km, magnitude, threshold=0.4):
    distance = distance_km / 1e6
    diam_avg = (diameter_min + diameter_max) / 2
    diam_range = diameter_max - diameter_min
    
    features = [
        diameter_min, 
        diameter_max, 
        diam_avg,
        diam_range,
        velocity, 
        distance, 
        magnitude,
        velocity / distance if distance > 0 else 0
    ]
    
    hazard_prob = predict_asteroid(model, features, norm_params)
    
    is_hazardous = hazard_prob > threshold
    confidence = abs(hazard_prob - threshold) * 100 / (1 - threshold)
    
    print("\n" + "=" * 70)
    print("CUSTOM ASTEROID PREDICTION")
    print("=" * 70)
    print(f"Input Features:")
    print(f"  • Diameter: {diameter_min:.3f} - {diameter_max:.3f} km (avg: {diam_avg:.3f})")
    print(f"  • Velocity: {velocity:,.0f} km/h")
    print(f"  • Miss Distance: {distance_km:,.0f} km")
    print(f"  • Absolute Magnitude: {magnitude}")
    print(f"  • Velocity/Distance Ratio: {features[7]:.2f}")
    print(f"\nPrediction:")
    print(f"  • Hazard Probability: {hazard_prob * 100:.2f}%")
    print(f"  • Confidence: {confidence:.2f}%")
    print(f"  • Classification: {'⚠️  POTENTIALLY HAZARDOUS' if is_hazardous else '✓ SAFE'}")
    print("=" * 70)
    
    return hazard_prob, is_hazardous

# Main demonstration
def main():
    print("=" * 70)
    print(" " * 20 + "NASA Asteroid Predictor")
    print(" " * 15 + "Testing on Live API Data")
    print("=" * 70)
    
    model, norm_params = load_model()
    
    print("\n[1] Testing on random asteroids from NASA API...")
    test_random_asteroids(model, norm_params, count=50)
    
    print("\n\n[2] Testing custom asteroid parameters...")
    predict_custom(
        model, 
        norm_params,
        diameter_min=1.5,
        diameter_max=3.2,
        velocity=85000,
        distance_km=3000000,
        magnitude=19.5
    )

if __name__ == "__main__":
    main()