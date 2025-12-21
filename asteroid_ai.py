import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import requests
from datetime import datetime, timedelta
import numpy as np
import time
from collections import Counter

NASA_API_KEY = "DEMO_KEY"
NASA_NEO_FEED_URL = "https://api.nasa.gov/neo/rest/v1/feed"
NASA_NEO_LOOKUP_URL = "https://api.nasa.gov/neo/rest/v1/neo/"

# Enhanced neural network with class weighting support
class AsteroidNet(nn.Module):
    def __init__(self, input_size=5, hidden_dims=[128, 64, 32]):
        super(AsteroidNet, self).__init__()
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.25)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Fetch data with improved error handling
def fetch_asteroid_feed_chunked(days=30, max_retries=5):
    print(f"Fetching {days} days of asteroid data from NASA...")
    
    all_asteroids = []
    end_date = datetime.now()
    
    chunks = [(end_date - timedelta(days=i+7), end_date - timedelta(days=i)) 
              for i in range(0, days, 7)]
    
    def fetch_chunk(start, end):
        params = {
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'api_key': NASA_API_KEY
        }
        
        for attempt in range(max_retries):
            try:
                print(f"  Fetching {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}...")
                response = requests.get(NASA_NEO_FEED_URL, params=params, timeout=30)
                response.raise_for_status()
                print(f"  ✓ Success: {start.strftime('%Y-%m-%d')}")
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  ⚠ Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ✗ Failed chunk {start.strftime('%Y-%m-%d')}: {e}")
                    return None
        return None
    
    for start, end in chunks:
        data = fetch_chunk(start, end)
        if data and 'near_earth_objects' in data:
            for asteroids in data['near_earth_objects'].values():
                all_asteroids.extend(asteroids)
        time.sleep(1)
    
    print(f"✓ Fetched {len(all_asteroids)} asteroids from {days} days")
    return all_asteroids

# Enhanced data processing with feature engineering
def process_asteroid_data(asteroids):
    if not asteroids:
        print("No data available")
        return None
    
    print(f"\nProcessing {len(asteroids)} asteroids...")
    
    features = []
    labels = []
    
    for asteroid in asteroids:
        try:
            diameter = asteroid['estimated_diameter']['kilometers']
            approach = asteroid['close_approach_data'][0]
            
            diam_min = diameter['estimated_diameter_min']
            diam_max = diameter['estimated_diameter_max']
            velocity = float(approach['relative_velocity']['kilometers_per_hour'])
            distance = float(approach['miss_distance']['kilometers']) / 1e6
            magnitude = float(asteroid['absolute_magnitude_h'])
            
            # Additional features for better prediction
            diam_avg = (diam_min + diam_max) / 2
            diam_range = diam_max - diam_min
            
            features.append([
                diam_min,
                diam_max,
                diam_avg,
                diam_range,
                velocity,
                distance,
                magnitude,
                velocity / distance  # velocity-to-distance ratio
            ])
            
            labels.append(1.0 if asteroid['is_potentially_hazardous_asteroid'] else 0.0)
        except (KeyError, IndexError, ValueError, TypeError):
            continue
    
    if not features:
        return None
    
    features_np = np.array(features, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.float32)
    
    # Robust normalization
    mean = np.median(features_np, axis=0)
    std = np.std(features_np, axis=0) + 1e-8
    features_normalized = (features_np - mean) / std
    
    X = torch.from_numpy(features_normalized)
    y = torch.from_numpy(labels_np).unsqueeze(1)
    
    hazardous = int(y.sum().item())
    safe = len(features) - hazardous
    
    print(f"\n✓ Processed {len(features)} valid asteroids")
    print(f"  • Hazardous: {hazardous} ({hazardous/len(features)*100:.1f}%)")
    print(f"  • Safe: {safe} ({safe/len(features)*100:.1f}%)")
    print(f"  • Class Imbalance Ratio: 1:{safe/hazardous if hazardous > 0 else 0:.1f}")
    
    return X, y, (mean, std)

# Training with class weighting to handle imbalance
def train_model(X, y, epochs=400, batch_size=32, lr=0.0005, val_split=0.2):
    dataset_size = len(X)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    dataset = TensorDataset(X, y)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate class weights - moderate weighting for balance
    pos_count = y.sum().item()
    neg_count = len(y) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count * 1.5]) if pos_count > 0 else torch.tensor([1.0])  # 1.5x multiplier
    
    print(f"\n{'=' * 60}")
    print("\nTraining Configuration:")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Positive class weight: {pos_weight.item():.2f}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    
    model = AsteroidNet(input_size=X.shape[1], hidden_dims=[128, 64, 32])
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    
    best_val_loss = float('inf')
    best_f1 = 0.0
    patience_counter = 0
    early_stop_patience = 60
    
    print("\nStarting training...\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            
            # Apply moderate class weighting
            loss = criterion(outputs, batch_y)
            weights = torch.where(batch_y == 1, pos_weight.item(), 1.0)
            loss = (loss * weights).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y).mean()
                val_loss += loss.item()
                
                val_preds.extend((outputs > 0.4).float().cpu().numpy())  # Balanced threshold
                val_labels.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Calculate validation metrics
        val_preds = np.array(val_preds).flatten()
        val_labels = np.array(val_labels).flatten()
        
        tp = np.sum((val_preds == 1) & (val_labels == 1))
        fp = np.sum((val_preds == 1) & (val_labels == 0))
        fn = np.sum((val_preds == 0) & (val_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        scheduler.step(val_loss)
        
        # Track best model based on F1 score for balance
        if f1 > best_f1:
            best_f1 = f1
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            print(f"  Best F1 Score: {best_f1*100:.2f}%")
            break
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'  Val Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1*100:.2f}%')
            print(f'  TP: {tp:.0f}, FP: {fp:.0f}, FN: {fn:.0f}')
            print()
    
    print("✓ Training complete!")
    return model

# Comprehensive evaluation with detailed metrics
def evaluate_model(model, X, y, threshold=0.3):  # Lower threshold to catch more hazards
    model.eval()
    
    with torch.no_grad():
        predictions = model(X)
        predicted_probs = predictions.numpy().flatten()
        predicted_classes = (predictions > threshold).float()
        y_true = y.numpy().flatten()
        y_pred = predicted_classes.numpy().flatten()
        
        correct = (y_pred == y_true).sum()
        accuracy = correct / len(y_true)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Additional metrics
        total_hazardous = int(y_true.sum())
        total_safe = len(y_true) - total_hazardous
        detected_hazardous = int(tp)
        missed_hazardous = int(fn)
    
    print(f"\nMODEL PERFORMANCE METRICS (Threshold: {threshold})")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {tp:>5.0f}  (Correctly identified hazardous)")
    print(f"  False Positives: {fp:>5.0f}  (Safe marked as hazardous)")
    print(f"  True Negatives:  {tn:>5.0f}  (Correctly identified safe)")
    print(f"  False Negatives: {fn:>5.0f}  (Hazardous marked as safe)")
    
    print(f"Classification Metrics:")
    print(f"  Overall Accuracy:  {accuracy * 100:>6.2f}%")
    print(f"  Precision:         {precision * 100:>6.2f}%  (How many predicted hazards are real)")
    print(f"  Recall:            {recall * 100:>6.2f}%  (How many real hazards we catch)")
    print(f"  F1 Score:          {f1 * 100:>6.2f}%  (Harmonic mean of P&R)")
    print(f"  Specificity:       {specificity * 100:>6.2f}%  (True negative rate)")
    
    print(f"Class-Specific Performance:")
    print(f"  Hazardous Asteroids: {total_hazardous}")
    print(f"    ✓ Detected:  {detected_hazardous} ({detected_hazardous/total_hazardous*100 if total_hazardous > 0 else 0:.1f}%)")
    print(f"    ✗ Missed:    {missed_hazardous} ({missed_hazardous/total_hazardous*100 if total_hazardous > 0 else 0:.1f}%)")
    print(f"  Safe Asteroids: {total_safe}")
    print(f"    ✓ Correct:   {tn} ({tn/total_safe*100 if total_safe > 0 else 0:.1f}%)")
    print(f"    ✗ Incorrect: {fp} ({fp/total_safe*100 if total_safe > 0 else 0:.1f}%)")
    
    return accuracy, precision, recall, f1

# Main execution
def main():
    print(" " * 15 + "NASA Asteroid AI Predictor")
    print(" " * 10 + "Powered by NASA NeoWs API & PyTorch\n")
    
    asteroids = fetch_asteroid_feed_chunked(days=30)
    if not asteroids:
        print("Failed to fetch data. Check your internet connection.")
        return
    
    result = process_asteroid_data(asteroids)
    if result is None:
        print("No asteroid data to process")
        return
    
    X, y, norm_params = result
    
    model = train_model(X, y, epochs=400, batch_size=32)
    evaluate_model(model, X, y)
    
    print("\nSample Predictions (First 5 Asteroids):")
    
    model.eval()
    with torch.no_grad():
        predictions = model(X[:5])
        for i in range(min(5, len(X))):
            hazard_prob = predictions[i].item()
            confidence = abs(hazard_prob - 0.3) * 200 / 0.7  # Adjusted for new threshold
            actual = "HAZARDOUS" if y[i].item() == 1.0 else "SAFE"
            predicted = "HAZARDOUS" if hazard_prob > 0.3 else "SAFE"  # Lower threshold
            match = "✓" if actual == predicted else "✗"
            
            print(f"\nAsteroid {i+1}:")
            print(f"  Predicted: {predicted} (prob: {hazard_prob*100:.1f}%, conf: {confidence:.1f}%) {match}")
            print(f"  Actual:    {actual}")
    
    save_path = r'C:\Users\Documents\GitHub\asteroid_classifier\asteroid_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_params': norm_params,
        'model_config': {'input_size': X.shape[1], 'hidden_dims': [128, 64, 32]},
        'threshold': 0.3  # Save optimal threshold
    }, save_path)
    
    print(f"\n✓ Model saved to '{save_path}'")

if __name__ == "__main__":

    main()
