"""
ML-Based Exoplanet Classification Module
Replaces rule-based scoring with trained machine learning models.
Integrates seamlessly with existing pipeline.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class ExoplanetMLClassifier:
    """Machine learning classifier for exoplanet transit signals"""
    
    def __init__(self, model_path='models/exoplanet_model.pkl', scaler_path='models/scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = StandardScaler()
        self.classes = ['Planet Transit', 'Eclipsing Binary', 'Starspot', 'Noise']
        self.feature_names = [
            'SNR', 'Depth', 'Duration_hrs', 'Period_days',
            'Signal_to_Noise', 'Depth_SNR_ratio', 'Duration_Period_ratio'
        ]
        self._ensure_model_dir()
        
    def _ensure_model_dir(self):
        """Create models directory if it doesn't exist"""
        os.makedirs('models', exist_ok=True)
        
    def extract_features(self, snr, depth, duration_hrs, period_days, rms=1.0):
        """Extract ML features from transit parameters"""
        # Handle edge cases safely
        snr = float(max(snr, 0.1))
        depth = float(max(depth, 0.00001))
        duration_hrs = float(max(duration_hrs, 0.1))
        period_days = float(max(period_days, 0.1))
        rms = float(max(rms, 0.0001))
        
        features = {
            'SNR': snr,
            'Depth': depth,
            'Duration_hrs': duration_hrs,
            'Period_days': period_days,
            'Signal_to_Noise': snr / rms,
            'Depth_SNR_ratio': depth / snr,
            'Duration_Period_ratio': duration_hrs / (period_days * 24)
        }
        return np.array([features[f] for f in self.feature_names]).reshape(1, -1)
    
    def predict(self, snr, depth, duration_hrs, period_days, rms=1.0):
        """Predict signal class and get confidence"""
        # Try ML first, fallback to rules
        if self.model is not None:
            try:
                X = self.extract_features(snr, depth, duration_hrs, period_days, rms)
                X_scaled = self.scaler.transform(X)
                
                # Get predictions
                prediction = self.model.predict(X_scaled)[0]
                probabilities = self.model.predict_proba(X_scaled)[0]
                
                # Find confidence and label
                confidence = float(max(probabilities) * 100)
                signal_type = self.classes[int(prediction)]
                
                return {
                    'signal_type': signal_type,
                    'confidence': round(confidence, 1),
                    'conf_lbl': 'HIGH' if confidence >= 80 else 'MEDIUM' if confidence >= 60 else 'LOW',
                    'probabilities': {self.classes[i]: float(p) for i, p in enumerate(probabilities)},
                    'reasons': self._generate_reasons(snr, depth, duration_hrs, period_days, probabilities),
                    'model_used': 'ML'
                }
            except Exception as e:
                print(f"ML prediction error: {e}. Falling back to rule-based.")
        
        return self._fallback_classification(snr, depth, duration_hrs, period_days)
    
    def _fallback_classification(self, snr, depth, duration_hrs, period_days):
        """Fallback rule-based classification"""
        score = {
            'Planet Transit': 0,
            'Eclipsing Binary': 0,
            'Starspot': 0,
            'Noise': 0
        }
        reasons = []
        
        # SNR rules
        if snr < 3:
            score['Noise'] += 60
            reasons.append(f'SNR {snr:.2f} < 3 → weak signal')
        elif snr < 7:
            score['Noise'] += 20
            score['Planet Transit'] += 10
            reasons.append(f'SNR {snr:.2f} moderate (3–7)')
        else:
            score['Planet Transit'] += 40
            reasons.append(f'SNR {snr:.2f} strong (>7)')
        
        # Depth rules
        if depth > 0.05:
            score['Eclipsing Binary'] += 60
            reasons.append(f'Depth {depth:.6f} very deep (>5%)')
        elif depth >= 0.001:
            score['Planet Transit'] += 40
            reasons.append(f'Depth {depth:.6f} planet-like (0.1–5%)')
        else:
            score['Noise'] += 30
            reasons.append(f'Depth {depth:.6f} too shallow (<0.1%)')
        
        # Duration rules
        if duration_hrs > 12:
            score['Starspot'] += 50
            reasons.append(f'Duration {duration_hrs:.2f}h very long → starspot')
        elif duration_hrs >= 1:
            score['Planet Transit'] += 30
            reasons.append(f'Duration {duration_hrs:.2f}h normal')
        else:
            score['Noise'] += 20
            reasons.append(f'Duration {duration_hrs:.2f}h too short')
        
        # Period rules
        if period_days < 1:
            score['Eclipsing Binary'] += 40
            reasons.append(f'Period {period_days:.4f}d very short → binary')
        elif period_days <= 15:
            score['Planet Transit'] += 30
            reasons.append(f'Period {period_days:.4f}d normal')
        else:
            score['Starspot'] += 20
            reasons.append(f'Period {period_days:.4f}d long → starspot')
        
        sig = max(score, key=score.get)
        tot = sum(score.values())
        conf = round(score[sig] / tot * 100, 1) if tot > 0 else 0
        
        return {
            'signal_type': sig,
            'confidence': conf,
            'conf_lbl': 'HIGH' if conf >= 80 else 'MEDIUM' if conf >= 60 else 'LOW',
            'probabilities': {k: v/tot if tot > 0 else 0 for k, v in score.items()},
            'reasons': reasons,
            'model_used': 'Rule-Based'
        }
    
    def _generate_reasons(self, snr, depth, duration_hrs, period_days, probabilities):
        """Generate human-readable ML-based classification reasons"""
        reasons = []
        winner_idx = np.argmax(probabilities)
        winner_prob = probabilities[winner_idx]
        
        # Confidence level
        reasons.append(f'🤖 ML Model Confidence: {winner_prob*100:.1f}%')
        
        # SNR insight
        if snr > 7:
            reasons.append(f'✓ Strong SNR ({snr:.2f}) → high signal confidence')
        elif snr > 3:
            reasons.append(f'⚠ Moderate SNR ({snr:.2f}) → marginal detection')
        else:
            reasons.append(f'✗ Weak SNR ({snr:.2f}) → likely noise')
        
        # Depth insight
        if 0.001 <= depth <= 0.05:
            reasons.append(f'✓ Reasonable depth ({depth:.6f}) → transiting object')
        elif depth > 0.05:
            reasons.append(f'⚠ Very deep ({depth:.6f}) → may be binary/eclipsing')
        else:
            reasons.append(f'✗ Too shallow ({depth:.6f}) → instrumental noise')
        
        # Duration insight
        if 1 <= duration_hrs <= 8:
            reasons.append(f'✓ Physical duration ({duration_hrs:.2f}h) → planet-like')
        elif duration_hrs > 12:
            reasons.append(f'⚠ Long duration ({duration_hrs:.2f}h) → stellar activity')
        
        # Period insight
        if 0.5 <= period_days <= 20:
            reasons.append(f'✓ Observable period ({period_days:.4f}d) → potential exoplanet')
        
        return reasons
    
    def train_from_data(self, X_train, y_train, test_size=0.2):
        """Train model from labeled data"""
        print("🔧 Training ML classifier...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y_train, test_size=test_size, random_state=42, stratify=y_train
        )
        
        self.model.fit(X_tr, y_tr)
        
        # Evaluate
        train_score = self.model.score(X_tr, y_tr)
        test_score = self.model.score(X_te, y_te)
        y_pred = self.model.predict(X_te)
        
        print(f"✓ Training accuracy: {train_score:.4f}")
        print(f"✓ Test accuracy: {test_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_te, y_pred, target_names=self.classes))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_te, y_pred))
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        print("\nFeature Importance:")
        for feat, imp in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat}: {imp:.4f}")
        
        return test_score
    
    def save_model(self):
        """Save trained model and scaler"""
        if self.model:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✓ Model saved to {self.model_path}")
            return True
        return False
    
    def load_model(self):
        """Load pre-trained model and scaler"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✓ Model loaded from {self.model_path}")
                return True
            except Exception as e:
                print(f"✗ Failed to load model: {e}")
                return False
        return False


def create_synthetic_training_data(n_samples=200):
    """Generate realistic synthetic training data for ML model"""
    np.random.seed(42)
    
    data = []
    
    # Class 0: Planet Transit (50 samples)
    # Real exoplanets: SNR 5-15, depth 0.1-5%, duration 1-8h, period 1-15d
    for _ in range(50):
        snr = np.random.normal(8, 2.5)  # Mean 8, std 2.5
        depth = np.random.uniform(0.001, 0.05)  # 0.1-5%
        duration = np.random.uniform(1, 8)  # 1-8 hours
        period = np.random.uniform(1, 15)  # 1-15 days
        ratio = depth / snr if snr > 0 else 0
        dur_period = duration / (period * 24)
        data.append([snr, depth, duration, period, snr, ratio, dur_period, 0])
    
    # Class 1: Eclipsing Binary (50 samples)
    # Binaries: SNR 2-10, depth >5%, duration 2-12h, period <1d
    for _ in range(50):
        snr = np.random.uniform(2, 10)
        depth = np.random.uniform(0.05, 0.3)  # >5%
        duration = np.random.uniform(2, 12)  # 2-12 hours
        period = np.random.uniform(0.1, 2)  # <1 day
        ratio = depth / snr if snr > 0 else 0
        dur_period = duration / (period * 24)
        data.append([snr, depth, duration, period, snr, ratio, dur_period, 1])
    
    # Class 2: Starspot (50 samples)
    # Stellar activity: SNR 1-5, depth <1%, duration >12h, period >15d
    for _ in range(50):
        snr = np.random.uniform(1, 5)
        depth = np.random.uniform(0.0001, 0.01)  # <1%
        duration = np.random.uniform(12, 100)  # >12 hours
        period = np.random.uniform(15, 50)  # >15 days
        ratio = depth / snr if snr > 0 else 0
        dur_period = duration / (period * 24)
        data.append([snr, depth, duration, period, snr, ratio, dur_period, 2])
    
    # Class 3: Noise (50 samples)
    # Random noise: SNR <2, depth <0.1%, duration <0.5h or random
    for _ in range(50):
        snr = np.random.uniform(0.1, 2)
        depth = np.random.uniform(0.00001, 0.001)  # <0.1%
        duration = np.random.uniform(0.1, 0.5)  # <0.5h
        period = np.random.uniform(0.5, 50)  # Random
        ratio = depth / snr if snr > 0 else 0
        dur_period = duration / (period * 24)
        data.append([snr, depth, duration, period, snr, ratio, dur_period, 3])
    
    return np.array(data)


def initialize_ml_model(force_retrain=False):
    """Initialize ML model: load existing or create/train new one"""
    classifier = ExoplanetMLClassifier()
    
    # Try to load existing model
    if classifier.load_model() and not force_retrain:
        return classifier
    
    # Create and train new model
    print("📊 Creating synthetic training data...")
    data = create_synthetic_training_data(n_samples=200)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    
    print("🎯 Training ML model with synthetic data...")
    classifier.train_from_data(X, y, test_size=0.2)
    classifier.save_model()
    
    return classifier
