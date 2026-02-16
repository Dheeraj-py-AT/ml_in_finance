"""
Machine Learning Methods Module
--------------------------------
Implements ML approaches for the same banking problems
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import time


class MLMethods:
    """Machine Learning methods for banking"""
    
    def __init__(self, random_state=42):
        """Initialize ML methods"""
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
    
    def fraud_detection_ml(self, df):
        """
        ML-based fraud detection using Random Forest
        
        Uses all available features to learn patterns
        
        Returns:
            predictions, metrics, model
        """
        print("\n🤖 Running ML Fraud Detection (Random Forest)...")
        start_time = time.time()
        
        # Prepare data
        X = df[['amount', 'hour', 'distance_from_home', 
                'merchant_risk_score', 'transactions_last_24h']]
        y = df['is_fraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        predictions = model.predict(X_test_scaled)
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions),
            'execution_time': execution_time,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"✓ Completed in {execution_time:.4f} seconds")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Training samples: {metrics['train_samples']}")
        
        # Store for later use
        self.models['fraud'] = model
        self.scalers['fraud'] = scaler
        
        return predictions, metrics, model
    
    def credit_scoring_ml(self, df):
        """
        ML-based credit scoring using Random Forest
        
        Learns complex patterns from credit data
        
        Returns:
            predictions, metrics, model
        """
        print("\n🤖 Running ML Credit Scoring (Random Forest)...")
        start_time = time.time()
        
        # Prepare data
        X = df[['age', 'income', 'credit_history_months', 
                'existing_debt', 'loan_amount', 
                'employment_length_years', 'debt_to_income']]
        y = df['default']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        predictions = model.predict(X_test_scaled)
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions),
            'execution_time': execution_time,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"✓ Completed in {execution_time:.4f} seconds")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Training samples: {metrics['train_samples']}")
        
        # Store for later use
        self.models['credit'] = model
        self.scalers['credit'] = scaler
        
        return predictions, metrics, model
    
    def trading_strategy_ml(self, df):
        """
        ML-based trading using Random Forest
        
        Learns patterns from technical indicators and price history
        
        Returns:
            predictions, metrics, model
        """
        print("\n🤖 Running ML Trading Strategy (Random Forest)...")
        start_time = time.time()
        
        # Prepare data
        X = df[['price', 'volume', 'ma_5', 'ma_20', 
                'volatility', 'volume_ma']]
        y = df['price_up']
        
        # Split data (time-series: use first 80% for train)
        split_idx = int(len(df) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        predictions = model.predict(X_test_scaled)
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions),
            'execution_time': execution_time,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print(f"✓ Completed in {execution_time:.4f} seconds")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Training samples: {metrics['train_samples']}")
        
        # Store for later use
        self.models['trading'] = model
        self.scalers['trading'] = scaler
        
        return predictions, metrics, model
    
    def get_feature_importance(self, model_name):
        """
        Get feature importance from trained model
        
        Args:
            model_name: 'fraud', 'credit', or 'trading'
            
        Returns:
            Dictionary of features and their importance scores
        """
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        # Feature names
        if model_name == 'fraud':
            features = ['amount', 'hour', 'distance_from_home', 
                       'merchant_risk_score', 'transactions_last_24h']
        elif model_name == 'credit':
            features = ['age', 'income', 'credit_history_months', 
                       'existing_debt', 'loan_amount', 
                       'employment_length_years', 'debt_to_income']
        else:  # trading
            features = ['price', 'volume', 'ma_5', 'ma_20', 
                       'volatility', 'volume_ma']
        
        importances = model.feature_importances_
        
        return dict(zip(features, importances))


# Test ML methods
if __name__ == "__main__":
    print("="*60)
    print("TESTING ML METHODS")
    print("="*60)
    
    # Load data
    fraud_df = pd.read_csv('data/fraud_data.csv')
    credit_df = pd.read_csv('data/credit_data.csv')
    trading_df = pd.read_csv('data/trading_data.csv')
    
    # Initialize
    ml = MLMethods()
    
    # Test fraud detection
    fraud_pred, fraud_metrics, fraud_model = ml.fraud_detection_ml(fraud_df)
    
    # Test credit scoring
    credit_pred, credit_metrics, credit_model = ml.credit_scoring_ml(credit_df)
    
    # Test trading
    trading_pred, trading_metrics, trading_model = ml.trading_strategy_ml(trading_df)
    
    print("\n✓ All ML methods tested!")
    
    # Show feature importance
    print("\n📊 Feature Importance (Fraud Detection):")
    for feature, importance in ml.get_feature_importance('fraud').items():
        print(f"  {feature:25s}: {importance:.4f}")