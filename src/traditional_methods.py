"""
Traditional Methods Module
---------------------------
Implements old-school rule-based approaches used before ML
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


class TraditionalMethods:
    """Traditional rule-based methods for banking"""
    
    def __init__(self):
        """Initialize traditional methods"""
        pass
    
    def fraud_detection_rules(self, df):
        """
        Traditional fraud detection using simple rules
        
        Rules:
        1. Amount > $500 = Flag as fraud
        2. Hour between 12am-6am = Flag as fraud
        3. Distance > 100 miles = Flag as fraud
        4. More than 10 transactions in 24h = Flag as fraud
        
        Returns:
            predictions, execution_time
        """
        print("\n🔍 Running Traditional Fraud Detection (Rule-Based)...")
        start_time = time.time()
        
        predictions = np.zeros(len(df))
        
        # Apply rules
        for i, row in df.iterrows():
            fraud_score = 0
            
            # Rule 1: High amount
            if row['amount'] > 500:
                fraud_score += 1
            
            # Rule 2: Late night
            if row['hour'] < 6:
                fraud_score += 1
            
            # Rule 3: Far from home
            if row['distance_from_home'] > 100:
                fraud_score += 1
            
            # Rule 4: Too many transactions
            if row['transactions_last_24h'] > 10:
                fraud_score += 1
            
            # Flag as fraud if 2+ rules triggered
            if fraud_score >= 2:
                predictions[i] = 1
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        y_true = df['is_fraud']
        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions, zero_division=0),
            'recall': recall_score(y_true, predictions, zero_division=0),
            'f1_score': f1_score(y_true, predictions, zero_division=0),
            'execution_time': execution_time
        }
        
        print(f"✓ Completed in {execution_time:.4f} seconds")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Flagged {int(predictions.sum())} transactions as fraud")
        
        return predictions, metrics
    
    def credit_scoring_rules(self, df):
        """
        Traditional credit scoring using manual underwriting rules
        
        Rules (FICO-style):
        1. Credit history < 24 months = Reject
        2. Debt-to-income > 0.5 = Reject
        3. Income < $30,000 = Reject
        4. Loan amount > 3x income = Reject
        
        Returns:
            predictions, execution_time
        """
        print("\n🔍 Running Traditional Credit Scoring (Manual Underwriting)...")
        start_time = time.time()
        
        predictions = np.zeros(len(df))  # 0 = Approve, 1 = Reject (predict default)
        
        # Apply rules
        for i, row in df.iterrows():
            reject = False
            
            # Rule 1: Short credit history
            if row['credit_history_months'] < 24:
                reject = True
            
            # Rule 2: High debt-to-income
            if row['debt_to_income'] > 0.5:
                reject = True
            
            # Rule 3: Low income
            if row['income'] < 30000:
                reject = True
            
            # Rule 4: Loan too large relative to income
            if row['loan_amount'] > (row['income'] * 3):
                reject = True
            
            if reject:
                predictions[i] = 1  # Predict default
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        y_true = df['default']
        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions, zero_division=0),
            'recall': recall_score(y_true, predictions, zero_division=0),
            'f1_score': f1_score(y_true, predictions, zero_division=0),
            'execution_time': execution_time
        }
        
        print(f"✓ Completed in {execution_time:.4f} seconds")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Rejected {int(predictions.sum())} applications")
        
        return predictions, metrics
    
    def trading_strategy_rules(self, df):
        """
        Traditional trading using technical indicators
        
        Rules:
        1. Buy if: MA_5 > MA_20 (golden cross)
        2. Sell if: MA_5 < MA_20 (death cross)
        
        Returns:
            predictions, execution_time
        """
        print("\n🔍 Running Traditional Trading (Technical Indicators)...")
        start_time = time.time()
        
        predictions = np.zeros(len(df))
        
        # Apply rules
        for i, row in df.iterrows():
            # Buy signal (predict price up)
            if row['ma_5'] > row['ma_20']:
                predictions[i] = 1
        
        execution_time = time.time() - start_time
        
        # Calculate metrics
        y_true = df['price_up']
        metrics = {
            'accuracy': accuracy_score(y_true, predictions),
            'precision': precision_score(y_true, predictions, zero_division=0),
            'recall': recall_score(y_true, predictions, zero_division=0),
            'f1_score': f1_score(y_true, predictions, zero_division=0),
            'execution_time': execution_time
        }
        
        print(f"✓ Completed in {execution_time:.4f} seconds")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Generated {int(predictions.sum())} buy signals")
        
        return predictions, metrics


# Test traditional methods
if __name__ == "__main__":
    print("="*60)
    print("TESTING TRADITIONAL METHODS")
    print("="*60)
    
    # Load data
    fraud_df = pd.read_csv('data/fraud_data.csv')
    credit_df = pd.read_csv('data/credit_data.csv')
    trading_df = pd.read_csv('data/trading_data.csv')
    
    # Initialize
    traditional = TraditionalMethods()
    
    # Test fraud detection
    fraud_pred, fraud_metrics = traditional.fraud_detection_rules(fraud_df)
    
    # Test credit scoring
    credit_pred, credit_metrics = traditional.credit_scoring_rules(credit_df)
    
    # Test trading
    trading_pred, trading_metrics = traditional.trading_strategy_rules(trading_df)
    
    print("\n✓ All traditional methods tested!")