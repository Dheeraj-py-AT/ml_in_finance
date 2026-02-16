"""
Data Generator Module
---------------------
Generates synthetic banking data for demonstration:
1. Fraud Detection data
2. Credit Scoring data
3. Stock Trading data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class BankingDataGenerator:
    """Generates realistic synthetic banking data"""
    
    def __init__(self, random_seed=42):
        """Initialize generator with random seed for reproducibility"""
        np.random.seed(random_seed)
        self.random_seed = random_seed
    
    def generate_fraud_data(self, n_samples=1000):
        """
        Generate credit card transaction data
        
        Features:
        - Transaction amount
        - Time of day
        - Location distance from home
        - Merchant category
        - Previous transaction frequency
        
        Target: Fraud (1) or Legitimate (0)
        
        Returns:
            DataFrame with transaction data
        """
        print(f"Generating {n_samples} fraud detection samples...")
        
        # Normal transactions (90%)
        n_normal = int(n_samples * 0.9)
        n_fraud = n_samples - n_normal
        
        # Normal transactions
        normal_data = {
            'amount': np.random.normal(50, 30, n_normal).clip(5, 500),
            'hour': np.random.choice(range(8, 23), n_normal),  # Business hours
            'distance_from_home': np.random.exponential(5, n_normal).clip(0, 50),
            'merchant_risk_score': np.random.normal(0.2, 0.1, n_normal).clip(0, 1),
            'transactions_last_24h': np.random.poisson(3, n_normal),
            'is_fraud': 0
        }
        
        # Fraudulent transactions (unusual patterns)
        fraud_data = {
            'amount': np.random.normal(300, 200, n_fraud).clip(100, 2000),  # Higher amounts
            'hour': np.random.choice(range(0, 6), n_fraud),  # Late night
            'distance_from_home': np.random.normal(500, 300, n_fraud).clip(100, 2000),  # Far from home
            'merchant_risk_score': np.random.normal(0.7, 0.2, n_fraud).clip(0.5, 1),  # Risky merchants
            'transactions_last_24h': np.random.poisson(15, n_fraud),  # Many transactions
            'is_fraud': 1
        }
        
        # Combine
        df_normal = pd.DataFrame(normal_data)
        df_fraud = pd.DataFrame(fraud_data)
        df = pd.concat([df_normal, df_fraud], ignore_index=True)
        
        # Shuffle
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"✓ Generated: {len(df)} transactions ({n_fraud} fraudulent)")
        return df
    
    def generate_credit_data(self, n_samples=1000):
        """
        Generate loan application data
        
        Features:
        - Age
        - Income
        - Credit history (months)
        - Existing debt
        - Loan amount requested
        
        Target: Default (1) or Repay (0)
        
        Returns:
            DataFrame with loan application data
        """
        print(f"Generating {n_samples} credit scoring samples...")
        
        # Good borrowers (70%)
        n_good = int(n_samples * 0.7)
        n_bad = n_samples - n_good
        
        # Good borrowers
        good_data = {
            'age': np.random.normal(40, 12, n_good).clip(22, 70),
            'income': np.random.normal(75000, 25000, n_good).clip(30000, 200000),
            'credit_history_months': np.random.normal(120, 60, n_good).clip(12, 360),
            'existing_debt': np.random.normal(10000, 8000, n_good).clip(0, 50000),
            'loan_amount': np.random.normal(25000, 15000, n_good).clip(5000, 100000),
            'employment_length_years': np.random.normal(8, 4, n_good).clip(1, 30),
            'default': 0
        }
        
        # Bad borrowers (higher risk profiles)
        bad_data = {
            'age': np.random.normal(28, 8, n_bad).clip(18, 50),
            'income': np.random.normal(35000, 15000, n_bad).clip(20000, 80000),
            'credit_history_months': np.random.normal(24, 18, n_bad).clip(0, 60),
            'existing_debt': np.random.normal(25000, 15000, n_bad).clip(5000, 100000),
            'loan_amount': np.random.normal(40000, 20000, n_bad).clip(10000, 150000),
            'employment_length_years': np.random.normal(2, 2, n_bad).clip(0, 10),
            'default': 1
        }
        
        # Combine
        df_good = pd.DataFrame(good_data)
        df_bad = pd.DataFrame(bad_data)
        df = pd.concat([df_good, df_bad], ignore_index=True)
        
        # Calculate debt-to-income ratio
        df['debt_to_income'] = df['existing_debt'] / df['income']
        
        # Shuffle
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"✓ Generated: {len(df)} applications ({n_bad} high-risk)")
        return df
    
    def generate_trading_data(self, n_days=500):
        """
        Generate stock price data with patterns
        
        Features:
        - Price history
        - Volume
        - Technical indicators (calculated)
        
        Target: Price movement (Up=1, Down=0)
        
        Returns:
            DataFrame with daily stock data
        """
        print(f"Generating {n_days} days of trading data...")
        
        # Generate base price series (random walk with drift)
        returns = np.random.normal(0.0005, 0.02, n_days)  # Small positive drift
        prices = 100 * np.exp(np.cumsum(returns))  # Convert to prices
        
        # Generate volume
        volumes = np.random.lognormal(15, 0.5, n_days)
        
        # Create DataFrame
        dates = [datetime.now() - timedelta(days=n_days-i) for i in range(n_days)]
        
        df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'volume': volumes
        })
        
        # Calculate technical indicators
        df['ma_5'] = df['price'].rolling(window=5).mean()
        df['ma_20'] = df['price'].rolling(window=20).mean()
        df['volatility'] = df['price'].rolling(window=20).std()
        df['volume_ma'] = df['volume'].rolling(window=5).mean()
        
        # Price change (target)
        df['price_change'] = df['price'].diff()
        df['price_up'] = (df['price_change'] > 0).astype(int)
        
        # Drop NaN rows from rolling calculations
        df = df.dropna().reset_index(drop=True)
        
        print(f"✓ Generated: {len(df)} trading days")
        return df
    
    def save_all_datasets(self):
        """Generate and save all datasets"""
        import os
        os.makedirs('data', exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING ALL DATASETS")
        print("="*60 + "\n")
        
        # Generate datasets
        fraud_df = self.generate_fraud_data(1000)
        credit_df = self.generate_credit_data(1000)
        trading_df = self.generate_trading_data(500)
        
        # Save to CSV
        fraud_df.to_csv('data/fraud_data.csv', index=False)
        credit_df.to_csv('data/credit_data.csv', index=False)
        trading_df.to_csv('data/trading_data.csv', index=False)
        
        print("\n✓ All datasets saved to data/ folder")
        
        return fraud_df, credit_df, trading_df


# Test the generator
if __name__ == "__main__":
    generator = BankingDataGenerator()
    
    print("="*60)
    print("TESTING DATA GENERATOR")
    print("="*60)
    
    # Generate and save all
    fraud_df, credit_df, trading_df = generator.save_all_datasets()
    
    # Show samples
    print("\n📊 FRAUD DETECTION DATA (first 5 rows):")
    print(fraud_df.head())
    
    print("\n📊 CREDIT SCORING DATA (first 5 rows):")
    print(credit_df.head())
    
    print("\n📊 TRADING DATA (first 5 rows):")
    print(trading_df.head())
    
    print("\n✓ Data generation complete!")