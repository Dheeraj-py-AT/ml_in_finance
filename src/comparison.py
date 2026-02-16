"""
Comparison Engine Module
-------------------------
Compares Traditional vs ML methods across all use cases
"""

import pandas as pd
import numpy as np
from tabulate import tabulate


class ComparisonEngine:
    """Compares traditional and ML methods"""
    
    def __init__(self):
        """Initialize comparison engine"""
        self.results = {}
    
    def compare_methods(self, traditional_metrics, ml_metrics, use_case_name):
        """
        Compare traditional vs ML metrics
        
        Args:
            traditional_metrics: Dict of traditional method metrics
            ml_metrics: Dict of ML method metrics
            use_case_name: Name of the use case
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            'use_case': use_case_name,
            'traditional': traditional_metrics,
            'ml': ml_metrics,
            'improvements': {}
        }
        
        # Calculate improvements
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            trad_val = traditional_metrics[metric]
            ml_val = ml_metrics[metric]
            
            if trad_val > 0:
                improvement = ((ml_val - trad_val) / trad_val) * 100
            else:
                improvement = 0
            
            comparison['improvements'][metric] = improvement
        
        # Store results
        self.results[use_case_name] = comparison
        
        return comparison
    
    def print_comparison_table(self, use_case_name):
        """
        Print detailed comparison table for a use case
        
        Args:
            use_case_name: Name of the use case
        """
        if use_case_name not in self.results:
            print(f"No results for {use_case_name}")
            return
        
        comp = self.results[use_case_name]
        trad = comp['traditional']
        ml = comp['ml']
        imp = comp['improvements']
        
        print(f"\n{'='*70}")
        print(f"  {use_case_name.upper()} - TRADITIONAL VS ML")
        print(f"{'='*70}")
        
        # Create table data
        table_data = [
            ['Accuracy', f"{trad['accuracy']:.2%}", f"{ml['accuracy']:.2%}", 
             f"+{imp['accuracy']:.1f}%"],
            ['Precision', f"{trad['precision']:.2%}", f"{ml['precision']:.2%}", 
             f"+{imp['precision']:.1f}%"],
            ['Recall', f"{trad['recall']:.2%}", f"{ml['recall']:.2%}", 
             f"+{imp['recall']:.1f}%"],
            ['F1-Score', f"{trad['f1_score']:.2%}", f"{ml['f1_score']:.2%}", 
             f"+{imp['f1_score']:.1f}%"],
            ['Speed (sec)', f"{trad['execution_time']:.4f}", 
             f"{ml['execution_time']:.4f}", 
             f"{((ml['execution_time']/trad['execution_time'])-1)*100:.1f}%"]
        ]
        
        print(tabulate(table_data, 
                      headers=['Metric', 'Traditional', 'ML', 'Improvement'],
                      tablefmt='grid'))
    
    def generate_summary_report(self):
        """
        Generate summary report across all use cases
        
        Returns:
            Summary statistics dictionary
        """
        if not self.results:
            return {}
        
        summary = {
            'use_cases': list(self.results.keys()),
            'avg_accuracy_improvement': 0,
            'avg_precision_improvement': 0,
            'avg_recall_improvement': 0,
            'avg_f1_improvement': 0,
            'total_use_cases': len(self.results)
        }
        
        # Calculate averages
        acc_improvements = []
        prec_improvements = []
        rec_improvements = []
        f1_improvements = []
        
        for use_case, comp in self.results.items():
            acc_improvements.append(comp['improvements']['accuracy'])
            prec_improvements.append(comp['improvements']['precision'])
            rec_improvements.append(comp['improvements']['recall'])
            f1_improvements.append(comp['improvements']['f1_score'])
        
        summary['avg_accuracy_improvement'] = np.mean(acc_improvements)
        summary['avg_precision_improvement'] = np.mean(prec_improvements)
        summary['avg_recall_improvement'] = np.mean(rec_improvements)
        summary['avg_f1_improvement'] = np.mean(f1_improvements)
        
        return summary
    
    def print_summary_report(self):
        """Print overall summary report"""
        summary = self.generate_summary_report()
        
        print("\n" + "="*70)
        print("  OVERALL SUMMARY: WHY BANKS NEED ML")
        print("="*70)
        
        print(f"\n📊 Analyzed {summary['total_use_cases']} Use Cases:")
        for use_case in summary['use_cases']:
            print(f"  • {use_case}")
        
        print(f"\n📈 Average Improvements (ML vs Traditional):")
        print(f"  • Accuracy:  +{summary['avg_accuracy_improvement']:.1f}%")
        print(f"  • Precision: +{summary['avg_precision_improvement']:.1f}%")
        print(f"  • Recall:    +{summary['avg_recall_improvement']:.1f}%")
        print(f"  • F1-Score:  +{summary['avg_f1_improvement']:.1f}%")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"  ✓ ML consistently outperforms traditional methods")
        print(f"  ✓ Average improvement: {summary['avg_accuracy_improvement']:.1f}%")
        print(f"  ✓ Better at detecting complex patterns")
        print(f"  ✓ Adapts to changing data automatically")
    
    def calculate_business_impact(self, use_case_name, transaction_volume, 
                                  cost_per_error):
        """
        Calculate business impact of ML vs Traditional
        
        Args:
            use_case_name: Name of use case
            transaction_volume: Number of daily transactions
            cost_per_error: Cost of each error in dollars
            
        Returns:
            Business impact dictionary
        """
        if use_case_name not in self.results:
            return {}
        
        comp = self.results[use_case_name]
        
        # Calculate error rates
        trad_error_rate = 1 - comp['traditional']['accuracy']
        ml_error_rate = 1 - comp['ml']['accuracy']
        
        # Calculate daily costs
        trad_daily_cost = transaction_volume * trad_error_rate * cost_per_error
        ml_daily_cost = transaction_volume * ml_error_rate * cost_per_error
        
        # Calculate savings
        daily_savings = trad_daily_cost - ml_daily_cost
        annual_savings = daily_savings * 365
        
        impact = {
            'use_case': use_case_name,
            'transaction_volume': transaction_volume,
            'cost_per_error': cost_per_error,
            'traditional_error_rate': trad_error_rate,
            'ml_error_rate': ml_error_rate,
            'traditional_daily_cost': trad_daily_cost,
            'ml_daily_cost': ml_daily_cost,
            'daily_savings': daily_savings,
            'annual_savings': annual_savings,
            'roi_percentage': (annual_savings / ml_daily_cost) * 100 if ml_daily_cost > 0 else 0
        }
        
        return impact
    
    def print_business_impact(self, impact):
        """Print business impact report"""
        print(f"\n💰 BUSINESS IMPACT: {impact['use_case'].upper()}")
        print("="*70)
        print(f"  Transaction Volume: {impact['transaction_volume']:,} per day")
        print(f"  Cost per Error: ${impact['cost_per_error']:,.2f}")
        print(f"\n  Traditional Method:")
        print(f"    Error Rate: {impact['traditional_error_rate']:.2%}")
        print(f"    Daily Cost: ${impact['traditional_daily_cost']:,.2f}")
        print(f"\n  ML Method:")
        print(f"    Error Rate: {impact['ml_error_rate']:.2%}")
        print(f"    Daily Cost: ${impact['ml_daily_cost']:,.2f}")
        print(f"\n  💵 SAVINGS:")
        print(f"    Daily: ${impact['daily_savings']:,.2f}")
        print(f"    Annual: ${impact['annual_savings']:,.2f}")
        print(f"    ROI: {impact['roi_percentage']:.1f}%")


# Test comparison engine
if __name__ == "__main__":
    print("="*60)
    print("TESTING COMPARISON ENGINE")
    print("="*60)
    
    # Create dummy metrics for testing
    trad_metrics = {
        'accuracy': 0.75,
        'precision': 0.70,
        'recall': 0.65,
        'f1_score': 0.67,
        'execution_time': 0.05
    }
    
    ml_metrics = {
        'accuracy': 0.92,
        'precision': 0.90,
        'recall': 0.88,
        'f1_score': 0.89,
        'execution_time': 0.15
    }
    
    engine = ComparisonEngine()
    
    # Compare
    comparison = engine.compare_methods(trad_metrics, ml_metrics, 'Fraud Detection')
    
    # Print
    engine.print_comparison_table('Fraud Detection')
    
    # Business impact
    impact = engine.calculate_business_impact('Fraud Detection', 
                                             transaction_volume=100000,
                                             cost_per_error=50)
    engine.print_business_impact(impact)
    
    print("\n✓ Comparison engine tested!")