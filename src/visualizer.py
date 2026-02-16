"""
Visualizer Module
-----------------
Creates visualizations comparing Traditional vs ML methods
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class ComparisonVisualizer:
    """Creates visualizations for method comparisons"""
    
    def __init__(self, output_dir='outputs'):
        """Initialize visualizer"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def plot_accuracy_comparison(self, comparison_results):
        """
        Create bar chart comparing accuracies across use cases
        
        Args:
            comparison_results: Dictionary of comparison results
        """
        print("\n📊 Creating accuracy comparison chart...")
        
        use_cases = list(comparison_results.keys())
        trad_acc = [comparison_results[uc]['traditional']['accuracy'] 
                   for uc in use_cases]
        ml_acc = [comparison_results[uc]['ml']['accuracy'] 
                 for uc in use_cases]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(use_cases))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, trad_acc, width, label='Traditional', 
                      color='#e74c3c', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, ml_acc, width, label='Machine Learning', 
                      color='#2ecc71', alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1%}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Use Case', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Traditional vs ML: Accuracy Comparison Across Banking Use Cases', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(use_cases)
        ax.legend(loc='lower right', fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'accuracy_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_improvement_percentages(self, comparison_results):
        """
        Create chart showing improvement percentages
        
        Args:
            comparison_results: Dictionary of comparison results
        """
        print("\n📈 Creating improvement chart...")
        
        use_cases = list(comparison_results.keys())
        improvements = [comparison_results[uc]['improvements']['accuracy'] 
                       for uc in use_cases]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#27ae60' if imp > 20 else '#3498db' if imp > 10 else '#f39c12' 
                  for imp in improvements]
        bars = ax.barh(range(len(use_cases)), improvements, 
                      color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            ax.text(imp + 1, i, f'+{imp:.1f}%', 
                   va='center', fontsize=12, fontweight='bold')
        
        ax.set_yticks(range(len(use_cases)))
        ax.set_yticklabels(use_cases, fontsize=11)
        ax.set_xlabel('Accuracy Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_title('ML Improvement Over Traditional Methods', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'improvement_chart.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_metrics_radar(self, comparison_results, use_case):
        """
        Create radar chart comparing all metrics for one use case
        
        Args:
            comparison_results: Dictionary of comparison results
            use_case: Specific use case to visualize
        """
        print(f"\n🎯 Creating radar chart for {use_case}...")
        
        if use_case not in comparison_results:
            print(f"✗ No data for {use_case}")
            return
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        trad_vals = [
            comparison_results[use_case]['traditional']['accuracy'],
            comparison_results[use_case]['traditional']['precision'],
            comparison_results[use_case]['traditional']['recall'],
            comparison_results[use_case]['traditional']['f1_score']
        ]
        ml_vals = [
            comparison_results[use_case]['ml']['accuracy'],
            comparison_results[use_case]['ml']['precision'],
            comparison_results[use_case]['ml']['recall'],
            comparison_results[use_case]['ml']['f1_score']
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        trad_vals += trad_vals[:1]
        ml_vals += ml_vals[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, trad_vals, 'o-', linewidth=2, label='Traditional', 
               color='#e74c3c')
        ax.fill(angles, trad_vals, alpha=0.25, color='#e74c3c')
        
        ax.plot(angles, ml_vals, 'o-', linewidth=2, label='ML', 
               color='#2ecc71')
        ax.fill(angles, ml_vals, alpha=0.25, color='#2ecc71')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(f'{use_case}: Traditional vs ML\nPerformance Comparison', 
                    fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, f'{use_case.lower().replace(" ", "_")}_radar.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_business_impact(self, impacts):
        """
        Create chart showing annual savings from ML adoption
        
        Args:
            impacts: List of business impact dictionaries
        """
        print("\n💰 Creating business impact chart...")
        
        use_cases = [imp['use_case'] for imp in impacts]
        savings = [imp['annual_savings'] for imp in impacts]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(range(len(use_cases)), savings, 
                     color='#27ae60', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, saving in zip(bars, savings):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${saving/1e6:.1f}M',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xticks(range(len(use_cases)))
        ax.set_xticklabels(use_cases)
        ax.set_ylabel('Annual Savings (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Business Impact: Annual Cost Savings from ML Adoption', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Format y-axis as millions
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'business_impact.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_all_metrics_comparison(self, comparison_results):
        """
        Create comprehensive comparison of all metrics across use cases
        
        Args:
            comparison_results: Dictionary of comparison results
        """
        print("\n📊 Creating comprehensive metrics comparison...")
        
        use_cases = list(comparison_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            trad_vals = [comparison_results[uc]['traditional'][metric] 
                        for uc in use_cases]
            ml_vals = [comparison_results[uc]['ml'][metric] 
                      for uc in use_cases]
            
            x = np.arange(len(use_cases))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, trad_vals, width, label='Traditional', 
                          color='#e74c3c', alpha=0.8)
            bars2 = ax.bar(x + width/2, ml_vals, width, label='ML', 
                          color='#2ecc71', alpha=0.8)
            
            ax.set_xlabel('Use Case', fontsize=10, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
            ax.set_title(f'{metric_name} Comparison', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(use_cases, rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=9)
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Comprehensive Performance Comparison: Traditional vs ML', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'all_metrics_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def generate_all_visualizations(self, comparison_results, business_impacts):
        """
        Generate all visualizations at once
        
        Args:
            comparison_results: Dictionary of comparison results
            business_impacts: List of business impact dictionaries
        """
        print("\n" + "="*60)
        print("GENERATING ALL VISUALIZATIONS")
        print("="*60)
        
        self.plot_accuracy_comparison(comparison_results)
        self.plot_improvement_percentages(comparison_results)
        self.plot_all_metrics_comparison(comparison_results)
        
        # Radar charts for each use case
        for use_case in comparison_results.keys():
            self.plot_metrics_radar(comparison_results, use_case)
        
        self.plot_business_impact(business_impacts)
        
        print("\n" + "="*60)
        print("✓ ALL VISUALIZATIONS COMPLETE!")
        print(f"✓ Saved to: {self.output_dir}/")
        print("="*60)


# Test visualizer
if __name__ == "__main__":
    print("="*60)
    print("TESTING VISUALIZER")
    print("="*60)
    
    visualizer = ComparisonVisualizer()
    print("\n✓ Visualizer initialized")
    print(f"  Output directory: {visualizer.output_dir}")