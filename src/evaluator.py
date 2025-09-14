"""
Comprehensive model evaluation module
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class ModelEvaluator:
    def __init__(self):
        """Initialize model evaluator"""
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive evaluation of a single model"""
        print(f"📈 Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        self.results[model_name] = metrics
        
        # Print results
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        
        return metrics
    
    def compare_models(self):
        """Compare all evaluated models"""
        if not self.results:
            print("❌ No models to compare")
            return None
        
        print("\n📊 MODEL COMPARISON")
        print("=" * 60)
        
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False, float_format='%.4f'))
        
        # Determine best model
        best_model = df_comparison.loc[df_comparison['F1-Score'].idxmax(), 'Model']
        print(f"\n🏆 Best performing model: {best_model}")
        
        return df_comparison
    
    def create_confusion_matrix_plot(self, model_name):
        """Create confusion matrix heatmap"""
        if model_name not in self.results:
            return None
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'],
            cbar_kws={'label': 'Count'}
        )
        
        plt.title(f'{model_name.replace("_", " ").title()} - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        return plt
    
    def create_metrics_comparison_chart(self):
        """Create interactive metrics comparison chart"""
        if not self.results:
            return None
        
        # Prepare data
        models = []
        metrics_data = {
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }
        
        for model_name, metrics in self.results.items():
            models.append(model_name.replace('_', ' ').title())
            metrics_data['Accuracy'].append(metrics['accuracy'])
            metrics_data['Precision'].append(metrics['precision'])
            metrics_data['Recall'].append(metrics['recall'])
            metrics_data['F1-Score'].append(metrics['f1_score'])
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=['Model Performance Comparison']
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=models,
                    y=values,
                    marker_color=colors[i],
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto'
                )
            )
        
        fig.update_layout(
            title='Model Performance Metrics Comparison',
            xaxis_title='Models',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1]),
            barmode='group',
            height=500
        )
        
        return fig
    
    def create_detailed_report(self):
        """Create detailed evaluation report"""
        if not self.results:
            return "No models evaluated"
        
        report = []
        report.append("DETAILED MODEL EVALUATION REPORT")
        report.append("=" * 50)
        
        for model_name, metrics in self.results.items():
            report.append(f"\n{model_name.upper().replace('_', ' ')} MODEL:")
            report.append("-" * 30)
            report.append(f"Accuracy:  {metrics['accuracy']:.4f}")
            report.append(f"Precision: {metrics['precision']:.4f}")
            report.append(f"Recall:    {metrics['recall']:.4f}")
            report.append(f"F1-Score:  {metrics['f1_score']:.4f}")
            
            cm = metrics['confusion_matrix']
            report.append(f"\nConfusion Matrix:")
            report.append(f"True Negative (Ham):  {cm[0][0]}")
            report.append(f"False Positive:       {cm[0][1]}")
            report.append(f"False Negative:       {cm[1][0]}")
            report.append(f"True Positive (Spam): {cm[1][1]}")
        
        return "\n".join(report)
