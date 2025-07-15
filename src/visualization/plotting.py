# src/visualization/plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

def plot_feature_importance(ax, feature_importance_df):
    """繪製特徵重要性圖。"""
    if feature_importance_df is None or feature_importance_df.empty:
        ax.text(0.5, 0.5, 'Feature Importance Data\nNot Available', ha='center', va='center')
        return
    top_features = feature_importance_df.head(10)
    sns.barplot(x='importance', y='feature', data=top_features, ax=ax, color='steelblue')
    ax.set_title('Top 10 Important Features')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')

def plot_training_history(ax, history):
    """繪製神經網路訓練歷史圖。"""
    if not history or 'loss' not in history.history:
        ax.text(0.5, 0.5, 'Training History\nNot Available', ha='center', va='center')
        return
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Neural Network Training Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

def plot_efficiency_improvement(ax, sim_results):
    """繪製效率改善分佈圖。"""
    if sim_results is None or sim_results.empty:
        ax.text(0.5, 0.5, 'Simulation Results\nNot Available', ha='center', va='center')
        return
    sns.histplot(sim_results['improvement'], kde=True, ax=ax, color='lightgreen')
    mean_improve = sim_results['improvement'].mean()
    ax.axvline(mean_improve, color='red', linestyle='--', label=f'Mean: {mean_improve:.4f}')
    ax.set_title('Efficiency Improvement Distribution')
    ax.set_xlabel('Improvement')
    ax.legend()

def create_comprehensive_visualization(feature_importance_df=None, history=None, sim_results=None, output_path='comprehensive_visualization.png'):
    """建立一個包含多個子圖的綜合視覺化圖表。"""
    print("🎨 正在產生綜合視覺化圖表...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Network Slice Optimization Analysis', fontsize=18, fontweight='bold')
    
    # 繪製各個圖表
    plot_feature_importance(axes[0, 0], feature_importance_df)
    plot_training_history(axes[0, 1], history)
    plot_efficiency_improvement(axes[1, 0], sim_results)
    
    # 第四個圖表留白或用於其他分析
    axes[1, 1].text(0.5, 0.5, 'Additional Analysis Plot', ha='center', va='center', fontsize=12, color='gray')
    axes[1, 1].set_title('Placeholder')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300)
    print(f"✅ 圖表已儲存至: {output_path}")
    plt.show()

