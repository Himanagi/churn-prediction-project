# churn_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

def plot_churn_distribution(df, column='Attrition_Flag'):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=column)
    plt.title(f"{column} Distribution")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

# Add any additional helper functions here as your project grows
