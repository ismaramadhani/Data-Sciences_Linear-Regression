import matplotlib.pyplot as plt
import seaborn as sns

def histogram(df, column, title="Histogram"):
    """Generate a histogram for a given column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, color='skyblue')
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    return plt

def boxplot(df, column, title="Boxplot"):
    """Generate a boxplot for a given column."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column], color='skyblue')
    plt.title(title)
    plt.xlabel(column)
    plt.tight_layout()
    return plt

def lineplot(df, column, title="Line Plot"):
    """Generate a lineplot for a given column."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=df.index, y=column, color='skyblue')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.tight_layout()
    return plt
