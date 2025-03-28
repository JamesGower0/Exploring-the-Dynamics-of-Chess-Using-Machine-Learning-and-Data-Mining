import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 1. Data Loading and Preparation with proper type conversion
print("Loading data...")
try:
    df = pd.read_csv('output.csv', low_memory=False)
    
    # Convert ELO columns to numeric, coercing errors to NaN
    df['WhiteElo'] = pd.to_numeric(df['WhiteElo'], errors='coerce')
    df['BlackElo'] = pd.to_numeric(df['BlackElo'], errors='coerce')
    
    # Drop rows where ELO is missing
    df = df.dropna(subset=['WhiteElo', 'BlackElo'])
    
    # Define strategies to analyze
    strategies = {
        'Battery': ('BatteryUsed', 'BatteryEvalBefore', 'BatteryEvalAfter'),
        'Fianchetto': ('FianchettoUsed', 'FianchettoEvalBefore', 'FianchettoEvalAfter'),
        'BishopPair': ('BishopPairUsed', 'BishopPairEvalBefore', 'BishopPairEvalAfter')
    }
    
    # Convert strategy evaluation columns to numeric
    for _, (_, before_col, after_col) in strategies.items():
        df[before_col] = pd.to_numeric(df[before_col], errors='coerce')
        df[after_col] = pd.to_numeric(df[after_col], errors='coerce')
    
    # Create ELO bins (300-3400 in 100-point increments)
    elo_bins = list(range(300, 3401, 100))
    df['WhiteEloBin'] = pd.cut(df['WhiteElo'], bins=elo_bins, right=False)
    df['BlackEloBin'] = pd.cut(df['BlackElo'], bins=elo_bins, right=False)
    
    # Calculate success metrics
    for name, (used_col, before_col, after_col) in strategies.items():
        # Convert used column to numeric if needed
        df[used_col] = pd.to_numeric(df[used_col], errors='coerce')
        df[f'{name}_Success'] = df[after_col] - df[before_col]
        df[f'{name}_Success'] = df[f'{name}_Success'].where(df[used_col] == 1)
    
    print("Data loaded and processed successfully!")
    
except Exception as e:
    print(f"Error during data loading: {str(e)}")
    print("Please check your CSV file format and column names")
    exit()

# 2. Basic Statistics
print("\nBasic Statistics:")
print(f"Total games: {len(df):,}")
print(f"White ELO range: {df['WhiteElo'].min():.0f}-{df['WhiteElo'].max():.0f}")
print(f"Black ELO range: {df['BlackElo'].min():.0f}-{df['BlackElo'].max():.0f}")

# 3. Visualization with bold lines
print("\nGenerating visualizations...")
plt.figure(figsize=(18, 12))

# Success rate by ELO
for i, strategy in enumerate(strategies.keys(), 1):
    plt.subplot(2, 2, i)
    
    # Plot mean success with bold line
    df.groupby('WhiteEloBin')[f'{strategy}_Success'].mean().plot(
        marker='o', 
        label='Mean Success',
        linewidth=3,  # Increased line width for bold effect
        markersize=8  # Larger markers for better visibility
    )
    
    # Plot success rate with bold line
    df.groupby('WhiteEloBin')[f'{strategy}_Success'].apply(
        lambda x: (x > 0).mean()).plot(
        marker='x', 
        label='Success Rate',
        linewidth=3,  # Increased line width for bold effect
        markersize=8  # Larger markers for better visibility
    )
    
    # Make title and labels bold
    plt.title(f'{strategy} Performance by White ELO', fontweight='bold')
    plt.xlabel('ELO Range', fontweight='bold')
    plt.ylabel('Performance Metric', fontweight='bold')
    
    # Make legend and grid more prominent
    plt.legend(prop={'weight': 'bold'})
    plt.grid(True, linewidth=1.5)  # Thicker grid lines

plt.tight_layout()
plt.savefig('strategy_performance.eps', format='eps', dpi=300)  # Higher DPI for better quality
plt.close()

# Distribution plots
plt.figure(figsize=(15, 10))
for i, strategy in enumerate(strategies.keys(), 1):
    plt.subplot(2, 2, i)
    sns.violinplot(data=df, x='WhiteEloBin', y=f'{strategy}_Success', 
                  cut=0, scale='width', inner='quartile')
    plt.title(f'{strategy} Success Distribution by ELO')
    plt.xticks(rotation=45)
    plt.ylim(-15, 15)  # Adjust based on your actual eval ranges

plt.tight_layout()
plt.savefig('strategy_distributions.png')
plt.close()

# 4. Statistical Analysis
print("\nStatistical Analysis:")
for strategy in strategies.keys():
    # Filter groups with at least 30 observations
    groups = []
    for _, group in df.groupby('WhiteEloBin'):
        success_data = group[f'{strategy}_Success'].dropna()
        if len(success_data) >= 30:
            groups.append(success_data)
    
    if len(groups) >= 2:  # Need at least 2 groups for ANOVA
        f_val, p_val = stats.f_oneway(*groups)
        print(f"{strategy}: F-value = {f_val:.2f}, p-value = {p_val:.4f}")

# 5. Machine Learning Analysis (Corrected)
print("\nTraining predictive models...")
# Prepare data
X = df[['WhiteElo']].copy()
features = ['WhiteElo']

# Add game type if available
if 'Event' in df.columns:
    # Extract time control categories
    df['TimeControl'] = 'Other'
    df.loc[df['Event'].str.contains('Bullet', case=False, na=False), 'TimeControl'] = 'Bullet'
    df.loc[df['Event'].str.contains('Blitz', case=False, na=False), 'TimeControl'] = 'Blitz'
    df.loc[df['Event'].str.contains('Classical', case=False, na=False), 'TimeControl'] = 'Classical'
    
    X = pd.get_dummies(X.join(df['TimeControl']), columns=['TimeControl'])
    features.extend([col for col in X.columns if 'TimeControl' in col])

# Model each strategy
results = []
for strategy in strategies.keys():
    y = df[f'{strategy}_Success'].dropna()
    X_clean = X.loc[y.index]
    
    if len(y) > 1000:  # Only model if sufficient data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=0.2, random_state=42)
        
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate RMSE (fixed version)
        mse = mean_squared_error(y_test, model.predict(X_test))
        rmse = np.sqrt(mse)
        
        importance = dict(zip(features, model.feature_importances_))
        
        results.append({
            'Strategy': strategy,
            'RMSE': rmse,
            'ELO_Importance': importance['WhiteElo'],
            'Samples': len(y)
        })
        
        # Save partial dependence plot
        plt.figure()
        PartialDependenceDisplay.from_estimator(model, X_clean, features=['WhiteElo'])
        plt.title(f'{strategy} Success vs White ELO')
        plt.savefig(f'{strategy}_pdp.png')
        plt.close()

# Print model results
print("\nModel Performance:")
print(pd.DataFrame(results))

# 6. Generate Final Report
print("\nGenerating final metrics...")
metrics = []
for elo_bin, group in df.groupby('WhiteEloBin'):
    for strategy in strategies.keys():
        success = group[f'{strategy}_Success'].dropna()
        if len(success) > 30:  # Only report bins with sufficient data
            metrics.append({
                'ELO_Bin': elo_bin,
                'Strategy': strategy,
                'Mean_Success': success.mean(),
                'Std_Dev': success.std(),
                'Success_Rate': (success > 0).mean(),
                'N_Games': len(success)
            })

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('strategy_metrics_by_elo.csv', index=False)

# Create pivot tables for easy reading
pivot_mean = metrics_df.pivot_table(index='ELO_Bin', columns='Strategy', values='Mean_Success')
pivot_rate = metrics_df.pivot_table(index='ELO_Bin', columns='Strategy', values='Success_Rate')

print("\nMean Success by ELO and Strategy:")
print(pivot_mean.to_markdown(floatfmt=".2f"))

print("\nSuccess Rate by ELO and Strategy:")
print(pivot_rate.to_markdown(floatfmt=".2%"))

print("\nAnalysis complete! Results saved to:")
print("- strategy_performance.png")
print("- strategy_distributions.png")
print("- [strategy]_pdp.png (partial dependence plots)")
print("- strategy_metrics_by_elo.csv")