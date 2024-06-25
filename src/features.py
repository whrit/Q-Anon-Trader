import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

def select_optimal_indicators(df, target_col='close', n_select=10):
    # Add all technical indicators
    df_indicators = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume"
    )
    
    # Remove columns with NaN values
    df_indicators = df_indicators.dropna(axis=1)
    
    # Prepare the feature matrix and target variable
    X = df_indicators.drop(columns=['open', 'high', 'low', 'close', 'volume'])
    y = df_indicators[target_col].pct_change().shift(-1).dropna()
    
    # Align X and y
    X = X.iloc[:-1]
    y = y.iloc[1:]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate mutual information scores
    mi_scores = mutual_info_regression(X_scaled, y)
    
    # Create a dataframe of features and their MI scores
    mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
    
    # Sort by MI score and select top n_select features
    top_features = mi_df.sort_values('mi_score', ascending=False).head(n_select)['feature'].tolist()
    
    return top_features

def apply_optimal_indicators(df, optimal_features):
    df_indicators = add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume"
    )
    return df_indicators[['open', 'high', 'low', 'close', 'volume'] + optimal_features]