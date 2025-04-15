import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
warnings.filterwarnings('ignore')

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} fights")
    return df

def preprocess_data(df, impute_method='knn'):
    print("Preprocessing data...")
    
    # Make a copy to avoid warnings about setting values on a slice
    df = df.copy()
    
    # Convert date column to datetime if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        
    # Target variable - assuming 'Winner' column indicates the fight winner
    target = 'Winner'
    
    # If Winner is categorical (like 'Red' or 'Blue'), convert to binary
    if df[target].dtype == 'object':
        print("Converting Winner to binary target (Red=1, Blue=0)")
        # Convert winner names to uppercase to handle any case inconsistencies
        df['binary_winner'] = df[target].apply(lambda x: 1 if x.upper() == 'RED' else 0)
        target = 'binary_winner'
    
    # Define important features from each category with priority
    performance_features = [
        'RedWins', 'BlueLosses', 'RedLosses', 'BlueWins',
        'RedCurrentWinStreak', 'BlueCurrentWinStreak',
        'RedCurrentLoseStreak', 'BlueCurrentLoseStreak',
        'RedLongestWinStreak', 'BlueLongestWinStreak',
        'RedTotalRoundsFought', 'BlueTotalRoundsFought',
        'RedWinsByKO', 'BlueWinsByKO', 
        'RedWinsBySubmission', 'BlueWinsBySubmission',
        'RedWinsByDecisionUnanimous', 'BlueWinsByDecisionUnanimous',
        'RedAvgSigStrLanded', 'BlueAvgSigStrLanded',
        'RedAvgSigStrPct', 'BlueAvgSigStrPct', 
        'RedAvgSubAtt', 'BlueAvgSubAtt',
        'RedAvgTDLanded', 'BlueAvgTDLanded',
        'RedAvgTDPct', 'BlueAvgTDPct'
    ]
    
    physical_features = [
        'RedHeightCms', 'BlueHeightCms',
        'RedReachCms', 'BlueReachCms',
        'RedAge', 'BlueAge',
        'RedWeightLbs', 'BlueWeightLbs'
    ]
    
    fight_context_features = [
        'TitleBout', 'NumberOfRounds'
        # Removed 'RedOdds', 'BlueOdds' to exclude betting odds from prediction
    ]
    
    # Computed differential features
    differential_features = [
        'HeightDif', 'ReachDif', 'AgeDif',
        'WinDif', 'LossDif', 'WinStreakDif',
        'LongestWinStreakDif', 'TotalRoundDif',
        'KODif', 'SubDif', 'SigStrDif', 
        'AvgSubAttDif', 'AvgTDDif'
    ]
    
    # Ranking features - these often have many missing values
    ranking_features = [
        'RMatchWCRank', 'BMatchWCRank',
        'RPFPRank', 'BPFPRank'
    ]
    
    # ELO features - include but don't prioritize
    elo_features = [
        'RedElo', 'BlueElo', 'EloDifference'
    ]
    
    # Combine all features
    all_features = (
        performance_features + 
        physical_features + 
        fight_context_features + 
        differential_features +
        ranking_features +
        elo_features
    )
    
    # Keep only columns that exist in the dataframe
    features = [f for f in all_features if f in df.columns]
    print(f"Selected {len(features)} features")
    
    # Create feature matrix X and target vector y before imputation
    X_raw = df[features]
    y = df[target]
    
    # Handle missing values based on chosen method
    if impute_method == 'drop':
        # Drop rows with any missing values
        print("Dropping rows with missing values...")
        X_complete = X_raw.dropna()
        y = y.loc[X_complete.index]
        return X_complete, y
    
    elif impute_method == 'knn':
        # Use KNN imputation which respects feature relationships better than mean/median
        print("Using KNN imputation for missing values...")
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X_raw)
        X = pd.DataFrame(X_imputed, columns=X_raw.columns, index=X_raw.index)
    
    else:
        raise ValueError(f"Unknown imputation method: {impute_method}")
    
    # Print feature names and missing value counts
    print("\nFeature missing values after imputation:")
    for col in X.columns:
        missing = X[col].isnull().sum()
        if missing > 0:
            print(f"- {col}: {missing} missing values")
    
    # Remove any remaining rows with NaN if they exist
    X = X.dropna()
    y = y.loc[X.index]
    
    print(f"Final dataset shape: {X.shape}")
    return X, y

def build_xgboost_model(X, y, tune_hyperparams=True):
    print("Building XGBoost model...")
    
    # Split the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create a pipeline with preprocessing and model
    if tune_hyperparams:
        # Define parameter grid for XGBoost
        param_grid = {
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__subsample': [0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.8, 0.9, 1.0],
            'classifier__gamma': [0, 0.1],
            'classifier__min_child_weight': [1, 3, 5]
        }
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42
            ))
        ])
        
        # Grid search with cross-validation
        print("Performing grid search for hyperparameter tuning...")
        grid_search = GridSearchCV(
            pipeline, param_grid, 
            cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        pipeline = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
    else:
        # Use default XGBoost with reasonable parameters
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(
                max_depth=5,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.9,
                colsample_bytree=0.9,
                gamma=0,
                min_child_weight=3,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42
            ))
        ])
        
        # Train the model
        print("Training model...")
        pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Get detailed classification report
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # ROC AUC score
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Feature importances
    feature_importances = pipeline[-1].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 20 Feature Importances:")
    print(importance_df.head(20))
    
    return pipeline, importance_df, X_test, y_test

def visualize_results(model, importance_df, X_test, y_test):
    print("Generating visualizations...")
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Feature importance plot - top 20 features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importances.png')
    
    # Confusion matrix visualization
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Blue Win', 'Red Win'],
                yticklabels=['Blue Win', 'Red Win'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png')
    
    # ROC curve
    from sklearn.metrics import roc_curve, auc
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('visualizations/roc_curve.png')
    
    # Feature correlation heatmap
    # Get top 15 features and their correlations
    top_features = importance_df.head(15)['Feature'].tolist()
    if len(top_features) >= 5:
        plt.figure(figsize=(12, 10))
        correlation_matrix = X_test[top_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', linewidths=0.5)
        plt.title('Correlation Between Top Features')
        plt.tight_layout()
        plt.savefig('visualizations/feature_correlation.png')
    
    print("Visualizations saved to the 'visualizations' directory.")

def save_model(model, filename='ufc_xgboost_model.pkl'):
    print(f"Saving model to {filename}...")
    joblib.dump(model, filename)
    print("Model saved successfully!")
    
    # Also save feature names used by the model
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        # For pipelines, try to extract from the classifier
        try:
            feature_names = model[-1].feature_names_in_
        except (AttributeError, IndexError):
            feature_names = []
    
    if len(feature_names) > 0:
        with open('model_features.txt', 'w') as f:
            for feature in feature_names:
                f.write(f"{feature}\n")
        print("Feature names saved to model_features.txt")

def load_saved_model(filename='ufc_xgboost_model.pkl'):
    try:
        print(f"Loading model from {filename}...")
        model = joblib.load(filename)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def find_key_advantages(features):
    advantages = {
        'red': [],
        'blue': []
    }
    
    # Define thresholds for significant advantages
    thresholds = {
        'WinDif': 3,           # Win difference
        'LossDif': 3,          # Loss difference
        'WinStreakDif': 2,     # Win streak difference
        'HeightDif': 5.0,      # Height difference in cm
        'ReachDif': 5.0,       # Reach difference in cm
        'AgeDif': 5,           # Age difference in years
        'TotalRoundDif': 10,   # Experience difference in rounds
        'SigStrDif': 0.5,      # Significant strikes landed per minute
        'AvgSubAttDif': 0.5,   # Submission attempts per 15 minutes
        'AvgTDDif': 0.5,       # Takedowns landed per 15 minutes
        'EloDifference': 50,   # ELO rating difference
    }
    
    # Check for advantages in each key differential
    for feature, threshold in thresholds.items():
        if feature not in features:
            continue
            
        value = features[feature]
        if value >= threshold:
            # Red fighter advantage
            if feature == 'AgeDif':
                advantages['blue'].append(f"Younger by {abs(value)} years")
            else:
                feature_name = feature.replace('Dif', '').replace('Difference', '')
                advantages['red'].append(f"Better {feature_name} (+{value:.1f})")
        elif value <= -threshold:
            # Blue fighter advantage
            if feature == 'AgeDif':
                advantages['red'].append(f"Younger by {abs(value)} years")
            else:
                feature_name = feature.replace('Dif', '').replace('Difference', '')
                advantages['blue'].append(f"Better {feature_name} (+{abs(value):.1f})")
    
    # Check for specific advantages in non-differential features
    if 'RedWinsByKO' in features and 'BlueWinsByKO' in features:
        if features['RedWinsByKO'] >= features['BlueWinsByKO'] + 3:
            advantages['red'].append(f"More KO wins (+{features['RedWinsByKO'] - features['BlueWinsByKO']})")
        elif features['BlueWinsByKO'] >= features['RedWinsByKO'] + 3:
            advantages['blue'].append(f"More KO wins (+{features['BlueWinsByKO'] - features['RedWinsByKO']})")
    
    if 'RedWinsBySubmission' in features and 'BlueWinsBySubmission' in features:
        if features['RedWinsBySubmission'] >= features['BlueWinsBySubmission'] + 3:
            advantages['red'].append(f"More submission wins (+{features['RedWinsBySubmission'] - features['BlueWinsBySubmission']})")
        elif features['BlueWinsBySubmission'] >= features['RedWinsBySubmission'] + 3:
            advantages['blue'].append(f"More submission wins (+{features['BlueWinsBySubmission'] - features['RedWinsBySubmission']})")
        
    return advantages

def get_fighter_stats(df, fighter_name):
    stats = {}
    
    # Try to get the most recent fight data
    if 'Date' in df.columns:
        df = df.sort_values('Date', ascending=False)
    
    # Check if fighter has been in red corner
    red_matches = df[df['RedFighter'] == fighter_name]
    blue_matches = df[df['BlueFighter'] == fighter_name]
    
    if not red_matches.empty:
        # Get most recent red corner match
        match = red_matches.iloc[0]
        
        # Extract all columns that start with 'Red' and store without the 'Red' prefix
        for col in match.index:
            if col.startswith('Red') and col != 'RedFighter':
                feature_name = col[3:]  # Remove 'Red' prefix
                stats[feature_name] = match[col]
    
    elif not blue_matches.empty:
        # Get most recent blue corner match
        match = blue_matches.iloc[0]
        
        # Extract all columns that start with 'Blue' and store without the 'Blue' prefix
        for col in match.index:
            if col.startswith('Blue') and col != 'BlueFighter':
                feature_name = col[4:]  # Remove 'Blue' prefix
                stats[feature_name] = match[col]
    
    else:
        # This shouldn't happen as we checked existence earlier
        stats = {}
    
    return stats

def calculate_features(red_stats, blue_stats, df):
    features = {}
    
    # Add individual fighter stats
    # Red fighter stats
    for key, value in red_stats.items():
        features[f'Red{key}'] = value
    
    # Blue fighter stats
    for key, value in blue_stats.items():
        features[f'Blue{key}'] = value
    
    # Calculate differential features
    common_keys = set(red_stats.keys()) & set(blue_stats.keys())
    for key in common_keys:
        if key in ['Height', 'Reach', 'Age', 'Weight', 'Wins', 'Losses', 
                   'CurrentWinStreak', 'LongestWinStreak', 'TotalRoundsFought']:
            diff_key = f'{key}Dif'
            features[diff_key] = red_stats[key] - blue_stats[key]
    
    # Add specific computed differentials
    if 'Elo' in red_stats and 'Elo' in blue_stats:
        features['EloDifference'] = red_stats['Elo'] - blue_stats['Elo']
    
    if 'WinsByKO' in red_stats and 'WinsByKO' in blue_stats:
        features['KODif'] = red_stats['WinsByKO'] - blue_stats['WinsByKO']
    
    if 'WinsBySubmission' in red_stats and 'WinsBySubmission' in blue_stats:
        features['SubDif'] = red_stats['WinsBySubmission'] - blue_stats['WinsBySubmission']
    
    if 'AvgSigStrLanded' in red_stats and 'AvgSigStrLanded' in blue_stats:
        features['SigStrDif'] = red_stats['AvgSigStrLanded'] - blue_stats['AvgSigStrLanded']
    
    if 'AvgSubAtt' in red_stats and 'AvgSubAtt' in blue_stats:
        features['AvgSubAttDif'] = red_stats['AvgSubAtt'] - blue_stats['AvgSubAtt']
    
    if 'AvgTDLanded' in red_stats and 'AvgTDLanded' in blue_stats:
        features['AvgTDDif'] = red_stats['AvgTDLanded'] - blue_stats['AvgTDLanded']
    
    # Removed the odds median placeholder logic
    
    return features

def predict_from_names(model, red_fighter, blue_fighter, df):
    # First check if fighters exist in the dataset
    if red_fighter not in df['RedFighter'].values and red_fighter not in df['BlueFighter'].values:
        return f"Fighter {red_fighter} not found in the dataset", None, None, None
    
    if blue_fighter not in df['RedFighter'].values and blue_fighter not in df['BlueFighter'].values:
        return f"Fighter {blue_fighter} not found in the dataset", None, None, None
    
    red_stats = get_fighter_stats(df, red_fighter)
    blue_stats = get_fighter_stats(df, blue_fighter)
    
    features = calculate_features(red_stats, blue_stats, df)
    
    if hasattr(model, 'feature_names_in_'):
        required_features = model.feature_names_in_
    else:
        try:
            required_features = model[-1].feature_names_in_
        except (AttributeError, IndexError):
            required_features = list(features.keys())
    
    missing_features = set(required_features) - set(features.keys())
    for feature in missing_features:
        features[feature] = 0  # Use 0 as default for missing features
    
    X_pred = pd.DataFrame([{feat: features[feat] for feat in required_features}])
    
    prediction = model.predict(X_pred)[0]
    probability = model.predict_proba(X_pred)[0][1]  # Probability of red fighter winning
    
    # Determine confidence level
    if probability > 0.7 or probability < 0.3:
        confidence = "High"
    elif probability > 0.6 or probability < 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Find key advantages for explanation
    key_advantages = find_key_advantages(features)
    
    # Return result
    if prediction == 1:
        return red_fighter, probability, confidence, key_advantages
    else:
        return blue_fighter, 1 - probability, confidence, key_advantages

def predict_upcoming_fight():
    try:
        # Load the model
        model = load_saved_model()
        if model is None:
            print("Could not load model. Please train the model first.")
            return
        
        # Load the dataset for fighter information
        data_file = 'ufc_fights_with_elo.csv'
        print(f"Loading fighter data from {data_file}...")
        df = pd.read_csv(data_file)
        
        # Get fighter names
        red_fighter = input("Enter Red Corner fighter name: ")
        blue_fighter = input("Enter Blue Corner fighter name: ")
        
        # Make prediction
        winner, win_prob, confidence, key_advantages = predict_from_names(model, red_fighter, blue_fighter, df)
        
        if isinstance(winner, str) and (winner.startswith("Fighter") or confidence is None):
            print(winner)  # Error message
        else:
            print("\nFight Prediction Results:")
            print("========================")
            print(f"Red Fighter: {red_fighter}")
            print(f"Blue Fighter: {blue_fighter}")
            print(f"Prediction: {winner} will win with {win_prob:.2%} probability")
            print(f"Confidence Level: {confidence}")
            
            # If win probability is very close to 50%, give additional context
            if 0.45 <= win_prob <= 0.55:
                print("\nNote: This fight is very close to a toss-up according to the model.")
            
            # Print the key advantages
            if key_advantages:
                print("\nKey Advantages:")
                if winner == red_fighter and key_advantages['red']:
                    print(f"{red_fighter}'s advantages:")
                    for adv in key_advantages['red']:
                        print(f"- {adv}")
                elif winner == blue_fighter and key_advantages['blue']:
                    print(f"{blue_fighter}'s advantages:")
                    for adv in key_advantages['blue']:
                        print(f"- {adv}")
                
                # Print underdog's advantages if any
                if winner == red_fighter and key_advantages['blue']:
                    print(f"\n{blue_fighter}'s advantages despite predicted loss:")
                    for adv in key_advantages['blue']:
                        print(f"- {adv}")
                elif winner == blue_fighter and key_advantages['red']:
                    print(f"\n{red_fighter}'s advantages despite predicted loss:")
                    for adv in key_advantages['red']:
                        print(f"- {adv}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    file_path = 'ufc_fights_with_elo.csv'  # Using your actual file
    
    # Load data
    df = load_data(file_path)
    
    # Display basic info about the dataset
    print("\nDataset Info:")
    print(f"Shape: {df.shape}")
    print("\nColumns sample (first 10):")
    for col in list(df.columns)[:10]:
        print(f"- {col}: {df[col].dtype} ({df[col].isnull().sum()} missing values)")
    print(f"... plus {len(df.columns) - 10} more columns")
    
    # Preprocess data - using KNN imputation instead of mean/mode
    X, y = preprocess_data(df, impute_method='knn')
    
    # Check if we have features to work with
    if X.shape[1] == 0:
        print("Error: No valid features found in the dataset. Please check column names.")
        return
    
    # Build XGBoost model
    # Set tune_hyperparams to False for faster but less optimal model
    model, importance_df, X_test, y_test = build_xgboost_model(X, y, tune_hyperparams=False)
    
    # Visualize results
    visualize_results(model, importance_df, X_test, y_test)
    
    # Save model
    save_model(model, 'ufc_xgboost_model.pkl')
    
    # Example prediction using fighter names
    print("\nPrediction from fighter names:")
    try:
        # Sort the DataFrame by date if it has a Date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
        # Choose example fighters from the dataset
        example_fighters = df[['RedFighter', 'BlueFighter']].iloc[-100:] # Get fighters from recent matches
        red_fighter = example_fighters['RedFighter'].iloc[0]
        blue_fighter = example_fighters['BlueFighter'].iloc[0]
        
        winner, win_prob, confidence, key_advantages = predict_from_names(model, red_fighter, blue_fighter, df)
        
        if isinstance(winner, str) and (winner.startswith("Fighter") or confidence is None):
            print(winner)  # Error message
        else:
            print(f"Red Fighter: {red_fighter}")
            print(f"Blue Fighter: {blue_fighter}")
            print(f"Prediction: {winner} will win with {win_prob:.2%} probability")
            print(f"Confidence: {confidence}")
            
            # Print key advantages
            if key_advantages:
                print("\nKey Advantages:")
                if key_advantages['red']:
                    print(f"{red_fighter}'s advantages:")
                    for adv in key_advantages['red']:
                        print(f"- {adv}")
                
                if key_advantages['blue']:
                    print(f"\n{blue_fighter}'s advantages:")
                    for adv in key_advantages['blue']:
                        print(f"- {adv}")
    except Exception as e:
        print(f"Error making prediction from fighter names: {str(e)}")
        print("Please manually check fighter names in the dataset")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        # If running in prediction mode, just do a prediction
        predict_upcoming_fight()
    else:
        # Otherwise, train the model
        main()