import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import logging
import streamlit as st
import plotly.express as px

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

base_dirs = ['dataset_with_normal_noise', 'dataset_with_uniform_noise', 'dataset_with_poisson_noise']

models = {}

for base_dir in base_dirs:
    logging.info(f"Processing directory: {base_dir}")
    X = []
    y = []
    
    for movement in os.listdir(base_dir):
        movement_path = os.path.join(base_dir, movement)
        if os.path.isdir(movement_path):
            for participant in os.listdir(movement_path):
                participant_path = os.path.join(movement_path, participant)
                if os.path.isdir(participant_path):
                    for session in os.listdir(participant_path):
                        session_path = os.path.join(participant_path, session)
                        if os.path.isdir(session_path):
                            for file in os.listdir(session_path):
                                if file.endswith('.csv'):
                                    file_path = os.path.join(session_path, file)
                                    df = pd.read_csv(file_path)
                                    X.append(df.drop(columns=['id']).values)
                                    y.append(df['id'].values)
    
    X = [item for sublist in X for item in sublist]
    y = [item for sublist in y for item in sublist]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_clf = grid_search.best_estimator_
    logging.info(f"Best parameters for {base_dir}: {grid_search.best_params_}")
    
    # Cross-validation
    cv_scores = cross_val_score(best_clf, X_train, y_train, cv=5)
    logging.info(f"Cross-validation scores for {base_dir}: {cv_scores}")
    
    # Train and evaluate
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy for {base_dir}: {accuracy}")
    logging.info(f"Classification report for {base_dir}:\n{classification_report(y_test, y_pred)}")
    
    models[base_dir] = {'model': best_clf, 'accuracy': accuracy}

# Streamlit dashboard
st.title("Model Performance Dashboard")

accuracies = {base_dir: models[base_dir]['accuracy'] for base_dir in models}

df_accuracies = pd.DataFrame(list(accuracies.items()), columns=['Dataset', 'Accuracy'])

fig = px.bar(df_accuracies, x='Dataset', y='Accuracy', title='Model Accuracy Comparison')

st.plotly_chart(fig)

for base_dir in models:
    st.write(f"Dataset: {base_dir}")
    st.write(f"Accuracy: {models[base_dir]['accuracy']:.2f}")
