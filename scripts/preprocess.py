# preprocess.py: Loads raw CSV, cleans data, encodes categorical variables, scales features, splits into train/test sets, and saves processed data and scaler for reuse.
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path)
    drop_cols = [
        'CLIENTNUM',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon',
        'Naive_Bayes_Classifier_Attrition_Flag_Income_Category_Age'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    # Encode target variable
    df['Churn'] = df['Attrition_Flag'].apply(lambda x: 1 if x == 'Attrited Customer' else 0)
    df.drop('Attrition_Flag', axis=1, inplace=True)
    return df

def preprocess_features(df):
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, X.columns.tolist()

def main():
    df = load_and_clean_data('data/BankChurners.csv')
    X_scaled, y, scaler, feature_columns = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Save processed data and scaler
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)

    joblib.dump(scaler, 'models/scaler.save')
    joblib.dump(feature_columns, 'models/train_features.save')

    print("Preprocessing done and files saved.")

if __name__ == '__main__':
    main()
