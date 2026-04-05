import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = ['policy_number', 'insured_zip', '_c39', 'incident_location']
    df = df.drop(columns=cols_to_drop)
    df = df.replace('?', np.nan)
    for col in ['collision_type', 'property_damage', 'police_report_available']:
        df[col] = df[col].fillna(df[col].mode()[0])
    df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'], format='%d-%m-%Y')
    df['incident_date'] = pd.to_datetime(df['incident_date'], format='%d-%m-%Y')

    df['days_to_incident'] = (df['incident_date'] - df['policy_bind_date']).dt.days
    df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    severity_mapping = {
        'Trivial Damage': 0,
        'Minor Damage': 1,
        'Major Damage': 2,
        'Total Loss': 3
    }
    df['incident_severity'] = df['incident_severity'].map(severity_mapping)
    ohe_cols = [
        'policy_state', 'policy_csl', 'insured_sex',
        'insured_education_level', 'insured_occupation',
        'insured_hobbies', 'insured_relationship', 'incident_type',
        'collision_type', 'authorities_contacted',
        'incident_state', 'incident_city', 'property_damage',
        'police_report_available', 'auto_make', 'auto_model',
        'age_group', 'tenure_group'
    ]
    available_ohe_cols = [col for col in ohe_cols if col in df.columns]
    df = pd.get_dummies(df, columns=available_ohe_cols, drop_first=True)
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"One-hot encoding complete. New shape: {df.shape}")
    return df

def scale_numerical_features(df: pd.DataFrame, is_train=True, scaler=None) -> pd.DataFrame:
    df = df.copy()
    numerical_cols = [
        'months_as_customer', 'age', 'policy_annual_premium',
        'umbrella_limit', 'capital-gains', 'capital-loss',
        'incident_hour_of_the_day', 'number_of_vehicles_involved',
        'bodily_injuries', 'witnesses', 'total_claim_amount',
        'injury_claim', 'property_claim', 'vehicle_claim', 'days_to_incident'
    ]
    cols_to_scale = [col for col in numerical_cols if col in df.columns]
    if not cols_to_scale:
        if is_train:
            return df, StandardScaler()
        return df

    if is_train:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        return df, scaler

    if scaler is None:
        raise ValueError("`scaler` must be provided when `is_train=False`.")

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df
