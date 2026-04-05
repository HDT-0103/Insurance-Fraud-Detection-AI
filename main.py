from insurance_project.data_loader import load_data
from insurance_project.preprocessing import clean_data

if __name__ == "__main__":
    df = load_data("data/Automobile_insurance_fraud.csv")
    cleaned_df = clean_data(df)
    cleaned_df.to_csv("data/Automobile_insurance_fraud_cleaned.csv", index=False)
    print(cleaned_df.head())
