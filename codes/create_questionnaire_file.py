import pandas as pd

# Load the files
q_df = pd.read_csv("/Users/rhea/Desktop/pads-parkinsons-disease-smartwatch-dataset-1.0.0 2/our files/questionnaire_data.csv")
p_df = pd.read_csv("/Users/rhea/Desktop/pads-parkinsons-disease-smartwatch-dataset-1.0.0 2/our files/patient_data.csv")

# Merge on subject_id and id
merged_df = q_df.merge(p_df, how='inner', left_on='subject_id', right_on='id')

# Create target column
merged_df["target"] = merged_df["condition"].apply(
    lambda x: 1 if x.strip().lower() == "parkinson's" else 0)

# Drop unnecessary columns
drop_cols = ['subject_id', 'study_id_x', 'questionnaire_name', 'resource_type', 'id',
             'study_id_y', 'condition', 'disease_comment', 'age_at_diagnosis',
             'age', 'height', 'weight', 'gender', 'handedness',
             'appearance_in_kinship', 'appearance_in_first_grade_kinship',
             'effect_of_alcohol_on_tremor']
clean_df = merged_df.drop(columns=drop_cols)

# Save to CSV
clean_df.to_csv("/Users/rhea/Desktop/pads-parkinsons-disease-smartwatch-dataset-1.0.0 2/our files/clean_questionnaire_data.csv", index=False)
