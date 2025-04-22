
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ========== Step 1: Load and Clean MCI and Control Cohorts ==========

cutoff_date1 = pd.Timestamp('2023-11-20')
cutoff_date2 = pd.Timestamp('1901-01-01')

# Load MCI cohort
mci_person = pd.read_csv("mci_person.csv")
mci_person.drop(columns=['index_date'], inplace=True, errors='ignore')
mci_person['idx_date'] = pd.to_datetime(mci_person['idx_date'])
mci_person = mci_person[mci_person['idx_date'] <= cutoff_date1]
mci_person['group'] = 1  # MCI

# Load control cohort
pc_person = pd.read_csv("pc_person.csv")
pc_person.drop(columns=['index_date'], inplace=True, errors='ignore')
pc_person['idx_date'] = pd.to_datetime(pc_person['idx_date'])
pc_person = pc_person[pc_person['idx_date'] <= cutoff_date1]
pc_person['group'] = 0  # Control

combined_df = pd.concat([mci_person, pc_person], ignore_index=True)

# ========== Step 2: Propensity Score Matching (10:1) ==========

exclude_cols = ['person_id', 'idx_date', 'group']
features = [col for col in combined_df.columns if col not in exclude_cols]
combined_df = combined_df.dropna(subset=features)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(combined_df[features])

lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, combined_df['group'])
combined_df['propensity_score'] = lr.predict_proba(X_scaled)[:, 1]

treated = combined_df[combined_df['group'] == 1].copy()
control = combined_df[combined_df['group'] == 0].copy()

nn = NearestNeighbors(n_neighbors=10, metric='euclidean')
nn.fit(control[['propensity_score']])
distances, indices = nn.kneighbors(treated[['propensity_score']])
matched_controls = control.iloc[indices.flatten()].copy()
matched_controls['match_id'] = np.repeat(treated['person_id'].values, 10)

matched_df = pd.concat([treated, matched_controls], ignore_index=True)

# ========== Step 3: Load and Subset Additional Data ==========

person = pd.read_csv("person_10x_1yr.csv")
matched_ids = matched_df['person_id'].unique()
subset_person = person[person['person_id'].isin(matched_ids)].copy()

# Load and filter condition/problem occurrence files
mci_cond = pd.read_csv("mci_condition_occurrence.csv", encoding='latin1')
pc_cond = pd.read_csv("pc_condition_occurrence.csv", encoding='latin1')
mci_pl = pd.read_csv("mci_problem_occurrence.csv", encoding='latin1')
pc_pl = pd.read_csv("pc_problem_occurrence.csv", encoding='latin1')

pc_cond = pd.concat([pc_cond, pc_pl], axis=0)
mci_cond = pd.concat([mci_cond, mci_pl], axis=0)
condition = pd.concat([mci_cond, pc_cond, mci_pl, pc_pl], axis=0)
condition = condition[condition['person_id'].isin(person['person_id'])]
condition.to_csv("condition_10x.csv", index=False)

# Load and filter drug data
mci_drug = pd.read_csv("mci_drug_occurrence.csv", encoding='latin1')
pc_drug = pd.read_csv("pc_drug_occurrence.csv", encoding='latin1')
drug = pd.concat([mci_drug, pc_drug], axis=0)
drug = drug[drug['person_id'].isin(person['person_id'])]
drug.to_csv("drug_10x.csv", index=False)

# ========== Step 4: Final Processing Functions ==========

window_len = 0

def case_ctrl_id(person_data, case=1):
    result_id = person_data.loc[person_data['mci'] == case, 'person_id'].reset_index(drop=True)
    return result_id

def get_idx_date_case(case_data, window_length=window_len):
    case_data['idx_date'] = pd.to_datetime(case_data['idx_date'])
    case_data['idx_date'] -= pd.DateOffset(years=window_length)
    case_data['idx_date'] = case_data.apply(
        lambda row: row['idx_date'] - pd.DateOffset(days=366) - pd.DateOffset(years=window_length) + pd.DateOffset(years=1)
        if row['idx_date'].strftime('%m-%d') == '02-29' else row['idx_date'],
        axis=1
    )
    return case_data[['person_id', 'idx_date']]

# Done
print("Data process complete")
