import pandas as pd
import pickle
def single_format(user_prompt,model_answer):
    return f"<s>[INST] {user_prompt} [/INST] {model_answer} </s>"

df=pd.read_csv("Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
df=df[["instruction","response"]]
# print(df)
df['formatted'] = df.apply(lambda row: single_format(row['instruction'], row['response']), axis=1)
combined_string = '\n'.join(df['formatted'].tolist())
# print(combined_string )

with open("dataset_modified_for_fine_tuning.pkl", "wb") as file:
    pickle.dump(combined_string, file)

