import psycopg2
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict

conn = psycopg2.connect(database="legiscan_api",
                        host="localhost",
                        user="legiscan_api",
                        password="legiscan",
                        port="5432")
cursor = conn.cursor()


# get people
cursor.execute("SELECT people_id, name from ls_people order by people_id;")
people = pd.DataFrame(cursor.fetchall())
people.to_pickle("people.pkl")

# get votes and description
cursor.execute("SELECT lb.description as text, string_agg(lp.people_id::text, ',') as label FROM ls_bill lb join ls_bill_vote lbv on lbv.bill_id=lb.bill_id join ls_bill_vote_detail lbvd on lbv.roll_call_id=lbvd.roll_call_id join ls_people lp on lbvd.people_id=lp.people_id where lbvd.vote_id = 1 group by lb.bill_id;")
votes = pd.DataFrame(cursor.fetchall())
votes = votes.rename(columns={0: "text", 1: "people"})
votes["labels"] = None

# populate with votes
for index, row in votes.iterrows():
    labels = []
    ids_voted = row["people"].split(',')
    for people_id in people[0]:
        labels.append(float(1) if str(people_id) in ids_voted else float(0))
    
    votes.at[index, "labels"] = labels

votes = votes.fillna(0)
votes = votes.drop("people", axis=1)

print(votes)

# sort by random and split into groups for training
votes = votes.sample(frac=1)

grouped = []
for g, df in votes.groupby(np.arange(len(votes)) // 1200):
    grouped.append(df)

data_set = DatasetDict({
    "train": Dataset.from_pandas(pd.DataFrame(data=grouped[0]).reset_index(drop=True)),
    "test": Dataset.from_pandas(pd.DataFrame(data=grouped[1]).reset_index(drop=True)),
    "unsupervised": Dataset.from_pandas(pd.DataFrame(data=grouped[2]).reset_index(drop=True))
})

print(data_set)

data_set.save_to_disk('votes')

