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

cursor.execute("SELECT lb.description as text, string_agg(lp.people_id::text, ',') as label FROM ls_bill lb join ls_bill_vote lbv on lbv.bill_id=lb.bill_id join ls_bill_vote_detail lbvd on lbv.roll_call_id=lbvd.roll_call_id join ls_people lp on lbvd.people_id=lp.people_id where lbvd.vote_id = 1 group by lb.bill_id;")
votes = pd.DataFrame(cursor.fetchall()).rename(columns={0: "text", 1: "label"}).sample(frac=1)

grouped = []
for g, df in votes.groupby(np.arange(len(votes)) // 1200):
    grouped.append(df)

train_data = DatasetDict({
    "train": Dataset.from_pandas(pd.DataFrame(data=grouped[0]).reset_index()),
    "test": Dataset.from_pandas(pd.DataFrame(data=grouped[1]).reset_index()),
    "unsupervised": Dataset.from_pandas(pd.DataFrame(data=grouped[2]).reset_index())
})
train_data.save_to_disk('train_data.dataset')

# cursor.execute("SELECT lp.people_id, lp.name FROM ls_people lp;")
# people = pd.DataFrame(cursor.fetchall())
# people.set_index(0).to_pickle('./people.pkl')

