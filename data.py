import psycopg2
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import requests
from bs4 import BeautifulSoup
import time

class Data:
    def __init__(self):
        self.conn = psycopg2.connect(database="legiscan_api",
                                host="localhost",
                                user="legiscan_api",
                                password="legiscan",
                                port="5432")
        self.cursor = self.conn.cursor()

    def get_people(self):
        self.cursor.execute("SELECT people_id, name from ls_people order by people_id;")
        people = pd.DataFrame(self.cursor.fetchall())
        people.to_pickle("people.pkl")
        return people

    def get_votes(self):
        """ get votes and description """
        self.cursor.execute("SELECT lb.description as text, string_agg(lp.people_id::text, ',') as label FROM ls_bill lb join ls_bill_vote lbv on lbv.bill_id=lb.bill_id join ls_bill_vote_detail lbvd on lbv.roll_call_id=lbvd.roll_call_id join ls_people lp on lbvd.people_id=lp.people_id where lbvd.vote_id = 1 group by lb.bill_id;")
        votes = pd.DataFrame(self.cursor.fetchall())
        votes = votes.rename(columns={0: "text", 1: "people"})
        votes["labels"] = None
        return votes


    def transpose_votes_to_labels(self, votes, people):
        for index, row in votes.iterrows():
            labels = []
            ids_voted = row["people"].split(',')
            for people_id in people[0]:
                labels.append(float(1) if str(people_id) in ids_voted else float(0))
            
            votes.at[index, "labels"] = labels

        votes = votes.fillna(0)
        votes = votes.drop("people", axis=1)
        
        print(votes)
        return votes

    def create_data_set(self, votes):
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

    def get_full_text_links(self):
        self.cursor.execute("select bill_id, state_url from ls_bill_text;")
        full_text_links = pd.DataFrame(self.cursor.fetchall())
        return full_text_links

    def fetch_full_text(self, links):
        for index, row in links.iterrows():
            bill_id = row[0]
            text_link = row[1]
            print(text_link)
            time.sleep(1.5)
            response = requests.get(text_link)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.find("div", class_="document").get_text().strip()

            with open('./full_text/'+str(bill_id)+'.txt', 'w') as output_file:
                output_file.write(text)

if __name__ == "__main__":
    data = Data()

    links = data.get_full_text_links()
    data.fetch_full_text(links)

    # people = data.get_people()
    # votes = data.get_votes()
    # votes_with_labels = data.transpose_votes_to_labels(votes, people)
    # data.create_data_set(votes_with_labels)

