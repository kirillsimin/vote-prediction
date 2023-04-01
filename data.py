import psycopg2
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
import requests
from bs4 import BeautifulSoup
import os
import time

class Data:
    def __init__(self):
        self.conn = psycopg2.connect(database="legiscan_api",
                                host="localhost",
                                user="legiscan_api",
                                password="legiscan",
                                port="5432")
        self.cursor = self.conn.cursor()
        self.wait = 0

    def get_people(self):
        self.cursor.execute("SELECT people_id, name from ls_people order by people_id;")
        people = pd.DataFrame(self.cursor.fetchall())
        people.to_pickle("people.pkl")
        return people

    def get_votes(self):
        """ get votes and description """
        self.cursor.execute("SELECT lb.bill_id as bill_id, string_agg(lp.people_id::text, ',') as label FROM ls_bill lb join ls_bill_vote lbv on lbv.bill_id=lb.bill_id join ls_bill_vote_detail lbvd on lbv.roll_call_id=lbvd.roll_call_id join ls_people lp on lbvd.people_id=lp.people_id where lbvd.vote_id = 1 group by lb.bill_id;")
        votes = pd.DataFrame(self.cursor.fetchall())
        votes = votes.rename(columns={0: "bill_id", 1: "people"})
        votes["labels"] = None
        return votes


    def transpose_votes_to_labels(self, votes, people):
        print(votes)
        for index, row in votes.iterrows():
            full_text = ""
            try:
                file_name = './full_text/'+str(row[0])+'.txt'
                with open(file_name, 'r') as file:
                    full_text = file.read()
            except FileNotFoundError:
                print(str(row[0]) + " not found.")

            labels = []
            ids_voted = row["people"].split(',')
            for people_id in people[0]:
                labels.append(float(1) if str(people_id) in ids_voted else float(0))
            
            votes.at[index, "labels"] = labels
            votes.at[index, "text"] = full_text

        votes = votes.fillna(0)
        votes = votes.drop("people", axis=1)
        votes = votes.drop("bill_id", axis=1)
        votes = votes[["text", "labels"]]
        
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
            self.fetch_one_bill_text(row)

    def fetch_one_bill_text(self, row):
            bill_id = row[0]
            text_link = row[1]
            file_name = './full_text/'+str(bill_id)+'.txt'
            print(text_link)
            if os.path.isfile(file_name):
                return

            try:
                time.sleep(self.wait)
                response = requests.get(text_link)
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.find("body").get_text().strip()

                with open(file_name, 'w') as output_file:
                    output_file.write(text)
                
                self.wait = 0
            except requests.exceptions.ConnectionError:
                if self.wait > 63:
                    print("Request ERROR, wait is more than 63")
                    raise Exception

                if self.wait == 0:
                    self.wait = 1
                else:
                    self.wait = self.wait * 2
                print("Request ERROR; sleeping for " + str(self.wait))
                self.fetch_one_bill_text(row)



if __name__ == "__main__":
    data = Data()

    links = data.get_full_text_links()
    data.fetch_full_text(links)

    people = data.get_people()
    votes = data.get_votes()
    votes_with_labels = data.transpose_votes_to_labels(votes, people)
    data.create_data_set(votes_with_labels)

