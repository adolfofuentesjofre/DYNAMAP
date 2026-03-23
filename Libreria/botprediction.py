import itertools
import networkx as nx
import numpy as np
import os
import pandas as pd
import re
import readtime

# Creates empty dataframe with "uuid" column.
df_empty = pd.DataFrame(columns=["uuid"])

class BotPrediction:
    def __init__(self, DATASET="chile", threshold=10):
        # Reads votation dataset. Options are "chile", "colombia", "georgia", and "lebanon".
        path = os.path.join(os.path.dirname(__file__), f"../data/data_survey_dump_{DATASET}.csv")
        df = pd.read_csv(path, chunksize=10**5)
        df = pd.concat(df)

        # Converts the column type from string to datetime format, and sorts rows by datetime.
        df["datetime"] = pd.to_datetime(df["datetime"].str[0:19],  format="%Y-%m-%d %H:%M:%S")
        df = df.sort_values("datetime")

        # Threshold of minimum number of votes per uuid
        df_filtered = df[["id", "uuid"]]
        df_uuid_count = df_filtered.groupby(["uuid"]).agg({"id": "count"}).reset_index()
        df_uuid_count = df_uuid_count[df_uuid_count["id"] >= threshold]
        valid_uuid = df_uuid_count["uuid"].unique()
        df = df[df["uuid"].isin(valid_uuid)].copy()
        
        # Converts "option_a", "option_b", and "selected" columns to integer type.
        df = df[(df["option_a"].notna()) & (df["option_b"].notna())].copy().reset_index(drop=True)
        cols = ["option_a", "option_b", "selected"]
        df[cols] = df[cols].astype(int)
        
        # Generates "card_id"
        df = self.generate_card_id(df)
        
        # Saves data
        self.data = df
        self.DATASET = DATASET
        self.df_uuid_count = df_uuid_count
        
        # Creates empty layers and defines thresholds
        self.layers = {}
        self.THRESHOLD_LAYER_A = 3
        self.THRESHOLD_LAYER_B = 0.2
        self.THRESHOLD_LAYER_C = 20
        self.THRESHOLD_LAYER_D = 5
        self.THRESHOLD_LAYER_E = 0.95
        self.THRESHOLD_LAYER_F = 0.05

        # Log
        self.verbose = False

        # Words per minute
        self.WPM = 200
        
        # Defines descriptive dataframes
        self.df_bots = df_empty.copy()
        self.df_delta = df_empty.copy()
        self.df_n_ips = df_empty.copy()
        self.df_rate = df_empty.copy()
        self.df_score = df_empty.copy()
        self.df_statistics = df_empty.copy()
        self.df_transitivity = df_empty.copy()
        self.df_user_registered = df_empty.copy()
        self.df_uuid = df_empty.copy()
        self.df_uuid_unique = df_empty.copy()
        
        
    def find(self, uuid):
        """
        Find data by "uuid"
        """
        return self.data[self.data["uuid"] == uuid]


    def generate_card_id(self, df):
        """
        Generates a *card_id* considering each comparison pair.
        """

        a = df[["option_a", "option_b", "selected"]].values

        # Boolean variable, check if a/b was selected
        df["option_a_selected"] = np.where(a[:, 0] == a[:, 2], 1, 0)
        df["option_b_selected"] = np.where(a[:, 1] == a[:, 2], 1, 0)
        df["option_drawn_selected"] = np.where(a[:, 2] == 0, 1, 0)

        # Sorts options, always lower value on left column
        df["option_a_sorted"] = np.where(a[:, 0] < a[:, 1], a[:, 0], a[:, 1])
        df["option_b_sorted"] = np.where(a[:, 0] >= a[:, 1], a[:, 0], a[:, 1])

        # Generates a card id
        df["card_id"] = "1" + df["option_a_sorted"].astype(str).str.zfill(3) + df["option_b_sorted"].astype(str).str.zfill(3)
        df["card_id"] = df["card_id"].astype(int)

        # Boolean variable, check if a/b was selected
        df["option_source"] = np.where(a[:, 1] == a[:, 2], a[:, 0], a[:, 1])
        df["option_target"] = np.where(a[:, 0] == a[:, 2], a[:, 0], a[:, 1])

        # Creates option_source / option_target
        selected_zero = df["selected"] == 0
        df.loc[selected_zero, "option_source"] = np.nan
        df.loc[selected_zero, "option_target"] = np.nan

        return df


    def _layer_a(self):
        """
        Pairs of over-represented proposals.
        Calculates the number of times that an user voted by the same *card_id*. 
        Given the random nature of proposals, the expected value is 1. 
        If an user has a number of votes higher than a threshold in a proposal, he will be considered a bot.
        """
        df = self.data.copy()
        df_temp = df.groupby(["uuid", "card_id"]).count().reset_index().copy()
        
        THRESHOLD = self.THRESHOLD_LAYER_A
        df_temp = df_temp[df_temp["id"] > THRESHOLD].copy()
        bots_detected_a = df_temp["uuid"].unique()

        self.layers["layer_a"] = bots_detected_a

        if self.verbose:
            print(f"Layer A completed! {len(bots_detected_a)} bots detected.")


    def _layer_b(self):
        """
        Over-represented individual proposals.
        If any proposal was responded in a higher number of times. In the case the over-representation 
        of a proposal, I traced-down among uuid's that voted for that specific proposal, and I added to the bot-list.
        """
        df = self.data.copy()
        df_temp = df.copy()

        df_temp_1 = df_temp[["option_a", "uuid", "id"]]
        df_temp_1 = df_temp_1.rename(columns={"option_a": "option"})
        df_temp_2 = df_temp[["option_b", "uuid", "id"]]
        df_temp_2 = df_temp_2.rename(columns={"option_b": "option"})

        df_type_b = pd.concat([df_temp_1, df_temp_2])
        df_type_b = df_type_b.groupby(["uuid", "option"]).count()
        # Change: groupby state_office and divide by sum
        rate = df_type_b.groupby(level=0).apply(lambda x: x / float(x.sum()))
        df_type_b["rate"] = rate

        l = df_type_b.reset_index()

        THRESHOLD_VOTES = self.THRESHOLD_LAYER_B
        df_type_b = l[l["rate"] > THRESHOLD_VOTES]
        bots_detected_b = df_type_b["uuid"].unique()

        self.layer_b = bots_detected_b
        self.layers["layer_b"] = bots_detected_b

        if self.verbose:
            print(f"Layer B completed! {len(bots_detected_b)} bots detected.")


    def _layer_c(self):
        """
        Analyzing user's patterns, we calculated the mean of each user for each proposal' pair. 
        The expected value is 1, because the proposals was showed randomly to the user.
        """
        # Calculates the number of times that a user votes by the same card_id
        df = self.data.copy()
        THRESHOLD_MEAN = self.THRESHOLD_LAYER_C

        df_filtered = df[["card_id", "id", "uuid"]]
        df1 = df_filtered.groupby(["uuid", "card_id"]).agg({"id": "count"}).reset_index()
        df1["std"] = df1["id"].copy()
        df1["mean"] = df1["id"].copy()
        df2 = df1.groupby("uuid").agg({"mean": "mean", "std": "std"}).reset_index()
        
        self.df_statistics = df2

        # Generates dataframe with valid users by std
        df_mean = df2[df2["mean"] > THRESHOLD_MEAN]
        bots_detected_c = df_mean["uuid"].unique()

        self.layer_c = bots_detected_c
        self.layers["layer_c"] = bots_detected_c

        if self.verbose:
            print(f"Layer C completed! {len(bots_detected_c)} bots detected.")


    def _layer_d(self):
        """
        Over-represented by standard deviation
        Analogously to C, we calculated the standard deviation of each user for each proposal. The expected value is 0, 
        because the proposals was showed randomly to the user. If an user showed a std higher than 5, I added to my bot-list
        """
        # Calculates the number of times that a user votes by the same card_id
        df = self.data.copy()
        THRESHOLD_STD = self.THRESHOLD_LAYER_D

        df_filtered = df[["card_id", "id", "uuid"]]
        df1 = df_filtered.groupby(["uuid", "card_id"]).agg({"id": "count"}).reset_index()
        df1["std"] = df1["id"].copy()
        df1["mean"] = df1["id"].copy()
        df2 = df1.groupby("uuid").agg({"mean": "mean", "std": "std"}).reset_index()

        # Generates dataframe with valid users by std
        df_std = df2[df2["std"] > THRESHOLD_STD]
        bots_detected_d = list(df_std["uuid"].unique()) + list(df2[df2["std"].isna()]["uuid"].unique())

        self.layer_d = bots_detected_d
        self.layers["layer_d"] = bots_detected_d

        if self.verbose:
            print(f"Layer D completed! {len(bots_detected_d)} bots detected.")


    def _layer_e(self):
        """
        Random bots.
        This layer analyzes bots that voted all the time by the same proposal (a / b)
        """
        df = self.data.copy()
        # Checks the percentage of questions that user read
        item = {
            "id": "count",
            "option_a_selected": "sum",
            "option_b_selected": "sum",
            "option_drawn_selected": "sum"
        }
        a = df.groupby(["uuid"]).agg(item).reset_index().rename(columns={"id": "count"})

        b = a.groupby(["uuid"]).sum().reset_index()
        b["rate_a_selected"] = b["option_a_selected"] / b["count"]
        b["rate_b_selected"] = b["option_b_selected"] / b["count"]
        b["rate_drawn_selected"] = b["option_drawn_selected"] / b["count"]
        

        THRESHOLD = self.THRESHOLD_LAYER_E
        b = b[(b["rate_a_selected"] >= THRESHOLD) | (b["rate_b_selected"] >= THRESHOLD) | (b["rate_drawn_selected"] >= THRESHOLD)]

        bots_detected_e = b["uuid"].unique()

        self.layers["layer_e"] = bots_detected_e

        if self.verbose:
            print(f"Layer E completed! {len(bots_detected_e)} bots detected.")
        
        
    def _layer_f(self):
        """
        Rate of no-read proposals.
        """
        # Creates dataframe based on Words per minute
        path_labels = os.path.join(os.path.dirname(__file__), f"../data/labels/{self.DATASET}.tsv")
        df_labels = pd.read_csv(path_labels, delimiter="\t")
        df_labels = df_labels[["id", "name"]]
        df_labels = df_labels.rename(columns={"id": "proposal_id"})

        wpm = self.WPM
        wpm_label = f"read_time_{wpm}_wpm"
        wpm_label_x = f"{wpm_label}_x"
        wpm_label_y = f"{wpm_label}_y"

        df_labels[wpm_label] = df_labels["name"].apply(lambda x: readtime.of_text(x, wpm=wpm).seconds)

        df = self.data.copy()
        df_wpm = df[["option_a", "option_b", "id", "uuid", "datetime", "option_a_selected", "option_b_selected"]]
        labels = df_labels[["proposal_id", wpm_label]]

        df_wpm = df_wpm.merge(labels, left_on="option_a", right_on="proposal_id")
        df_wpm = df_wpm.merge(labels, left_on="option_b", right_on="proposal_id")

        df_wpm["minimal_time_wpm"] = df_wpm[wpm_label_x] + df_wpm[wpm_label_y]

        df_wpm = df_wpm.drop(columns=["proposal_id_x", wpm_label_x, "proposal_id_y", wpm_label_y])

        df_wpm["diff"] = df_wpm.sort_values(["uuid", "datetime"]).groupby("uuid")["datetime"].diff()
        df_wpm["diff_seconds"] = df_wpm["diff"].apply(lambda x: x.seconds).astype(float)
        
        # Removes nan columns related with the first vote of an user
        df_wpm = df_wpm.dropna().copy()

        item = {
            "id": "count",
            "minimal_time_wpm": np.mean, 
            "diff_seconds": np.mean, 
            "option_a_selected": "sum", 
            "option_b_selected": "sum"
        }

        # Checks the percentage of questions that user read
        a = df_wpm.groupby(["uuid", "id"]).agg(item).rename(columns={"id": "count"}).reset_index()
        a["delta"] = a["diff_seconds"] > a["minimal_time_wpm"]

        b = a.groupby(["uuid"]).sum().reset_index()
        b["rate"] = b["delta"] / b["count"]

        THRESHOLD_USER_READ = self.THRESHOLD_LAYER_F

        user_read_bots = b[b["rate"] < THRESHOLD_USER_READ]

        df_type_f = b[["uuid", "count", "rate"]]
        self.df_rate = df_type_f

        bots_detected_f = user_read_bots["uuid"].unique()

        self.layer_f = bots_detected_f
        self.layers["layer_f"] = bots_detected_f

        if self.verbose:
            print(f"Layer F completed! {len(bots_detected_f)} bots detected.")
        
        
    def _layer_g(self):
        """
        Six-sigma validation.
        """
        df_uuid_count = self.df_uuid_count.copy()
        
        df_uuid_count = df_uuid_count.rename(columns={"id": "votes"})

        median = df_uuid_count["votes"].median()
        mean = df_uuid_count["votes"].mean()
        sigma = df_uuid_count["votes"].std()

        ucl = mean + 6 * sigma
        lcl = mean - 6 * sigma
        if (lcl < 0):
            lcl = 0

        df_uuid_count["in_control"] = df_uuid_count["votes"].apply(lambda x: x < ucl)
        bots_detected_g = df_uuid_count[~df_uuid_count["in_control"]]["uuid"].unique()

        self.layer_g = bots_detected_g
        self.layers["layer_g"] = bots_detected_g

        if self.verbose:
            print(f"Layer G completed! {len(bots_detected_g)} bots detected.")
        
        
    def _layer_h(self):
        """
        Bots with wrong uuid format.
        """
        df_uuid_count = self.df_uuid_count.copy()
        r = re.compile(r"([a-z0-9]{8}\-[a-z0-9]{4}\-[a-z0-9]{4}\-[a-z0-9]{4}\-[a-z0-9]{12})")
        bots_detected_h = df_uuid_count[~df_uuid_count["uuid"].apply(lambda x: bool(r.match(x)))]["uuid"].unique()

        self.layers["layer_h"] = bots_detected_h

        if self.verbose:
            print(f"Layer H completed! {len(bots_detected_h)} bots detected.")
        
        
    def _generate_bots(self):
        """
        Generates a DataFrame including bots detected on each previously executed layers.
        """
        bot_layers = self.layers

        bots = list(dict.fromkeys(list(itertools.chain(*bot_layers.values()))))
        
        df_bots = pd.DataFrame(bots, columns=["uuid"])
        df_bots["is_bot"] = 1

        self.df_bots = df_bots


    def _user_registered(self):
        """
        User registered on the platform.
        """
        DATASET = self.DATASET

        path = os.path.join(os.path.dirname(__file__), f"../data/data_people_dump_{DATASET}.csv")
        df_people = pd.read_csv(path, chunksize=10**5)
        df_people = pd.concat(df_people)

        # Converts datetime columns to datetime format
        df_people["datetime"] = pd.to_datetime(df_people["datetime"].str[0:19],  format="%Y-%m-%d %H:%M:%S")
        # Keeps latest information registered by each user
        df_people = df_people.loc[df_people.groupby("uuid").datetime.idxmax()]

        df_people = df_people[["uuid"]].copy()
        df_people["registered"] = 1

        self.df_user_registered = df_people.copy()
        
        
    def _transitivity(self):
        """
        Calculates transitivity of each user.
        """
        df = self.data.copy()

        df_transitivity = []
        df_transitivity_copy = df.copy()

        for df_uuid_filtered in df_transitivity_copy.groupby("uuid"):
            df_temp = df_uuid_filtered[1]
            df_temp = df_temp[df_temp["selected"] != 0].copy()
            G = nx.from_pandas_edgelist(df_temp, "option_source", "option_target", create_using=nx.DiGraph)
            item = {
                "uuid": df_uuid_filtered[0],
                "transitivity": nx.transitivity(G)
            }
            df_transitivity.append(item)

        df_transitivity = pd.DataFrame(df_transitivity)
        
        self.df_transitivity = df_transitivity.copy()


    def _scoring(self):
        """
        Calcula el promedio de score por usuario
        """
        df = self.data.copy()

        df_score = df[["uuid", "score"]].groupby("uuid").mean().reset_index()
        self.df_score = df_score.copy()

    def _delta(self):
        df = self.data.copy()
        df_delta = []
        for i in df.groupby("uuid"):
            df_temp = i[1]
            
            start = df_temp["datetime"].min()
            finish = df_temp["datetime"].max()
            delta = (finish - start).seconds
            
            item = {
                "uuid": i[0],
                "delta": delta
            }
            df_delta.append(item)
            
        df_delta = pd.DataFrame(df_delta)
        self.df_delta = df_delta.copy()

    
    def _unique_uuid(self):
        """
        Calculates the number of uuids that voted on the same IP address.
        """
        df = self.data.copy()

        # Cantidad de usuarios que tienen la misma IP
        df_ip_hash = df.groupby("ip_hash").agg({"uuid": "nunique"}).reset_index().rename(columns={"uuid": "unique_uuid"})
        
        # Agrupaciones usuario-IP
        df_uuid_ip_hash = df[["uuid", "ip_hash"]].groupby(["uuid", "ip_hash"]).count().reset_index()
        # usuario | ip | n usuarios
        df_uuid_unique = df_uuid_ip_hash.merge(df_ip_hash, on="ip_hash").drop(columns=["ip_hash"])

        # usuario | sum(n usuarios)
        df_uuid_unique = df_uuid_unique.groupby("uuid").sum().reset_index()
        self.df_uuid_unique = df_uuid_unique


    def _generate_n_ips(self):
        # Calcula la cantidad de IP diferentes por cada usuario
        df = self.data.copy()
        df_n_ips = df.groupby("uuid").agg({"ip_hash": "nunique"}).reset_index().rename(columns={"ip_hash": "n_ips"})
        self.df_n_ips = df_n_ips


    def _generate_uuid(self):
        # Merges dataframe of mean/std with user registered
        df_uuid = pd.merge(self.df_statistics, self.df_user_registered, on="uuid", how="left")
        # Merges current df_uuid with bots
        df_uuid = pd.merge(df_uuid, self.df_bots, on="uuid", how="left")
        # Merges current df_uuid with table of transitivies
        df_uuid = pd.merge(df_uuid, self.df_transitivity, on="uuid", how="left")
        # Merges with count/rate of response
        df_uuid = pd.merge(df_uuid, self.df_rate, on="uuid", how="left")
        # Merges with count/rate of response
        df_uuid = pd.merge(df_uuid, self.df_score, on="uuid", how="left")
        # Merges with number of shared
        df_uuid = df_uuid.merge(self.df_uuid_unique, on="uuid", how="left")
         # Merges with n_ips
        df_uuid = pd.merge(df_uuid, self.df_n_ips, on="uuid", how="left")

        for col in ["registered", "is_bot"]:
            df_uuid[col] = df_uuid[col].fillna(0)
               
        ## Converts NaN in scoring 
        df_uuid["score_pred"] = df_uuid["score"].apply(lambda x: ~np.isnan(x)).astype(int)

        def input_scoring(x):
            if np.isnan(x["score"]):
                score = 0.9
                if x["count"] > 1000 or x["rate"] < 0.1:
                    score = 0.1
                return score
            return x["score"]

        df_uuid["score"] = df_uuid.apply(lambda x: input_scoring(x), axis=1)

        # Merges with delta of response
        df_uuid = pd.merge(df_uuid, self.df_delta, on="uuid", how="left")

        self.df_uuid = df_uuid
        
    def _load_step(self):
        self._scoring()
        self._transitivity()
        self._user_registered()
        self._generate_bots()
        self._unique_uuid()
        self._delta()
        self._generate_n_ips()
        self._generate_uuid()