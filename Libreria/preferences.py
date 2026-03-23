from random import choice
from .ranking.trueskill import trueskill
import networkx as nx
import numpy as np
import os
import pandas as pd
import seaborn as sns


def flip_coin():
    return choice(["head", "tail"])


class Preferences:
    def __init__(self, DATASET="chile", THRESHOLD_MIN_USER=10, DIFFERENTIAL_PRIVACY=False, verbose=False):
        #Lectura del archivo con la información de las votaciones
        path = os.path.join(os.path.dirname(__file__), f"../data/data_survey_dump_{DATASET}.csv")
        #  chunksize se refiere a cuántas filas por segundo pandas leen de un archivo
        data = pd.read_csv(path, chunksize=10**5)
        data = pd.concat(data)

        # Converts datetime column to datetime format
        data["datetime"] = pd.to_datetime(data["datetime"].str[0:19], format="%Y-%m-%d %H:%M:%S")
        data = data.sort_values("datetime")

        # Removes NaN values.
        data = data[(data["option_a"].notna()) & (data["option_b"].notna())].reset_index(drop=True)
        cols = ["option_a", "option_b", "selected"]
        data[cols] = data[cols].astype(int)

        # Creates drawn column.
        data["drawn"] = data["selected"].apply(lambda x: x == 0)

        def diff_privacy(x):
            # Funcion que define de forma aleatoria la selelección de la opcion a y la opcion b
            if x["selected"] != 0:
                coin = flip_coin()
                if coin == "tail":
                    coin = flip_coin()
                    if coin == "tail":
                        return x["option_b"] if x["option_a"] == x["selected"] else x["option_a"]

            return x["selected"]

        if DIFFERENTIAL_PRIVACY:
            data["selected"] = data.apply(lambda x: diff_privacy(x), axis=1)

        # Reads list of labels of each proposal.
        path_labels = os.path.join(os.path.dirname(__file__), f"../data/labels/{DATASET}.tsv")
        LABELS = pd.read_csv(path_labels, delimiter="\t")
        
        # Reads predicted_prob of bot for each uuid.
        path_bots = os.path.join(os.path.dirname(__file__), f"../Chile Todos los Ciclos/results/bot_prediction/{DATASET}_uuid.csv")
        BOT_LIST = pd.read_csv(path_bots, chunksize=10**5)
        BOT_LIST = pd.concat(BOT_LIST)
        
        # Reads list of users registered on the platform.
        path_people = os.path.join(os.path.dirname(__file__), f"../data/data_people_dump_{DATASET}.csv")
        USERS = pd.read_csv(path_people, chunksize=10**5)
        USERS = pd.concat(USERS)

        self.BOT_LIST = BOT_LIST
        self.DATASET = DATASET
        self.FILTER = True
        self.LABELS = LABELS
        self.THRESHOLD_MIN_USER = THRESHOLD_MIN_USER
        self.USERS = self.users_step(USERS, verbose) # Se filtran usuarios que no cumplen ciertos criterios
        self.data = data
        self.data_filtered = data
        self.data_processed = data
        self.predicted_prob = 0.5
        self.verbose = verbose

    # TODO
    def users_step(self, USERS, verbose):
        """
        ETL of users registered on the platform. Filtra a los clientes segun varios criterios.

        Parameters:
            USERS (DataFrame): List of users registered on the platform.

        Returns:
            USERS: (DataFrame)
        """
        def log(df, desc=""):
            """Returns a log with steps."""
            if verbose:
                print(desc, f"{df.shape[0]} users.")

        log(USERS, "Original dataset")

        USERS["datetime"] = pd.to_datetime(USERS["datetime"].str[0:19], format="%Y-%m-%d %H:%M:%S")
        # Se deja solo el ultimo registro de un usuario. Elimina duplicados
        USERS = USERS.loc[USERS.groupby("uuid").datetime.idxmax()]

        log(USERS, "Keeps latest register by uuid")

        #Se filtra a los usarios que sean menoy o mayo a 18 y 80 años, respectivamente
        min_age = 18
        max_age = 80
        USERS = USERS[(USERS["age"] >= min_age) & (USERS["age"] < max_age)]

        log(USERS, f"Keeps age range [{min_age}, {max_age}[")

        bins = [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        # Se define el rango de edad de acuerdo al bins más cercano
        USERS["age_range"] = pd.cut(USERS.age, bins, include_lowest=True)

        return USERS

    def threshold_uuid(self, df):
        """
        Removes users with less than <THRESHOLD_MIN_USER> votes on the platform.

        Parameters:
            df (DataFrame)
        """
        df_temp = df[["id", "uuid"]]
        df_temp = df_temp.groupby(["uuid"]).agg({"id": "count"}).reset_index()
        df_temp = df_temp[df_temp["id"] >= self.THRESHOLD_MIN_USER]
        valid_uuid = df_temp["uuid"].unique()

        return df[df["uuid"].isin(valid_uuid)].copy()
    
    def card_id_step(self, df):
        """
        Creates a new column called "card_id".
        Format: 1 + <proposal_id> + <proposal_id>.
        Each <proposal_id> includes 3-digits.

        Parameters:
            df (DataFrame)
        """
        a = df[["option_a", "option_b", "selected"]].values

        # Boolean variable, check if a/b was selected
        df["option_a_selected"] = np.where(a[:, 0] == a[:, 2], 1, 0)
        df["option_b_selected"] = np.where(a[:, 1] == a[:, 2], 1, 0)

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
        df.loc[selected_zero, "option_source"] = df.loc[selected_zero, "option_a"]
        df.loc[selected_zero, "option_target"] = df.loc[selected_zero, "option_b"]
        
        return df
    
    def latest_step(self, df):
        """
        Includes a new column called "latest", that allows to filter the dataset by latest valid vote, based on card_id.

        Parameters:
            df (DataFrame)
        """
        # Sorts by datetime clicks done on dataset
        df = df.sort_values(["id", "datetime"])

        u = df[["uuid", "card_id", "datetime", "id"]]
        u = u.groupby(["datetime", "uuid", "card_id", "id"]).count().reset_index()
        u = u.drop_duplicates(["uuid", "card_id"], keep="last").drop(columns=["datetime", "uuid", "card_id"])
        u["latest"] = 1

        df = pd.merge(df, u, on="id", how="left")
        df.loc[df["latest"].isna(), "latest"] = 0

        df["latest"] = df["latest"].astype(int)

        return df


    def filter_by_cycle(self, df, cycle):
        """
        Filters data by wave of voting.
        Only valid in the case of Chilecracia.
        """
        path = os.path.join(os.path.dirname(__file__), f"chile_cycles.csv")
        df_cycles = pd.read_csv(path)
        # Fechas menores al rango definido [X, Y[
        cycle_range = {
            "cycle_1": ["2019-10-24 11:35:05", "2019-10-31 01:35:54"],
            "cycle_2": ["2019-10-31 01:35:54", "2019-11-07 00:55:15"],
            "cycle_3": ["2019-11-07 00:55:15", "2019-11-14 13:10:08"],
            "cycle_4": ["2019-11-14 13:10:08", "2019-11-28 15:21:02"],
            "cycle_5": ["2019-11-28 15:21:02", "2099-12-31 23:59:59"]
        }
        selected = cycle_range[cycle] or cycle_range["cycle_1"]

        ids = df_cycles[df_cycles[cycle] == 1]["id"].unique()
        return df[(df["option_a"].isin(ids)) & (df["option_b"].isin(ids))\
                  & (df["datetime"] >= pd.Timestamp(selected[0])) & (df["datetime"] < pd.Timestamp(selected[1]))]
        
    def transform_step(self, cycle="cycle_1", bot_prob=True):
        df = self.data
        # Removes uuid's with less than N votes on the platform.
        df = self.threshold_uuid(df)
        # Enrichs column list, including card_id, source/targe, option_sorted, among others.
        df = self.card_id_step(df)
        # Includes column called latest, that "removes" power to the bots.
        df = self.latest_step(df)
        
        # Merges data with bot list
        if bot_prob:
            df = pd.merge(df, self.BOT_LIST, on="uuid")

        # Updates 
        self.data_processed = df

        # If you're using chile as dataset, you'll need to filter by participation cycle
        if self.DATASET == "chile" and cycle != "all":
            df = self.filter_by_cycle(df, cycle)
        # Saves on data the current dataframe
        if self.FILTER:
            df = df[df["latest"] == 1].copy()

        self.data_filtered = df
        return self


    def get_bots_step(self):
        """
        Gets two dataframes: real users and bots
        """
        df = self.data_filtered.copy()

        df_no_bots = df[df["predicted_prob"] <= self.predicted_prob]
        df_bots = df[df["predicted_prob"] > self.predicted_prob]
        
        return df_no_bots, df_bots


    def get_political_step(self):
        """
        Gets two dataframes based on political orientation informed by the user: left and right
        Available for chile and colombia
        """
        df = self.data_filtered.copy()
        USERS = self.USERS

        uuid_left = USERS[USERS["politica"] < 5]["uuid"].unique()
        uuid_right = USERS[USERS["politica"] > 5]["uuid"].unique()

        predicted_prob = df["predicted_prob"] <= self.predicted_prob
        df_left = df[(predicted_prob) & (df["uuid"].isin(uuid_left))]
        df_right = df[(predicted_prob) & (df["uuid"].isin(uuid_right))]
        
        return df_left, df_right


    def filter_by(self, cut=None):
        """
        Gets a filtered dataframe. Options accepted are Politica, Sex, and Threshold.
        Example: Politica:left,Sex:female,Threshold:0.5
        @params {string} cut
        @returns DataFrame
        """
        df = self.data_filtered.copy()
        USERS = self.USERS.copy()

        if cut:
            def yn(x):
                value = x and x in ["true", "1", True]
                return value or x.lower()

            cut_params = dict([item.split(":") for item in cut.split(",")])
            cut_params = {k.capitalize():yn(cut_params[k]) for k in cut_params}
            
            if cut_params.get("Sex"):
                value = cut_params.get("Sex")
                filter_sex = []
                if value in ["female", "femenino"]:
                    filter_sex.extend(["Female", "Femenino"])
                if value in ["male", "masculino"]:
                    filter_sex.extend(["Male", "Masculino"])

                uuid = USERS[USERS["sex"].isin(filter_sex)]["uuid"].unique()
                df = df[df["uuid"].isin(uuid)]

            if cut_params.get("Politica"):
                uuids = []
                value = cut_params.get("Politica")
                if value == "left":
                    uuid_left = list(USERS[USERS["politica"] < 5]["uuid"].unique())
                    uuids.extend(uuid_left)
                if value == "center":
                    uuid_center = list(USERS[USERS["politica"] == 5]["uuid"].unique())
                    uuids.extend(uuid_center)
                if value == "right":
                    uuid_right = list(USERS[USERS["politica"] > 5]["uuid"].unique())
                    uuids.extend(uuid_right)
                
                uuid = USERS[USERS["uuid"].isin(uuids)]["uuid"].unique()
                
                df = df[df["uuid"].isin(uuid)]

            if cut_params.get("Threshold"):
                value = cut_params.get("Threshold")
                threshold = float(value.replace("!", ""))
                
                df_filter = df["predicted_prob"] > threshold if value[0] == "!" else df["predicted_prob"] <= threshold

                df = df[df_filter]


            if cut_params.get("Bot"):
                print("Bot", cut_params.get("Bot"))
                value = int(cut_params.get("Bot"))
                df_filter = df["prediction"] == value
                df = df[df_filter]


            if cut_params.get("BotMean"):
                print("BotMean", cut_params.get("BotMean"))
                value = int(cut_params.get("BotMean"))
                df_filter = df["mean_prediction"] == value
                df = df[df_filter]

        return df


    def get_secard_id_step(self):
        df = self.data_filtered.copy()
        USERS = self.USERS

        uuid_female = USERS[USERS["sex"].isin(["Female", "Femenino"])]["uuid"].unique()
        uuid_male = USERS[USERS["sex"].isin(["Male", "Masculino"])]["uuid"].unique()

        predicted_prob = df["predicted_prob"] <= self.predicted_prob
        df_female = df[(predicted_prob) & (df["uuid"].isin(uuid_female))]
        df_male = df[(predicted_prob) & (df["uuid"].isin(uuid_male))]
        
        return df_female, df_male


    def generate_ts_space(self, df1, df2):
        """
        Creates a dataframe for being used on scatter plot of preferences. It join two ranking's dataframes with the same id
        """
        LABELS = self.LABELS
        votes1 = pd.merge(df1, LABELS, on="id")
        votes1.columns = [str(col) + "_1" for col in votes1.columns]
        votes2 = pd.merge(df2, LABELS, on="id")
        votes2.columns = [str(col) + "_2" for col in votes2.columns]
        
        return pd.merge(votes1, votes2, left_on="id_1", right_on="id_2")


    def number_preferences(self, df):
        """
        Generates the number of preferences of each dataset.
        """
        n_options = len(set(df["option_a"]) | set(df["option_b"])) # Number of total options.
        n_users = len(set(df["uuid"])) # Number of users

        return n_options, n_users