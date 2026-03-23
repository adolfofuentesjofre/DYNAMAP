import numpy as np
import pandas as pd
import json
from trueskill import Rating, rate_1vs1

# TRUESKILL ALGORITHM #
#######################

# Input: dataframe with 3 columns: option_a, option_b, selected
def trueskill(data):
    df = data[["option_a", "option_b", "selected"]].copy()
    df = df.dropna()
    for col in df.columns:
        df[col] = df[col].astype("int")
    
    # Get IDs and initialize values
    all_ids = sorted(list(dict.fromkeys([i for i in df["option_a"].unique()] + [i for i in df["option_b"].unique()])))
    mu = len(all_ids) / 2
    sigma = len(all_ids) / 6
    
    # Ratings initialization
    ratings = {_id: Rating(mu, sigma) for _id in all_ids}
    score = {_id: {"won": 0, "lost": 0} for _id in all_ids}

    df = df[["option_a", "option_b", "selected"]].rename(
        columns={"option_a": "left", "option_b": "right", "selected": "win"}
    )

    games = [
        (left, right, win) 
        for (left, right, win) in zip(df["left"], df["right"], df["win"])
    ]

    for (left, right, win) in games:
        if win == left:
            ratings[left], ratings[right] = rate_1vs1(ratings[left], ratings[right])
            score[left]["won"] += 1
            score[right]["lost"] += 1
        elif win == right:
            ratings[right], ratings[left] = rate_1vs1(ratings[right], ratings[left])
            score[right]["won"] += 1
            score[left]["lost"] += 1
        elif win == 0:
            ratings[left], ratings[right] = rate_1vs1(ratings[left], ratings[right], drawn=True)

    df = pd.DataFrame({"id": all_ids})
    df["mu"] = [ratings[_id].mu for _id in all_ids]
    df["sigma"] = [ratings[_id].sigma for _id in all_ids]
    df["skill"] = df["mu"] - 3 * df["sigma"]
    df["won"] = [score[_id]["won"] for _id in all_ids]
    df["lost"] = [score[_id]["lost"] for _id in all_ids]
    df["score"] = df["won"] - df["lost"]

    df = df.sort_values(by=["skill"], ascending=False)
    df["rank"] = list(range(1, len(all_ids) + 1))

    # df = df[["id", "skill", "rank"]].sort_values(by="id")
    # print(df) # FOR TESTING

    return df


# TEST #
########

#df = pd.read_csv("backend/compute/notebooks/sample2.csv")
#df = json.loads(df.to_json(orient="split"))["data"]
#skill(df)