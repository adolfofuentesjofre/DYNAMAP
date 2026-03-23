from copy import copy
import numpy as np
import pandas as pd
import json
from scipy.linalg import eig


def get_max_id(l):
    '''
    Reads a list of tuples (id, total) and returns the id for the max value of total in the list.
    '''
    highest_id = 0
    highest_total = 0
    for (i, t) in l:
        if t > highest_total:
            highest_id = i
            highest_total = t
    #print("highest_id: {}, highest_total: {}".format(highest_id,highest_total))
    return highest_id


# EIGENVECTOR ALGORITHM WITH POLITICAL ORIENTATION #
####################################################

def eigenvector(data):
    matrix = [list(row) for row in data]
    df = data
    for col in df.columns:
        df[col] = df[col].astype(int)

    # Get IDs and initialize values
    all_ids = sorted(list(dict.fromkeys([i for i in df["option_a"].unique()] + [i for i in df["option_b"].unique()])))
    num = {i:n for (n,i) in enumerate(all_ids)}
    num_inv = {n:i for (n,i) in enumerate(all_ids)}
    N = len(all_ids)

    df = df[["option_a", "option_b", "selected"]].rename(
        columns={"option_a": "left", "option_b": "right", "selected": "win"}
    )

    score_matrix = [[0 for i in range(N)] for j in range(N)]

    normalized_matrix = copy(score_matrix)

    games = [
        (left, right, win)
        for (left, right, win) in zip(df["left"], df["right"], df["win"])
    ]

    for (left, right, win) in games:
        if win == left:
            score_matrix[num[left]][num[right]] += 1
        elif win == right:
            score_matrix[num[right]][num[left]] += 1

    # Get reference proposal
    m = score_matrix
    percentages = [sum(m[i]) / (sum(m[i]) + sum([r[i] for r in m])) if (sum(m[i]) + sum([r[i] for r in m]))!=0 else 0 for i in range(N)]
    percentages = [(i,percentages[i]) for i in range(N)]
    #print("num_inv[id]: {}".format(num_inv[get_max_id(percentages)]))
    ref_id = num_inv[get_max_id(percentages)]


    for i in range(N):
        for j in range(N):
            if score_matrix[i][j] + score_matrix[j][i] != 0:
                normalized_matrix[i][j] /= score_matrix[i][j] + score_matrix[j][i]
            else:
                normalized_matrix[i][j] = 0

    normalized_matrix = np.array(normalized_matrix)

    eig_vals, eig_vecs = eig(normalized_matrix)

    df = pd.DataFrame(
        {
            "id": all_ids,
            "eigenvector": [eig_vecs[i][0].real for i in range(N)],
        }
    )

    df_test = df.sort_values(by=["eigenvector"])
    df_test["rank"] = [i for i in range(1, N + 1)]

    test_dict = {k:v for (k,v) in zip(df_test["id"], df_test["rank"])}

    if test_dict[ref_id] > int(N/2):
        df["eigenvector"] = df["eigenvector"] * -1
        df = df.sort_values(by=["eigenvector"])
        df["rank"] = [i for i in range(1, N + 1)]
    else:
        df = df.sort_values(by=["eigenvector"])
        df["rank"] = [i for i in range(1, N + 1)]

    df = df[["id", "eigenvector", "rank"]].sort_values(by="id")
    #print(df.sort_values(by="rank")) # FOR TESTING

    return df.to_json(orient="records")


# TEST #
########

#df = pd.read_csv("backend/compute/notebooks/sample2.csv")
#df = json.loads(df.to_json(orient="split"))["data"]
#eigenvector(df)