import pandas as pd
import seaborn as sns

class ScatterSpace:
    def __init__(self, space, space2=None):
        self.THRESHOLD = 10
        self.size = 15
        self.space = space
        self.space2 = space2

    def create_limit_line(self, ax):
        THRESHOLD = self.THRESHOLD
        size = self.space.shape[0]
        upper_line = pd.DataFrame([
            {"a": 1 + THRESHOLD, "b": 1},
            {"a": size, "b": size - THRESHOLD}
        ])

        lower_line = pd.DataFrame([
            {"a": 1, "b": 1 + THRESHOLD},
            {"a": size - THRESHOLD, "b": size}
        ])

        trend_line = pd.DataFrame([
            {"a": 1, "b": 1},
            {"a": size, "b": size}
        ])
        
        ax = sns.lineplot(x="a", y="b", color="blue", data=trend_line)
        ax = sns.lineplot(x="a", y="b", color="#bbbbbb", data=upper_line)
        ax = sns.lineplot(x="a", y="b", color="#bbbbbb", data=lower_line)
        
        return ax
    
    def create_space(self):
        space = self.space
        size = self.size
        sns.set(rc={"figure.figsize": (size, size)})
        ax = sns.scatterplot(x="rank_1", y="rank_2", sizes=(50, 50), data=self.space)

        ax.invert_xaxis()
        ax.invert_yaxis()
        
        self.create_limit_line(ax)
        
        m = 15
        ticks = [1] + list(range(m, space.shape[0] + 1, m))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        
        for line in range(self.space.shape[0]):
            ax.text(space.rank_1[line]+0.01, space.rank_2[line], 
            space.nombre_corto_1[line], horizontalalignment="left", 
            size="small", color="black")
            
        return ax
    
    def create_diff_space(self):
        space1 = self.space
        space2 = self.space2
        key = "rank" # rank
        space1["delta_1"] = space1[f"{key}_1"] - space1[f"{key}_2"]
        space1 = space1[["delta_1", "id_1", "nombre_corto_1"]]

        space2["delta_2"] = space2[f"{key}_1"] - space2[f"{key}_2"]
        space2 = space2[["delta_2", "id_2"]]

        space = pd.merge(space1, space2, left_on="id_1", right_on="id_2")

        size = self.size
        sns.set(rc={"figure.figsize": (size, size)})

        ax = sns.scatterplot(x="delta_2", y="delta_1", data=space)
        for line in range(space.shape[0]):
             ax.text(space.delta_2[line]+0.01, space.delta_1[line], 
             space.nombre_corto_1[line], horizontalalignment="left", 
             size="small", color="black")

        n = space[["delta_2", "delta_1"]].abs().values.max()
        limit = pd.DataFrame([{"a": -n, "b": -n}, {"a": n, "b": n}])

        ax = sns.lineplot(x="a", y="b", color="red", data=limit)
        return ax
