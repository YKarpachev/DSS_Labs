from typing import List
import pandas as pd

df: pd.DataFrame = pd.DataFrame(
    [
        ("Bistro Nova", 1.2, 18, 4.4, 15),
        ("Pasta Punto", 3.5, 14, 4.1, 25),
        ("Sakura Bar", 2.0, 22, 4.6, 10),
        ("Green Fork", 0.8, 16, 4.2, 20),
        ("BBQ House", 4.2, 20, 4.0, 5),
    ],
    columns=["Restaurant", "Distance", "Price", "Rating",
             "Wait"]).set_index("Restaurant")

maximize: List[str] = ["Rating"]
minimize: List[str] = ["Distance", "Price", "Wait"]

weights: pd.Series = pd.Series({
    "Rating": 0.4,
    "Distance": 0.2,
    "Price": 0.2,
    "Wait": 0.2
})


def norm_maximize(s: pd.Series | pd.DataFrame) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn)


def norm_minimize(s) -> pd.Series:
    mn, mx = s.min(), s.max()
    return (mx - s) / (mx - mn)


df_norm: pd.DataFrame = pd.DataFrame(index=df.index)
for col in maximize:
    df_norm[col] = norm_maximize(df[col])
for col in minimize:
    df_norm[col] = norm_minimize(df[col])

scores: pd.DataFrame = (df_norm * weights).sum(axis=1)
result: pd.DataFrame = pd.DataFrame({
    "Score": scores
}).sort_values("Score", ascending=False)

print(result)
