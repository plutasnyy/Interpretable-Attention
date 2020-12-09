import pandas as pd
import plotly.express as px

df = pd.read_csv('threshold_cumulated.csv')
fig = px.box(df, y="threshold")
fig.show()
# fig = px.scatter(df, x="tokens", y="threshold")
# fig.write_image("threshold_cumulated.png")
