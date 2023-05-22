import pandas as pd
import cpt_tools    # Quelle: https://gist.github.com/grahamharrison68/1187c53d078c3c899b534852fe8edf9c
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture as GMM
from matplotlib.patches import Ellipse
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from pgmpy.models import BayesianNetwork


def create_correlation_matrix(df):
    sns.heatmap(df.corr(), cmap='coolwarm', linecolor='white', linewidths=1, annot=True)
    plt.show()


def save_and_clean_synthetic_df(synthetic_data):
    synthetic_data.to_csv('Faker_Gesundheitsbefragung_synthetic_data_Output.csv', index=False, encoding='utf-8-sig', sep=';')
    synthetic_data.replace('Männlich', 0, inplace=True)
    synthetic_data.replace('Weiblich', 1, inplace=True)
    synthetic_data.replace('15-29', 15, inplace=True)
    synthetic_data.replace('30-44', 30, inplace=True)
    synthetic_data.replace('45-59', 45, inplace=True)
    synthetic_data.replace('60-74', 60, inplace=True)
    synthetic_data.replace('75-100', 75, inplace=True)
    synthetic_data = synthetic_data.loc[:, ['Geschlecht', 'Alterskategorie', 'Nackenschmerzen', 'Rückenschmerzen']]
    return synthetic_data


def get_and_clean_input_df():
    df = pd.read_csv('Faker_Gesundheitsbefragung_Output.csv', sep=';')
    df.replace('Männlich', 0, inplace=True)
    df.replace('Weiblich', 1, inplace=True)
    df.replace('15-29', 15, inplace=True)
    df.replace('30-44', 30, inplace=True)
    df.replace('45-59', 45, inplace=True)
    df.replace('60-74', 60, inplace=True)
    df.replace('75-100', 75, inplace=True)
    df.drop('Alter', axis=1, inplace=True)
    df.drop('Bluthochdruck', axis=1, inplace=True)
    return df


def create_bayesian_network(df):
    model = BayesianNetwork([('Geschlecht', 'Nackenschmerzen'),
                             ('Alterskategorie', 'Nackenschmerzen'),
                             ('Geschlecht', 'Rückenschmerzen'),
                             ('Alterskategorie', 'Rückenschmerzen')])
    model.to_daft().render()
    plt.show()
    model.fit(df)
    model.check_model()
    for cpt in model.get_cpds():
        print(cpt_tools.display_cpt(cpt))
    return model



def plot_as_line_graph_synth():
    df = pd.read_csv('Faker_Gesundheitsbefragung_synthetic_data_Output.csv', sep=';')

    df_neck_pain = df.query("Nackenschmerzen == 1")
    df_back_pain = df.query("Rückenschmerzen == 1")
    df_age_count = df.groupby("Alterskategorie")["Alterskategorie"].count().rename_axis("Alterskategorie").reset_index(name="total_age_count")
    df_neck_pain_count = df_neck_pain["Alterskategorie"].value_counts().rename_axis("Alterskategorie").reset_index(name="neck_pain_count")
    df_back_pain_count = df_back_pain["Alterskategorie"].value_counts().rename_axis("Alterskategorie").reset_index(name="back_pain_count")

    merged_df = pd.merge(pd.merge(df_neck_pain_count, df_back_pain_count, on="Alterskategorie"), df_age_count, on="Alterskategorie")
    merged_df['neck_pain_percent'] = (merged_df['neck_pain_count'] / merged_df['total_age_count'])
    merged_df['back_pain_percent'] = (merged_df['back_pain_count'] / merged_df['total_age_count'])
    merged_df = merged_df.sort_values(by="Alterskategorie", ascending=True)

    fig = px.line(merged_df, x="Alterskategorie", y=["neck_pain_percent", "back_pain_percent"],
                  labels={"x": "Alterskategorie",
                          "y": "Prozentualer Anteil",
                          "neck_pain_percent": "Nackenschmerzen",
                          "back_pain_percent": "Rückenschmerzen"},
                  markers=True,
                  color_discrete_sequence=["#1F497D", "#8EB4E3"])

    fig.layout.yaxis.tickformat = ',.0%'
    fig.update_layout(
        title="Prozentualer Anteil von chronischen Nacken- oder Rückenschmerzen im Alter",
        xaxis_title="Alterskategorie",
        yaxis_title="Prozentualer Anteil",
        legend_title="Krankheit",
        plot_bgcolor='rgb(233,233,233)'
    )

    labels = {"neck_pain_percent": "Nackenschmerzen",
              "back_pain_percent": "Rückenschmerzen"}

    fig.for_each_trace(lambda x: x.update(name=labels[x.name]))
    fig.show()


def main():
    original_df = get_and_clean_input_df()
    model = create_bayesian_network(original_df)
    synthetic_data = model.simulate(n_samples=100)
    synthetic_data = save_and_clean_synthetic_df(synthetic_data)
    create_correlation_matrix(original_df)
    create_correlation_matrix(synthetic_data)
    plot_as_line_graph_synth()


if __name__ == "__main__":
    main()


