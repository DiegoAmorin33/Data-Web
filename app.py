import streamlit as st
import pandas as pd
import plotly.express as px
import kagglehub
from kagglehub import KaggleDatasetAdapter

st.set_page_config(
    page_title="Dashboard Videojuegos",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Dashboard de Videojuegos Populares")

@st.cache_data
def load_data():
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "matheusfonsecachaves/popular-video-games",
        ""
    )
    return df

df = load_data()

df = df.dropna()

st.sidebar.header("Filtros")

if "year" in df.columns:
    year_range = st.sidebar.slider(
        "Selecciona rango de años",
        int(df["year"].min()),
        int(df["year"].max()),
        (int(df["year"].min()), int(df["year"].max()))
    )

    df = df[(df["year"] >= year_range[0]) &
            (df["year"] <= year_range[1])]

if "genre" in df.columns:
    genres = st.sidebar.multiselect(
        "Selecciona género",
        options=df["genre"].unique(),
        default=df["genre"].unique()[:5]
    )

    if genres:
        df = df[df["genre"].isin(genres)]

st.subheader("🎯 Juegos por género")

if "genre" in df.columns:
    genre_count = df["genre"].value_counts().reset_index()
    genre_count.columns = ["genre", "count"]

    fig1 = px.bar(
        genre_count.head(10),
        x="genre",
        y="count",
        title="Top 10 géneros más populares"
    )

    st.plotly_chart(fig1, use_container_width=True)

st.subheader("📈 Evolución de juegos por año")

if "year" in df.columns:
    games_per_year = df.groupby("year").size().reset_index(name="count")

    fig2 = px.line(
        games_per_year,
        x="year",
        y="count",
        markers=True,
        title="Cantidad de juegos lanzados por año"
    )

    st.plotly_chart(fig2, use_container_width=True)

st.subheader("⭐ Rating vs Popularidad")

rating_col = None
popularity_col = None

for col in df.columns:
    if "rating" in col.lower():
        rating_col = col
    if "popularity" in col.lower() or "score" in col.lower():
        popularity_col = col

if rating_col and popularity_col:
    fig3 = px.scatter(
        df,
        x=rating_col,
        y=popularity_col,
        title="Relación entre rating y popularidad",
        opacity=0.6
    )

    st.plotly_chart(fig3, use_container_width=True)

st.subheader("📋 Datos del dataset")
st.dataframe(df.head(50))
