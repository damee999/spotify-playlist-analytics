import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import streamlit as st
import requests
import plotly.express as px

load_dotenv()


# DB
def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )


# AI (Ollama + Mock)
def build_ai_prompt(
    playlist_name: str,
    selected_id: str,
    start_date,
    end_date,
    kpis: dict,
    top_artists_text: str,
    trend_text: str,
    changes_text: str,
) -> str:
    return f"""
You are an AI data analyst.

Analyze this Spotify playlist using the metrics below. Write 6–9 sentences.
Rules:
- Be specific, avoid generic fluff.
- Mention stability/volatility and what it implies.
- Mention trend direction and what might be happening.
- Mention the top artists and what that suggests about the playlist identity.
- End with 2 bullet-point suggestions for what to monitor next.

Playlist:
- Name: {playlist_name}
- ID: {selected_id}
- Date range: {start_date} to {end_date}

KPIs (latest day):
- Latest snapshot: {kpis["latest_day"]}
- Tracks in latest snapshot: {kpis["n_tracks_latest"]}
- Avg popularity (latest): {kpis["avg_pop_latest"]:.2f}

Top artists (latest day, avg popularity):
{top_artists_text}

Trend (avg popularity by day):
{trend_text}

Changes vs previous snapshot (if available):
{changes_text}
""".strip()


def ollama_available(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def get_ollama_insights(prompt: str, base_url: str, model: str) -> str:
    """
    Uses Ollama local API: POST {base_url}/api/generate
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3},
    }
    r = requests.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip() or "Ollama returned empty output."


def get_mock_insights(_: str) -> str:
    
    return (
        "AI Insights (mock mode)\n\n"
        "This playlist’s latest snapshot suggests a stable core with change concentrated in a smaller set of tracks. "
        "The popularity trend will become more reliable after you collect more snapshot days.\n\n"
        "- Monitor stability % and the count of new/removed tracks daily.\n"
        "- Track movers (delta popularity) to spot songs gaining momentum early."
    )


@st.cache_data(ttl=3600)
def get_ai_insights_cached(prompt: str, provider: str, ollama_url: str, ollama_model: str) -> str:
    if provider == "Mock (free)":
        return get_mock_insights(prompt)

    if not ollama_available(ollama_url):
        return (
            "Ollama is not reachable on this machine, so I’m showing mock insights.\n\n"
            "To enable local AI:\n"
            "1) Install Ollama\n"
            "2) Make sure Ollama is running (it serves on localhost:11434)\n"
            "3) Pull a small model that fits your RAM (e.g. `llama3.2:1b`)\n"
        )

    try:
        return get_ollama_insights(prompt, ollama_url, ollama_model)
    except Exception as e:
        return (
            f"Failed to generate via Ollama ({type(e).__name__}). Showing mock insights.\n\n"
            f"{get_mock_insights(prompt)}"
        )


# UI
st.set_page_config(page_title="Spotify Playlist Analytics", layout="wide")
st.title("🎧 Spotify Playlist Analytics Dashboard")

# Sidebar AI settings
st.sidebar.markdown("### AI settings")
provider = st.sidebar.selectbox("Provider", ["Local (Ollama)", "Mock (free)"])
ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
ollama_model = st.sidebar.text_input("Ollama model", value="llama3.2:1b")

if st.sidebar.button("Test AI connection"):
    if provider == "Mock (free)":
        st.sidebar.success("Mock mode OK (no setup needed).")
    else:
        st.sidebar.success("Ollama reachable ✅" if ollama_available(ollama_url) else "Ollama not reachable ❌")


# Load data
conn = get_conn()

playlists = pd.read_sql_query(
    """
    SELECT DISTINCT playlist_id, playlist_name
    FROM track_snapshots
    WHERE playlist_id IS NOT NULL
    ORDER BY playlist_name;
    """,
    conn,
)

if playlists.empty:
    st.error("No playlist_id data found yet. Run spotify_extract.py after inserting playlist_id/playlist_name.")
    conn.close()
    st.stop()

playlist_name_to_id = dict(zip(playlists["playlist_name"], playlists["playlist_id"]))
selected_name = st.sidebar.selectbox("Playlist", playlists["playlist_name"].tolist())
selected_id = playlist_name_to_id[selected_name]
st.caption(f"Playlist ID: {selected_id}")

dates_df = pd.read_sql_query(
    """
    SELECT DISTINCT snapshot_date::date AS snapshot_date
    FROM track_snapshots
    WHERE playlist_id = %s
    ORDER BY snapshot_date;
    """,
    conn,
    params=(selected_id,),
)

min_date = dates_df["snapshot_date"].min()
max_date = dates_df["snapshot_date"].max()

start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

df = pd.read_sql_query(
    """
    SELECT
      s.snapshot_date::date AS snapshot_date,
      t.track_id,
      t.track_name,
      t.artist_name,
      t.album_name,
      s.popularity
    FROM track_snapshots s
    JOIN tracks t ON t.track_id = s.track_id
    WHERE s.playlist_id = %s
      AND s.snapshot_date::date BETWEEN %s AND %s;
    """,
    conn,
    params=(selected_id, start_date, end_date),
)

if df.empty:
    st.warning("No data for this playlist in the selected date range.")
    conn.close()
    st.stop()

latest_day = df["snapshot_date"].max()
df_latest = df[df["snapshot_date"] == latest_day].copy()

# KPIs
k1, k2, k3 = st.columns(3)
k1.metric("Latest snapshot date", str(latest_day))
k2.metric("Tracks in latest snapshot", int(df_latest.shape[0]))
k3.metric("Avg popularity (latest)", round(float(df_latest["popularity"].mean()), 2))


# Charts 
left, right = st.columns(2)

with left:
    st.subheader("Top artists (avg popularity) — latest day")
    top_artists = (
        df_latest.groupby("artist_name")["popularity"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    top_artists_df = top_artists.reset_index()
    top_artists_df.columns = ["artist_name", "avg_popularity"]

    fig_artists = px.bar(
        top_artists_df,
        x="artist_name",
        y="avg_popularity",
        title=None,
    )
    fig_artists.update_layout(xaxis_tickangle=-35, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_artists, use_container_width=True)

with right:
    st.subheader("Top tracks — latest day")
    top_tracks = df_latest.sort_values("popularity", ascending=False).head(20)
    st.dataframe(
        top_tracks[["track_name", "artist_name", "album_name", "popularity"]],
        use_container_width=True,
        hide_index=True,
    )

trend = df.groupby("snapshot_date")["popularity"].mean().reset_index().sort_values("snapshot_date")
st.subheader("Average popularity trend (selected range)")

lc, rc = st.columns([2, 3])
with lc:
    fig_trend = px.line(trend, x="snapshot_date", y="popularity", markers=True)
    fig_trend.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_trend, use_container_width=True)

with rc:
    st.markdown(
        """
**What this shows**
- Tracks average track popularity over time for the selected playlist.
- This becomes a stronger signal after you collect more daily snapshots.
"""
    )


st.subheader("Playlist changes (new / removed tracks)")

dates_check = pd.read_sql_query(
    """
    SELECT COUNT(DISTINCT snapshot_date::date) AS n_days
    FROM track_snapshots
    WHERE playlist_id = %s;
    """,
    conn,
    params=(selected_id,),
)
n_days = int(dates_check["n_days"].iloc[0])


n_today = n_prev = n_new = n_removed = n_overlap = 0
stability = 0.0

if n_days < 2:
    st.info("Not enough history yet. Run spotify_extract.py on another day for this playlist to unlock changes + stability.")
else:
    changes = pd.read_sql_query(
        """
        WITH days AS (
          SELECT DISTINCT snapshot_date::date AS d
          FROM track_snapshots
          WHERE playlist_id = %s
          ORDER BY d DESC
          LIMIT 2
        ),
        today AS (
          SELECT track_id
          FROM track_snapshots
          WHERE playlist_id = %s AND snapshot_date::date = (SELECT MAX(d) FROM days)
        ),
        prev AS (
          SELECT track_id
          FROM track_snapshots
          WHERE playlist_id = %s AND snapshot_date::date = (SELECT MIN(d) FROM days)
        ),
        new_tracks AS (
          SELECT t.track_id
          FROM today t
          LEFT JOIN prev p ON p.track_id = t.track_id
          WHERE p.track_id IS NULL
        ),
        removed_tracks AS (
          SELECT p.track_id
          FROM prev p
          LEFT JOIN today t ON t.track_id = p.track_id
          WHERE t.track_id IS NULL
        )
        SELECT
          (SELECT COUNT(*) FROM today) AS n_today,
          (SELECT COUNT(*) FROM prev) AS n_prev,
          (SELECT COUNT(*) FROM new_tracks) AS n_new,
          (SELECT COUNT(*) FROM removed_tracks) AS n_removed,
          (SELECT COUNT(*) FROM today JOIN prev USING(track_id)) AS n_overlap;
        """,
        conn,
        params=(selected_id, selected_id, selected_id),
    )

    stats = changes.iloc[0].to_dict()
    n_today = int(stats["n_today"])
    n_prev = int(stats["n_prev"])
    n_new = int(stats["n_new"])
    n_removed = int(stats["n_removed"])
    n_overlap = int(stats["n_overlap"])

    stability = (n_overlap / n_today * 100) if n_today else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracks today", n_today)
    c2.metric("Tracks previous", n_prev)
    c3.metric("New tracks", n_new)
    c4.metric("Removed tracks", n_removed)
    st.metric("Stability (overlap % of today)", f"{stability:.1f}%")

st.subheader("Biggest movers between last two snapshots")

if n_days < 2:
    st.info("Not enough snapshot dates yet for movers. Run spotify_extract.py on at least 2 different days for this playlist.")
else:
    last_two = pd.read_sql_query(
        """
        WITH days AS (
          SELECT DISTINCT snapshot_date::date AS d
          FROM track_snapshots
          WHERE playlist_id = %s
          ORDER BY d DESC
          LIMIT 2
        ),
        p AS (
          SELECT s.snapshot_date::date AS d, s.track_id, s.popularity
          FROM track_snapshots s
          WHERE s.playlist_id = %s
            AND s.snapshot_date::date IN (SELECT d FROM days)
        ),
        pivoted AS (
          SELECT
            track_id,
            MAX(CASE WHEN d = (SELECT MAX(d) FROM days) THEN popularity END) AS pop_today,
            MAX(CASE WHEN d = (SELECT MIN(d) FROM days) THEN popularity END) AS pop_prev
          FROM p
          GROUP BY track_id
        )
        SELECT
          t.track_name,
          t.artist_name,
          (pop_today - pop_prev) AS delta,
          pop_prev,
          pop_today
        FROM pivoted
        JOIN tracks t ON t.track_id = pivoted.track_id
        WHERE pop_today IS NOT NULL AND pop_prev IS NOT NULL
        ORDER BY delta DESC
        LIMIT 20;
        """,
        conn,
        params=(selected_id, selected_id),
    )

    if last_two.empty:
        st.info("Not enough overlap between the last two days to compute movers yet.")
    else:
        st.dataframe(last_two, use_container_width=True, hide_index=True)

conn.close()


# Signals 
st.subheader("📌 Signals")

avg_pop_latest = float(df_latest["popularity"].mean())
signals = []

if avg_pop_latest >= 70:
    signals.append("High average popularity → likely mainstream / currently relevant.")
elif avg_pop_latest >= 50:
    signals.append("Mid popularity → mixed catalog; some hits, some deeper cuts.")
else:
    signals.append("Lower average popularity → likely niche, older catalog, or more underground picks.")

if n_days >= 2:
    if stability >= 85:
        signals.append(f"Very stable ({stability:.1f}%) → strong identity / consistent curation.")
    elif stability >= 60:
        signals.append(f"Moderately stable ({stability:.1f}%) → rotates tracks but keeps a core sound.")
    else:
        signals.append(f"Volatile ({stability:.1f}%) → heavy rotation; possibly trend-driven.")
else:
    signals.append("Need 2+ snapshot days to compute stability and change signals.")

for s in signals:
    st.write("• " + s)


# AI prompt text
top_artists_text = "\n".join([f"- {name}: {val:.2f}" for name, val in top_artists.items()])
trend_text = "\n".join([f"- {row['snapshot_date']}: {row['popularity']:.2f}" for _, row in trend.iterrows()])

if n_days < 2:
    changes_text = "Not enough history (need 2+ snapshot days for this playlist)."
else:
    changes_text = (
        f"- Tracks today: {n_today}\n"
        f"- Tracks previous: {n_prev}\n"
        f"- New tracks: {n_new}\n"
        f"- Removed tracks: {n_removed}\n"
        f"- Overlap: {n_overlap}\n"
        f"- Stability (overlap % of today): {stability:.1f}%"
    )

kpis = {
    "latest_day": str(latest_day),
    "n_tracks_latest": int(df_latest.shape[0]),
    "avg_pop_latest": float(avg_pop_latest),
}

prompt = build_ai_prompt(
    playlist_name=selected_name,
    selected_id=selected_id,
    start_date=start_date,
    end_date=end_date,
    kpis=kpis,
    top_artists_text=top_artists_text,
    trend_text=trend_text,
    changes_text=changes_text,
)


# AI Insights 
st.subheader("🤖 AI Playlist Insights")

with st.expander("Show prompt sent to AI (for transparency)"):
    st.code(prompt)

if "ai_text" not in st.session_state:
    st.session_state.ai_text = None

colA, colB = st.columns([1, 3])
with colA:
    run_ai = st.button("Generate insights", type="primary")
with colB:
    st.caption("Insights generate only when you click the button (prevents reruns from spamming AI).")

if run_ai:
    with st.spinner("Generating insights..."):
        st.session_state.ai_text = get_ai_insights_cached(
            prompt=prompt,
            provider=provider,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )

if st.session_state.ai_text:
    st.markdown(st.session_state.ai_text)
else:
    st.info("Click **Generate insights** to create an analysis.")
