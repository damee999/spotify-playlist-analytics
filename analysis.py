import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )

# --- Load all snapshot data ---
conn = get_conn()

df = pd.read_sql_query("""
SELECT
  s.snapshot_date::date AS snapshot_date,
  s.track_id,
  s.popularity,
  s.danceability, s.energy, s.tempo, s.valence,
  t.track_name, t.artist_name, t.album_name
FROM track_snapshots s
JOIN tracks t ON t.track_id = s.track_id;
""", conn)

conn.close()

if df.empty:
    print("No data found in track_snapshots.")
    raise SystemExit(0)

latest_date = df["snapshot_date"].max()
print("✅ Loaded rows:", len(df))
print("📅 Latest snapshot_date:", latest_date)

# --- Filter to latest day by default (so you don't mix old runs) ---
df_latest = df[df["snapshot_date"] == latest_date].copy()
print("✅ Rows for latest date:", len(df_latest))
print("\nSample rows (latest date):")
print(df_latest[["artist_name", "track_name", "popularity"]].head(10).to_string(index=False))

# 1) Top artists (avg popularity) - latest date
top_artists = (
    df_latest.groupby("artist_name")["popularity"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure()
top_artists.plot(kind="bar")
plt.title(f"Top 10 Artists by Avg Popularity ({latest_date})")
plt.ylabel("Avg Popularity")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 2) Top tracks (popularity) - latest date
top_tracks = (
    df_latest.sort_values("popularity", ascending=False)
    .head(15)
    .set_index("track_name")["popularity"]
)

plt.figure()
top_tracks.plot(kind="bar")
plt.title(f"Top 15 Tracks by Popularity ({latest_date})")
plt.ylabel("Popularity")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 3) Trend over time (only if you have multiple snapshot dates)
if df["snapshot_date"].nunique() > 1:
    ts = (
        df.groupby("snapshot_date")["popularity"]
        .mean()
        .reset_index()
        .sort_values("snapshot_date")
    )

    plt.figure()
    plt.plot(ts["snapshot_date"], ts["popularity"], marker="o")
    plt.title("Average Popularity Over Time (All Snapshots)")
    plt.ylabel("Avg Popularity")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("\nℹ️ Only one snapshot date exists so far. Run the ETL on multiple days to unlock trend charts.")
