import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    dbname=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
)

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS tracks (
    track_id TEXT PRIMARY KEY,
    track_name TEXT,
    artist_name TEXT,
    album_name TEXT
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS track_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE,
    track_id TEXT REFERENCES tracks(track_id),
    popularity INTEGER,
    danceability REAL,
    energy REAL,
    tempo REAL,
    valence REAL
);
""")

conn.commit()
cur.close()
conn.close()

print("✅ PostgreSQL database and tables created successfully.")
