import os
from datetime import datetime, timezone
from dotenv import load_dotenv
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.exceptions as sp_exceptions
import psycopg2
from psycopg2.extras import execute_values
import sys

load_dotenv()

# Force UTF-8 for Windows console/logging
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- Spotify auth ---
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
))

PLAYLISTS = {
    "GOAT": "2lgylr5wlbiJ6UgVo9Ldxu",
    "World Hits": "6GC52W2Fznh6gv0ktEX1qH",
    "Rock Classics": "6b2dBnxolvwV2L1L4thWRm",
    "Game Music": "5lKJWYRfti3c9mudeYwecQ",
}

def fetch_all_playlist_items(playlist_id: str, page_limit: int = 100):
    """Paginate through a playlist to get all tracks (safe for big lists)."""
    items = []
    offset = 0
    while True:
        page = sp.playlist_items(
            playlist_id,
            limit=page_limit,
            offset=offset,
            additional_types=["track"],
        )
        items.extend(page.get("items", []))
        if page.get("next") is None or len(page.get("items", [])) == 0:
            break
        offset += page_limit
    return items


# --- PostgreSQL load (connect once) ---
conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    port=os.getenv("PGPORT"),
    dbname=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
)
cur = conn.cursor()

for PLAYLIST_NAME, PLAYLIST_ID in PLAYLISTS.items():
    print(f"\nProcessing playlist: {PLAYLIST_NAME} ({PLAYLIST_ID})")

    # (Optional) fetch real name from Spotify, but keep your label if you prefer
    try:
        playlist_meta = sp.playlist(PLAYLIST_ID, market="MK")
        playlist_name_real = playlist_meta.get("name") or PLAYLIST_NAME
    except Exception:
        playlist_name_real = PLAYLIST_NAME

    print(f"Fetching tracks from playlist {PLAYLIST_ID}...")
    items = fetch_all_playlist_items(PLAYLIST_ID)

    rows = []
    snapshot_date = datetime.now(timezone.utc).date().isoformat()

    for it in items:
        tr = it.get("track")
        if not tr or not tr.get("id"):
            continue
        rows.append({
            "snapshot_date": snapshot_date,
            "track_id": tr["id"],
            "track_name": tr["name"],
            "artist_name": tr["artists"][0]["name"] if tr.get("artists") else None,
            "album_name": tr["album"]["name"] if tr.get("album") else None,
            "popularity": tr.get("popularity"),
        })

    if not rows:
        print("⚠️ No tracks fetched (skipping).")
        continue

    track_ids = [r["track_id"] for r in rows]

    # Try audio features (will continue if forbidden)
    features_map = {}
    try:
        for i in range(0, len(track_ids), 100):
            chunk = track_ids[i:i+100]
            feats = sp.audio_features(chunk)
            for tid, feat in zip(chunk, feats):
                if feat:
                    features_map[tid] = {
                        "danceability": feat.get("danceability"),
                        "energy": feat.get("energy"),
                        "tempo": feat.get("tempo"),
                        "valence": feat.get("valence"),
                    }
    except sp_exceptions.SpotifyException as e:
        print("⚠️ Could not fetch audio features (continuing without them):")
        print(e)

    # Merge features (or None)
    for r in rows:
        f = features_map.get(r["track_id"], {})
        r["danceability"] = f.get("danceability")
        r["energy"]       = f.get("energy")
        r["tempo"]        = f.get("tempo")
        r["valence"]      = f.get("valence")

    df = pd.DataFrame(rows)
    print(f"Fetched {len(df)} tracks from the playlist.")

    # --- Save CSV snapshot (one per playlist per day) ---
    filename = f"spotify_snapshot_{PLAYLIST_ID}_{snapshot_date}.csv"
    df.to_csv(filename, index=False)
    print(f"💾 CSV snapshot saved -> {filename}")

    # =========================
    # BULK UPSERT: tracks table
    # =========================
    tracks_values = [
        (row["track_id"], row["track_name"], row["artist_name"], row["album_name"])
        for _, row in df.iterrows()
    ]

    execute_values(cur, """
        INSERT INTO tracks (track_id, track_name, artist_name, album_name)
        VALUES %s
        ON CONFLICT (track_id) DO UPDATE
        SET track_name = EXCLUDED.track_name,
            artist_name = EXCLUDED.artist_name,
            album_name = EXCLUDED.album_name;
    """, tracks_values, page_size=1000)

    # ============================
    # BULK UPSERT: track_snapshots
    # ============================
    snap_values = []
    for _, row in df.iterrows():
        snap_values.append((
            row["snapshot_date"],
            row["track_id"],
            int(row["popularity"]) if pd.notna(row["popularity"]) else None,
            row["danceability"],
            row["energy"],
            row["tempo"],
            row["valence"],
            PLAYLIST_ID,
            playlist_name_real,
        ))

    execute_values(cur, """
        INSERT INTO track_snapshots (
            snapshot_date, track_id, popularity, danceability, energy, tempo, valence,
            playlist_id, playlist_name
        )
        VALUES %s
        ON CONFLICT (playlist_id, snapshot_date, track_id)
        DO UPDATE SET
            popularity = EXCLUDED.popularity,
            danceability = EXCLUDED.danceability,
            energy = EXCLUDED.energy,
            tempo = EXCLUDED.tempo,
            valence = EXCLUDED.valence,
            playlist_name = EXCLUDED.playlist_name;
    """, snap_values, page_size=1000)

    conn.commit()
    print("✅ Loaded into PostgreSQL for this playlist.")

cur.close()
conn.close()
print("\n✅ All playlists processed.")
