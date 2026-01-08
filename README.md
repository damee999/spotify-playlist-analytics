# Spotify Playlist Analytics — AI-Ready Data Pipeline & Dashboard

An end-to-end analytics project that automatically collects Spotify playlist data, stores daily snapshots in PostgreSQL, visualizes trends in a Streamlit dashboard, and generates **AI-assisted insights using a pluggable LLM layer** (local or mock).

This project was built with a **production mindset**: automation first, clean data modeling, explainable analytics, and AI integration that is cost-aware and provider-agnostic.

---

## What This Project Does

**Automatically:**
- Extracts track data from multiple Spotify playlists
- Stores **daily snapshots** in PostgreSQL (history preserved)
- Detects playlist changes (new / removed tracks)
- Tracks popularity trends over time
- Identifies biggest movers between snapshots
- Generates AI-assisted analytical summaries

---

## Tech Stack

**Data & Backend**
- Python
- Spotify Web API
- PostgreSQL
- psycopg2
- pandas

**Frontend / Analytics**
- Streamlit
- Matplotlib

**AI / Automation**
- Prompt engineering
- Local LLM via Ollama
- Deterministic mock AI fallback

**Dev & Ops**
- Virtual environments
- `.env` configuration
- Scheduled extraction via `.bat`
- GitHub version control

---

## How to Run Locally

### 1. Clone repository
```bash
git clone https://github.com/<your-username>/spotify-playlist-analytics.git
cd spotify-playlist-analytics
```

<img width="1920" height="1080" alt="1" src="https://github.com/user-attachments/assets/8f7c1fae-7ee6-4405-965e-00a0134a1749" />

<img width="1920" height="1080" alt="2" src="https://github.com/user-attachments/assets/f30fa551-0c97-4fc4-91fc-26c2ac15d18c" />

