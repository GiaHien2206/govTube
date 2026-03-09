import os, time
from datetime import datetime, timezone
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    raise RuntimeError("Missing YOUTUBE_API_KEY. Put it in .env")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def yt_get(endpoint: str, params: dict) -> dict:
    url = f"https://www.googleapis.com/youtube/v3/{endpoint}"
    params = {**params, "key": API_KEY}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"YT API error {r.status_code}: {r.text[:300]}")
    return r.json()

def chunk(lst, n=50):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def resolve_handle_to_channel_id(handle: str) -> str | None:
    h = str(handle).strip()
    if h.startswith("@"):
        h = h[1:]
    data = yt_get("channels", {"part": "id", "forHandle": h, "maxResults": 1})
    items = data.get("items", [])
    if not items:
        return None
    return items[0].get("id")

def get_uploads_playlist(channel_id: str) -> str:
    data = yt_get("channels", {"part": "contentDetails", "id": channel_id, "maxResults": 1})
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"No channel found for id={channel_id}")
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

def fetch_playlist_video_ids(playlist_id: str, max_items=200) -> list[str]:
    vids = []
    page = None
    while True:
        data = yt_get("playlistItems", {
            "part": "contentDetails",
            "playlistId": playlist_id,
            "maxResults": 50,
            "pageToken": page
        })
        for it in data.get("items", []):
            vids.append(it["contentDetails"]["videoId"])
            if len(vids) >= max_items:
                return vids
        page = data.get("nextPageToken")
        if not page:
            return vids

def fetch_videos_metadata_and_stats(video_ids: list[str], snapshot_time_utc: str) -> pd.DataFrame:
    rows = []
    for ids in chunk(video_ids, 50):
        data = yt_get("videos", {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(ids),
            "maxResults": 50
        })
        for it in data.get("items", []):
            sn = it.get("snippet", {})
            st = it.get("statistics", {})
            rows.append({
                "video_id": it["id"],
                "snapshot_time_utc": snapshot_time_utc,
                "publishedAt": sn.get("publishedAt"),
                "channelId": sn.get("channelId"),
                "title": sn.get("title"),
                "categoryId": sn.get("categoryId"),
                "viewCount": st.get("viewCount"),
                "likeCount": st.get("likeCount"),
                "commentCount": st.get("commentCount"),
            })
        time.sleep(0.2)
    return pd.DataFrame(rows)

def fetch_mostpopular(region: str, snapshot_time_utc: str, topn=50) -> pd.DataFrame:
    data = yt_get("videos", {
        "part": "id,snippet,statistics",
        "chart": "mostPopular",
        "regionCode": region,
        "maxResults": topn
    })
    rows = []
    rank = 1
    for it in data.get("items", []):
        sn = it.get("snippet", {})
        rows.append({
            "snapshot_time_utc": snapshot_time_utc,
            "region": region,
            "rank": rank,
            "video_id": it["id"],
            "channelId": sn.get("channelId"),
            "title": sn.get("title"),
        })
        rank += 1
    return pd.DataFrame(rows)

def append_csv(path: str, df: pd.DataFrame):
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def main():
    gov = pd.read_csv("gov_channels.csv")

    if "channel_id" not in gov.columns:
        gov["channel_id"] = None

    for i, row in gov.iterrows():
        if pd.isna(row["channel_id"]):
            cid = resolve_handle_to_channel_id(row["channel_handle"])
            gov.at[i, "channel_id"] = cid
            time.sleep(0.2)

    gov.to_csv(os.path.join(DATA_DIR, "gov_channels_resolved.csv"), index=False)

    rows = []
    for _, row in gov.iterrows():
        if not isinstance(row["channel_id"], str):
            continue
        uploads = get_uploads_playlist(row["channel_id"])
        vids = fetch_playlist_video_ids(uploads)
        for v in vids:
            rows.append({"agency": row["agency"], "channel_id": row["channel_id"], "video_id": v})

    channel_videos = pd.DataFrame(rows).drop_duplicates()
    channel_videos.to_csv(os.path.join(DATA_DIR, "channel_videos.csv"), index=False)

    snap = datetime.now(timezone.utc).isoformat()
    video_ids = channel_videos["video_id"].unique().tolist()
    meta = fetch_videos_metadata_and_stats(video_ids, snap)
    meta.to_csv(os.path.join(DATA_DIR, "videos_metadata_latest.csv"), index=False)

    ts = meta[["video_id", "snapshot_time_utc", "viewCount", "likeCount", "commentCount"]]
    append_csv(os.path.join(DATA_DIR, "engagement_timeseries.csv"), ts)

    mp = fetch_mostpopular("US", snap)
    append_csv(os.path.join(DATA_DIR, "mostpopular_timeseries.csv"), mp)

    print("DONE")

if __name__ == "__main__":
    main()
