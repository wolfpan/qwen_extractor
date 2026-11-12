# -*- coding: utf-8 -*-
"""只读拉取 Qdrant 四个集合全量点，存本地 JSON。仅用 scroll API，不做任何写操作。"""
import json, urllib.request, os

HOST = "http://10.8.0.5:6333"
COLLECTIONS = ["ai_news_bge_orig", "ai_news_bge_test", "ai_news_bge_testV2", "ai_news_bge_testV4"]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def scroll_all(name):
    points, offset = [], None
    while True:
        body = {"limit": 100, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(
            f"{HOST}/collections/{name}/points/scroll",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        r = json.load(urllib.request.urlopen(req, timeout=30))["result"]
        points.extend(r["points"])
        offset = r.get("next_page_offset")
        if offset is None:
            return points

for c in COLLECTIONS:
    pts = scroll_all(c)
    path = os.path.join(OUT_DIR, f"{c}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pts, f, ensure_ascii=False)
    print(c, len(pts), "->", path)
