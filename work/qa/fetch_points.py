# -*- coding: utf-8 -*-
"""只读拉取三个集合全量点，存本地 JSON。"""
import json, urllib.request, sys, os

QDRANT = "http://10.8.0.5:6333"
OUT = os.path.dirname(os.path.abspath(__file__))

def _post(path, body):
    req = urllib.request.Request(QDRANT + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))

def scroll_all(collection):
    pts, offset = [], None
    while True:
        body = {"limit": 128, "with_payload": True, "with_vector": False}
        if offset:
            body["offset"] = offset
        r = _post(f"/collections/{collection}/points/scroll", body)
        pts += r["result"]["points"]
        offset = r["result"].get("next_page_offset")
        if not offset:
            return pts

for c in ["ai_news_bge_orig", "ai_news_bge_test", "ai_news_bge_testV2"]:
    pts = scroll_all(c)
    fn = os.path.join(OUT, c + ".json")
    json.dump(pts, open(fn, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{c}: {len(pts)} points -> {fn}")
