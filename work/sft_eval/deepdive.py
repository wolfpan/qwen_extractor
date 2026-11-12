# -*- coding: utf-8 -*-
"""深挖：坏案例明细 + 数字幻觉核查 + 阶段2评分一致性。"""
import json, os, re
from collections import Counter
from difflib import SequenceMatcher

D = os.path.dirname(os.path.abspath(__file__))
def load(n): return json.load(open(os.path.join(D, n), encoding="utf-8"))

orig, R = load("ai_news_bge_orig.json"), {
    "R1": load("ai_news_bge_test.json"),
    "R2": load("ai_news_bge_testV2.json"),
    "R3": load("ai_news_bge_testV4.json"),
}
def norm_url(u):
    return re.sub(r"[?#].*$", "", (u or "").strip().rstrip("/")).lower()

gold = {}
for p in orig:
    m = p["payload"].get("metadata", {})
    u = norm_url(m.get("url", ""))
    gold[u] = {"title": m.get("out_title", ""), "summary": m.get("out_summary", ""),
               "content": p["payload"].get("content", ""), "level": m.get("level", ""),
               "score": m.get("score"), "dims": m.get("dims", ""), "type": m.get("type", ""),
               "src_title": m.get("title", "")}

FIELDS = ["新闻标题", "核心摘要", "原文连接", "重要性等级", "重要性评分", "新闻类型", "关键实体", "一票规则"]
def parse_content(c):
    out, cur = {}, None
    for line in c.split("\n"):
        m = re.match(r"^\[(.+?)\]:\s*(.*)$", line)
        if m and m.group(1) in FIELDS: cur = m.group(1); out[cur] = m.group(2)
        elif cur: out[cur] += "\n" + line
    return out

NUM = re.compile(r"\d+(?:[.,]\d+)*%?")
def numset(s): return set(n.rstrip(".,") for n in NUM.findall(s or ""))

print("########## 1) 坏案例明细 ##########")
for rn, pts in R.items():
    print(f"\n===== {rn} =====")
    for p in pts:
        m = p["payload"].get("metadata", {})
        f = parse_content(p["payload"].get("content", ""))
        t, s = f.get("新闻标题", ""), f.get("核心摘要", "")
        u = norm_url(m.get("url", ""))
        g = gold.get(u)
        # 提示词/契约泄漏检查（全文）
        leak = [w for w in ["15-30字", "核心事件标题", "120字", "instruction", "输入新闻", "JSON", "{", "}"] if w in t + s]
        if leak:
            print(f"[泄漏] {u}")
            print(f"   标记: {leak}")
            print(f"   标题({len(t)}): {t}")
            print(f"   摘要({len(s)}): {s[:150]}")
    # 最长3条摘要 & 与金标准相似度最低3对
    rows = []
    for p in pts:
        m = p["payload"].get("metadata", {})
        f = parse_content(p["payload"].get("content", ""))
        u = norm_url(m.get("url", ""))
        g = gold.get(u)
        if not g: continue
        t, s = f.get("新闻标题", ""), f.get("核心摘要", "")
        tsim = SequenceMatcher(None, t, g["title"]).ratio()
        body = numset(g["content"]) | numset(g["summary"]) | numset(g["src_title"])
        mn = numset(s)
        halluc = [x for x in mn if x not in body]
        gn = numset(g["summary"])
        miss = [x for x in gn if x not in mn]
        rows.append((u, t, s, g, tsim, halluc, miss))
    rows.sort(key=lambda r: r[4])
    print(f"\n-- {rn} 标题相似度最低3对 --")
    for u, t, s, g, tsim, halluc, miss in rows[:3]:
        print(f"  sim={tsim:.2f} {u}")
        print(f"    0.8B标题({len(t)}): {t}")
        print(f"    金标题  ({len(g['title'])}): {g['title']}")
        print(f"    0.8B摘要({len(s)}): {s[:120]}...")
        print(f"    金摘要  ({len(g['summary'])}): {g['summary'][:120]}...")
        if halluc: print(f"    幻觉数字: {halluc[:15]}")
        if miss: print(f"    漏掉金标准数字: {miss[:15]}")

print("\n########## 2) 数字幻觉率按轮汇总（仅matched） ##########")
for rn, pts in R.items():
    hh, rr = [], []
    for p in pts:
        m = p["payload"].get("metadata", {})
        f = parse_content(p["payload"].get("content", ""))
        g = gold.get(norm_url(m.get("url", "")))
        if not g: continue
        s = f.get("核心摘要", "")
        body = numset(g["content"]) | numset(g["summary"]) | numset(g["src_title"])
        mn, gn = numset(s), numset(g["summary"])
        if mn: hh.append(len([x for x in mn if x not in body]) / len(mn))
        if gn: rr.append(len(gn & mn) / len(gn))
    print(f"{rn}: 幻觉率均值={sum(hh)/len(hh):.2f} (n={len(hh)}), 金标准数字召回={sum(rr)/len(rr):.2f}")

print("\n########## 3) 阶段2评分一致性（matched，同一URL两管线） ##########")
for rn, pts in R.items():
    lv_same, sc_diffs, ty_same, n = 0, [], 0, 0
    for p in pts:
        m = p["payload"].get("metadata", {})
        g = gold.get(norm_url(m.get("url", "")))
        if not g: continue
        n += 1
        if m.get("level") == g["level"]: lv_same += 1
        if m.get("type") == g["type"]: ty_same += 1
        try: sc_diffs.append(float(m.get("score")) - float(g["score"]))
        except (TypeError, ValueError): pass
    if n:
        print(f"{rn}: n={n} 等级一致率={lv_same/n:.0%} 类型一致率={ty_same/n:.0%} "
              f"评分差均值={sum(sc_diffs)/len(sc_diffs):+.2f} 绝对差均值={sum(abs(x) for x in sc_diffs)/len(sc_diffs):.2f}")
