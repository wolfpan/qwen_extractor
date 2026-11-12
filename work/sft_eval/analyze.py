# -*- coding: utf-8 -*-
"""SFT 三轮结果 vs DeepSeek 金标准 对比分析（只读本地 dump）。"""
import json, os, re, sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher

D = os.path.dirname(os.path.abspath(__file__))

def load(name):
    with open(os.path.join(D, name), "r", encoding="utf-8") as f:
        return json.load(f)

orig = load("ai_news_bge_orig.json")
rounds = {
    "R1_test": load("ai_news_bge_test.json"),
    "R2_testV2": load("ai_news_bge_testV2.json"),
    "R3_testV4": load("ai_news_bge_testV4.json"),
}

def norm_url(u):
    if not u: return ""
    u = u.strip().rstrip("/")
    u = re.sub(r"[?#].*$", "", u)
    return u.lower()

# ---------- orig: url -> gold ----------
gold = {}
for p in orig:
    m = p.get("payload", {}).get("metadata", {})
    u = norm_url(m.get("url", ""))
    if not u: continue
    gold[u] = {
        "id": p["id"],
        "title": m.get("out_title") or m.get("title") or "",
        "summary": m.get("out_summary") or "",
        "content": p.get("payload", {}).get("content", ""),
        "date": m.get("date", ""),
        "wf": m.get("wf_version", ""),
        "level": m.get("level", ""),
        "src_title": m.get("title", ""),
    }

print("== orig 金标准库 ==")
print("点数:", len(orig), " 唯一URL:", len(gold))
print("日期分布:", dict(sorted(Counter(g["date"] for g in gold.values()).items())))
print("wf_version:", dict(Counter(g["wf"] for g in gold.values())))

FIELDS = ["新闻标题", "核心摘要", "原文连接", "重要性等级", "重要性评分", "新闻类型", "关键实体", "一票规则"]

def parse_content(c):
    """解析 [字段]: 值 格式，摘要可能多行。"""
    out = {}
    cur = None
    for line in c.split("\n"):
        m = re.match(r"^\[(.+?)\]:\s*(.*)$", line)
        if m and m.group(1) in FIELDS:
            cur = m.group(1)
            out[cur] = m.group(2)
        elif cur:
            out[cur] += "\n" + line
    return out

NUM_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
SCAFFOLD = ["背景：", "背景:", "数据支撑", "趋势：", "趋势:", "影响：", "影响:", "意义：", "意义:"]
META = ["本文作者", "经授权发布", "本文由", "作者简介", "本文源自", "据本文"]

def numset(s):
    return set(n.rstrip(".,") for n in NUM_RE.findall(s or ""))

def bigrams(s):
    s = re.sub(r"\s+", "", s or "")
    return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else set([s])

def jac(a, b):
    A, B = bigrams(a), bigrams(b)
    return len(A & B) / len(A | B) if A | B else 0.0

def sim(a, b):
    return SequenceMatcher(None, a or "", b or "").ratio()

# ---------- per-round analysis ----------
all_stats = {}
pairs_all = {}

for rname, pts in rounds.items():
    rows = []
    unmatched = []
    for p in pts:
        pay = p.get("payload", {})
        m = pay.get("metadata", {})
        c = pay.get("content", "")
        f = parse_content(c)
        u = norm_url(m.get("url", "") or f.get("原文连接", ""))
        g = gold.get(u)
        rec = {
            "url": u, "date": m.get("date", ""), "level": m.get("level", ""),
            "title": f.get("新闻标题", ""), "summary": f.get("核心摘要", ""),
            "raw": c, "fields": sorted(f.keys()),
            "gold": g,
        }
        if g is None: unmatched.append(rec)
        rows.append(rec)
    pairs_all[rname] = rows
    matched = [r for r in rows if r["gold"]]

    tl = [len(r["title"]) for r in rows]
    sl = [len(r["summary"]) for r in rows]
    stats = {
        "n": len(rows), "n_matched": len(matched), "n_unmatched": len(unmatched),
        "dates": dict(sorted(Counter(r["date"] for r in rows).items())),
        "title_len_mean": sum(tl)/len(tl) if tl else 0,
        "title_len_range": (min(tl), max(tl)) if tl else None,
        "sum_len_mean": sum(sl)/len(sl) if sl else 0,
        "sum_len_range": (min(sl), max(sl)) if sl else None,
        "sum_over120": sum(1 for x in sl if x > 120),
        "sum_over150": sum(1 for x in sl if x > 150),
        "sum_over200": sum(1 for x in sl if x > 200),
        "title_out_15_30": sum(1 for r in rows if not (15 <= len(r["title"]) <= 30)),
        "skip": sum(1 for r in rows if "SKIP" in r["title"] or "SKIP" in r["summary"]),
        "scaffold": sum(1 for r in rows if any(s in r["summary"] for s in SCAFFOLD)),
        "multiline_summary": sum(1 for r in rows if "\n" in r["summary"].strip()),
        "meta_leak": sum(1 for r in rows if any(s in r["summary"] for s in META)),
        "field_set_variants": dict(Counter(tuple(r["fields"]) for r in rows)),
        "parse_fail": sum(1 for r in rows if not r["title"] and not r["summary"]),
    }
    if matched:
        ts = [sim(r["title"], r["gold"]["title"]) for r in matched]
        ss = [sim(r["summary"], r["gold"]["summary"]) for r in matched]
        tj = [jac(r["title"], r["gold"]["title"]) for r in matched]
        sj = [jac(r["summary"], r["gold"]["summary"]) for r in matched]
        # 数字忠实度：金标准摘要中的数字，模型摘要覆盖率
        nrec = []
        for r in matched:
            gn = numset(r["gold"]["summary"])
            if not gn: continue
            mn = numset(r["summary"])
            nrec.append(len(gn & mn) / len(gn))
        # 数字幻觉：模型摘要里、正文中没有的数字占比（正文含金标准）
        nhall = []
        for r in matched:
            body = numset(r["gold"]["content"]) | numset(r["gold"]["summary"]) | numset(r["gold"]["src_title"])
            mn = numset(r["summary"])
            if not mn: continue
            nhall.append(len([x for x in mn if x not in body]) / len(mn))
        stats.update({
            "title_sim_mean": sum(ts)/len(ts),
            "summary_sim_mean": sum(ss)/len(ss),
            "title_jac_mean": sum(tj)/len(tj),
            "summary_jac_mean": sum(sj)/len(sj),
            "num_recall_mean": sum(nrec)/len(nrec) if nrec else None,
            "num_halluc_mean": sum(nhall)/len(nhall) if nhall else None,
            "sum_exact_copy": sum(1 for x in ss if x > 0.98),
        })
        # 同事件标题长度对比
        gtl = [len(r["gold"]["title"]) for r in matched]
        gsl = [len(r["gold"]["summary"]) for r in matched]
        stats["gold_title_len_mean"] = sum(gtl)/len(gtl)
        stats["gold_sum_len_mean"] = sum(gsl)/len(gsl)
    all_stats[rname] = stats

# gold baseline stats over matched union
print()
for rname, s in all_stats.items():
    print(f"== {rname} ==")
    for k, v in s.items():
        if isinstance(v, float): v = round(v, 3)
        print(f"  {k}: {v}")
    print()

# unmatched urls
for rname, rows in pairs_all.items():
    um = [r for r in rows if not r["gold"]]
    if um:
        print(f"{rname} 未匹配 {len(um)} 条:")
        for r in um[:50]:
            print("   ", r["date"], r["url"][:80], "|", r["title"][:30])
print()

with open(os.path.join(D, "_pairs.json"), "w", encoding="utf-8") as f:
    json.dump({k: [{kk: (vv if kk != "gold" else None) for kk, vv in r.items()} for r in v] for k, v in pairs_all.items()}, f, ensure_ascii=False, indent=1)
print("明细已存 _pairs.json")
