# -*- coding: utf-8 -*-
"""三集合提取质量只读分析：契约合规 + 与 DeepSeek 金标准一致性。"""
import json, re, unicodedata
from collections import Counter, defaultdict

QA = r"C:\Codes\qwen_extractor\work\qa"

def load(c):
    return json.load(open(f"{QA}\\{c}.json", encoding="utf-8"))

# ---------------- 解析 test/testV2 的 content ----------------
RE_TITLE = re.compile(r"\[新闻标题\]:\s*(.*?)\s*\n", re.S)
RE_SUMM  = re.compile(r"\[核心摘要\]:\s*(.*?)\s*\n\[", re.S)

def parse_point(p):
    pl = p["payload"]; m = pl.get("metadata", {})
    c = pl.get("content", "")
    t = RE_TITLE.search(c); s = RE_SUMM.search(c)
    return {
        "id": p["id"], "url": m.get("url", ""), "date": m.get("date", ""),
        "title": t.group(1).strip() if t else None,
        "summary": s.group(1).strip() if s else None,
        "level": m.get("level", ""), "score": m.get("score", ""),
        "type": m.get("type", ""), "veto": m.get("veto", ""),
        "focus": m.get("focus", ""), "entities": m.get("entities", ""),
        "parse_ok": bool(t and s),
    }

def load_gold():
    g = {}
    for p in load("ai_news_bge_orig"):
        m = p["payload"].get("metadata", {})
        url = m.get("url", "")
        try:
            out = json.loads(m["out_json"]) if isinstance(m.get("out_json"), str) else m.get("out_json")
        except Exception:
            out = None
        g[url] = {
            "title": m.get("out_title") or (out or {}).get("新闻标题", ""),
            "summary": m.get("out_summary") or (out or {}).get("核心摘要", ""),
            "content_len": len(p["payload"].get("content", "")),
            "date": m.get("date", ""),
        }
    return g

# ---------------- 文本度量 ----------------
def bigrams(s):
    s = re.sub(r"\s+", "", s)
    return set(s[i:i+2] for i in range(len(s)-1))

def dice(a, b):
    A, B = bigrams(a), bigrams(b)
    if not A or not B: return 0.0
    return 2*len(A & B)/(len(A) + len(B))

RE_NUM = re.compile(r"\d+(?:[.,]\d+)*%?")
def numbers(s):
    return Counter(re.sub(r"[,，]$", "", n) for n in RE_NUM.findall(s or ""))

def num_recall(q, gold):
    """金标准摘要中的数字被 qwen 摘要保留的比例（多重集按出现次数算）。"""
    gq, qq = numbers(gold), numbers(q)
    tot = sum(gq.values())
    if tot == 0: return None, 0, 0
    kept = sum(min(c, qq[n]) for n, c in gq.items())
    extra = sum(max(0, c - gq[n]) for n, c in qq.items())  # 金标准里没有的数字（幻觉代理）
    return kept/tot, tot, extra

def has_repeat(s, n=12, k=3):
    """长度n的片段重复出现>=k次 → 复读机嫌疑。"""
    if not s or len(s) < n*k: return False
    c = Counter(s[i:i+n] for i in range(len(s)-n+1))
    return any(v >= k for v in c.values())

def wide_len(s):
    """中文按1、ASCII按0.5 的视觉宽度。"""
    return sum(1 if unicodedata.east_asian_width(ch) in "WF" else 0.5 for ch in s)

CURLY = re.compile(r"[\u201c\u201d\u2018\u2019]")   # “”‘’
ELLIPSIS = re.compile(r"…|\.{3,}")
CLICKBAIT = re.compile(r"[!！]{1,}|\?{2,}|……$|震惊|竟然|突发|重磅|刚刚")
MD_RESIDUE = re.compile(r"\*\*|__|\[.*?\]\(.*?\)|<[^>]+>|#{1,3}\s")

# ---------------- 主流程 ----------------
gold = load_gold()
sets = {}
for c in ["ai_news_bge_test", "ai_news_bge_testV2"]:
    sets[c] = [parse_point(p) for p in load(c)]

print("="*70)
print("0) 基本情况")
for c, rows in sets.items():
    dates = Counter(r["date"] for r in rows)
    dup = len(rows) - len(set(r["url"] for r in rows))
    print(f"{c}: {len(rows)} 点, 重复URL点 {dup}, 解析失败 {sum(not r['parse_ok'] for r in rows)}, 日期 {dict(dates)}")

def contract_report(name, rows):
    print("\n" + "="*70)
    print(f"1) 契约合规 — {name}  (n={len(rows)})")
    ok = [r for r in rows if r["parse_ok"]]
    skips = [r for r in ok if r["title"] == "SKIP" or r["summary"] == "SKIP"]
    real = [r for r in ok if r not in skips]
    print(f"  SKIP 输出: {len(skips)} / 正常输出: {len(real)}")
    tl = [len(r["title"]) for r in real if r["title"]]
    sl = [len(r["summary"]) for r in real if r["summary"]]
    def band(vals, lo, hi):
        return f"min {min(vals)} / 中位 {sorted(vals)[len(vals)//2]} / max {max(vals)} / 均值 {sum(vals)/len(vals):.0f} | 在[{lo},{hi}]内 {sum(lo<=v<=hi for v in vals)}/{len(vals)}"
    if tl: print("  标题字数(目标15-30):", band(tl, 15, 30))
    if sl: print("  摘要字数(目标<=120):", band(sl, 0, 120))
    if sl: print(f"  摘要>120字: {sum(v>120 for v in sl)}/{len(sl)}  >150字: {sum(v>150 for v in sl)}/{len(sl)}")
    def count(pred, label):
        n = sum(1 for r in real if pred(r))
        print(f"  {label}: {n}/{len(real)}")
        return [r for r in real if pred(r)]
    curly = count(lambda r: CURLY.search(r["title"]+r["summary"]), "含弯引号“”‘’(契约要求「」)")
    count(lambda r: ELLIPSIS.search(r["title"]), "标题含省略号(标题党残留)")
    count(lambda r: CLICKBAIT.search(r["title"]), "标题含点击诱饵标记(！/??/震惊/突发…)")
    count(lambda r: MD_RESIDUE.search(r["title"]+r["summary"]), "含markdown/HTML残留")
    count(lambda r: has_repeat(r["summary"]), "摘要疑似复读(12字片段重复>=3次)")
    count(lambda r: len(set(r["summary"])) < len(r["summary"])*0.55 and len(r["summary"])>60, "摘要字符多样性过低(<55%)")
    return real, curly

reals = {}
for c, rows in sets.items():
    reals[c], _ = contract_report(c, rows)

# ---------------- 2) 与金标准对比 ----------------
def gold_report(name, rows):
    print("\n" + "="*70)
    print(f"2) 与 DeepSeek 金标准对比 — {name}")
    pairs = [(r, gold[r["url"]]) for r in rows if r["parse_ok"] and r["url"] in gold]
    print(f"  可比对样本: {len(pairs)}")
    if not pairs: return
    # 金标准 SKIP 而模型没 SKIP / 反之
    gskip = [(r, g) for r, g in pairs if g["title"] == "SKIP"]
    mskip = [(r, g) for r, g in pairs if r["title"] == "SKIP"]
    print(f"  金标准SKIP: {len(gskip)} (其中模型仍输出: {sum(1 for r,g in gskip if r['title']!='SKIP')})")
    print(f"  模型SKIP: {len(mskip)} (其中金标准有正常输出: {sum(1 for r,g in mskip if g['title']!='SKIP')})")
    both = [(r, g) for r, g in pairs if r["title"] != "SKIP" and g["title"] != "SKIP"]
    if not both: return
    dt = [dice(r["title"], g["title"]) for r, g in both]
    ds = [dice(r["summary"], g["summary"]) for r, g in both]
    nr, tot, extra = [], [], []
    for r, g in both:
        rc, t, e = num_recall(r["summary"], g["summary"])
        if rc is not None: nr.append(rc); tot.append(t); extra.append(e)
    print(f"  标题 bigram Dice: 均值 {sum(dt)/len(dt):.2f} | 中位 {sorted(dt)[len(dt)//2]:.2f} | <0.15 的: {sum(x<0.15 for x in dt)}/{len(dt)}")
    print(f"  摘要 bigram Dice: 均值 {sum(ds)/len(ds):.2f} | 中位 {sorted(ds)[len(ds)//2]:.2f} | <0.15 的: {sum(x<0.15 for x in ds)}/{len(ds)}")
    if nr:
        print(f"  金标准摘要数字保留率: 均值 {sum(nr)/len(nr):.0%} | 全保留 {sum(x==1 for x in nr)}/{len(nr)} | 零保留 {sum(x==0 for x in nr)}/{len(nr)}")
        print(f"  金标准外新增数字(幻觉代理): 总 {sum(extra)} 个 / {len(extra)} 篇")
    # 逐条明细
    print(f"  -- 逐条 (Dice摘要 | 数字保留 | 标题) --")
    for (r, g), d_s, d_t in sorted(zip(both, ds, dt), key=lambda x: x[1]):
        rc, t, e = num_recall(r["summary"], g["summary"])
        rcs = f"{rc:.0%}({t}数,多{e})" if rc is not None else "-"
        print(f"  {d_s:.2f}|{d_t:.2f} | {rcs:14s} | Q: {r['title'][:28]}")
        print(f"  {' '*16} | G: {g['title'][:28]}")

for c, rows in sets.items():
    gold_report(c, rows)

# ---------------- 3) 金标准自身的长度基线 ----------------
print("\n" + "="*70)
print("3) 金标准(DeepSeek, orig全量155)长度基线")
gt = [len(g["title"]) for g in gold.values() if g["title"] and g["title"] != "SKIP"]
gs = [len(g["summary"]) for g in gold.values() if g["summary"] and g["title"] != "SKIP"]
print(f"  标题: 中位 {sorted(gt)[len(gt)//2]} / max {max(gt)} / >30字 {sum(v>30 for v in gt)}/{len(gt)}")
print(f"  摘要: 中位 {sorted(gs)[len(gs)//2]} / max {max(gs)} / >120字 {sum(v>120 for v in gs)}/{len(gs)}")
gsk = sum(1 for g in gold.values() if g["title"] == "SKIP")
print(f"  金标准 SKIP: {gsk}/{len(gold)}")

# ---------------- 4) testV2 内部重复 URL ----------------
print("\n" + "="*70)
print("4) testV2 重复URL明细(同文两次的输出是否一致)")
byu = defaultdict(list)
for r in sets["ai_news_bge_testV2"]:
    byu[r["url"]].append(r)
for u, rs in byu.items():
    if len(rs) > 1:
        same_t = len(set(r["title"] for r in rs)) == 1
        print(f"  {u[:60]} x{len(rs)} 标题一致={same_t}")
        if not same_t:
            for r in rs: print(f"    - {r['title'][:40]}")
