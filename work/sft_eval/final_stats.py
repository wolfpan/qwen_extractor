# -*- coding: utf-8 -*-
"""最终指标：修正 URL 归一后的三轮 vs 金标准对比 + 稳定性 + 阶段2一致性。"""
import json, re
from collections import Counter
from difflib import SequenceMatcher

def load(n): return json.load(open(n, encoding="utf-8"))
orig = load("ai_news_bge_orig.json")
R = {"R1(test)": load("ai_news_bge_test.json"), "R2(testV2)": load("ai_news_bge_testV2.json"), "R3(testV4)": load("ai_news_bge_testV4.json")}

def norm_url(u):
    u = (u or "").strip().lower()
    u = re.sub(r"#.*$", "", u)
    if "?" in u:
        b, q = u.split("?", 1)
        keep = [kv for kv in q.split("&") if kv and not kv.startswith(("utm_", "spm", "from", "share"))]
        u = b + ("?" + "&".join(sorted(keep)) if keep else "")
    return u.rstrip("/")

gold = {}
for p in orig:
    m = p["payload"].get("metadata", {})
    gold[norm_url(m.get("url", ""))] = {
        "title": m.get("out_title", ""), "summary": m.get("out_summary", ""),
        "content": p["payload"].get("content", ""), "level": m.get("level", ""),
        "score": m.get("score"), "dims": m.get("dims", ""), "type": m.get("type", ""),
        "src_title": m.get("title", ""), "date": m.get("date", "")}

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
def bigrams(s):
    s = re.sub(r"\s+", "", s or "")
    return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else set()
def contain(a, b):
    A, B = bigrams(a), bigrams(b)
    return len(A & B) / len(A) if A else 0.0
def sim(a, b): return SequenceMatcher(None, a or "", b or "").ratio()
def cjk_ratio(s):
    if not s: return 0
    return len(re.findall(r"[\u4e00-\u9fff]", s)) / len(s)

BOILER = ["华尔街见闻", "清朗", "违规内容专项整治", "举报热线", "上海阿牛信息科技", "违法和不良信息举报", "VIP会员服务及大师课"]

def analyze(pts, label):
    recs = []
    for p in pts:
        m = p["payload"].get("metadata", {})
        f = parse_content(p["payload"].get("content", ""))
        u = norm_url(m.get("url", ""))
        recs.append({"u": u, "f": f, "m": m, "g": gold.get(u)})
    # 去重：同 URL 多次运行分别统计稳定性；指标用首次出现
    by_url = {}
    for r in recs: by_url.setdefault(r["u"], []).append(r)
    dup_urls = {u: v for u, v in by_url.items() if len(v) > 1}

    first = [v[0] for v in by_url.values()]
    matched = [r for r in first if r["g"]]
    junk = [r for r in first if any(b in r["f"].get("核心摘要", "") for b in BOILER)]
    tl = [len(r["f"].get("新闻标题", "")) for r in first]
    sl = [len(r["f"].get("核心摘要", "")) for r in first]
    eng_title = [r for r in first if cjk_ratio(r["f"].get("新闻标题", "")) < 0.5 and r["f"].get("新闻标题")]
    tmpl_leak = [r for r in first if re.search(r"15-30字|核心事件标题|120字内|英文译为中文", r["f"].get("新闻标题", "") + r["f"].get("核心摘要", ""))]

    out = {
        "总点数": len(pts), "唯一URL": len(by_url), "重复运行URL数": len(dup_urls),
        "匹配金标准": len(matched), "垃圾页直摘(未SKIP)": len(junk),
        "标题均长": round(sum(tl)/len(tl), 1), "标题超30或不足15": sum(1 for x in tl if not 15 <= x <= 30),
        "摘要均长": round(sum(sl)/len(sl), 1), "摘要>120占比": f"{sum(1 for x in sl if x>120)}/{len(sl)}",
        "摘要>200": sum(1 for x in sl if x > 200),
        "英文标题条数": len(eng_title), "模板泄漏条数": len(tmpl_leak),
    }
    if matched:
        g_tl = [len(r["g"]["title"]) for r in matched]
        g_sl = [len(r["g"]["summary"]) for r in matched]
        out["金标准标题均长(同集)"] = round(sum(g_tl)/len(g_tl), 1)
        out["金标准摘要均长(同集)"] = round(sum(g_sl)/len(g_sl), 1)
        out["金标准摘要>120占比"] = f"{sum(1 for x in g_sl if x>120)}/{len(g_sl)}"
        out["标题相似度(vs金)"] = round(sum(sim(r["f"].get("新闻标题"), r["g"]["title"]) for r in matched)/len(matched), 3)
        out["摘要相似度(vs金)"] = round(sum(sim(r["f"].get("核心摘要"), r["g"]["summary"]) for r in matched)/len(matched), 3)
        # 忠实度：模型摘要 bigram 在原文正文的覆盖率（越高越 grounded）
        out["摘要-原文覆盖率"] = round(sum(contain(r["f"].get("核心摘要"), r["g"]["content"]) for r in matched)/len(matched), 3)
        # 数字：模型摘要数字在正文中的占比（1-幻觉率），金标准数字召回
        h, rec = [], []
        for r in matched:
            body = numset(r["g"]["content"]) | numset(r["g"]["src_title"])
            mn = numset(r["f"].get("核心摘要"))
            gn = numset(r["g"]["summary"])
            if mn: h.append(len([x for x in mn if x not in body]) / len(mn))
            if gn: rec.append(len(gn & mn) / len(gn))
        out["数字幻觉率"] = round(sum(h)/len(h), 3) if h else None
        out["金标准数字召回"] = round(sum(rec)/len(rec), 3) if rec else None
        # 阶段2一致性
        lv = sum(1 for r in matched if r["m"].get("level") == r["g"]["level"])
        sc = [float(r["m"]["score"]) - float(r["g"]["score"]) for r in matched if r["m"].get("score") is not None and r["g"]["score"] is not None]
        out["阶段2等级一致率"] = f"{lv}/{len(matched)}"
        out["阶段2评分差(test-orig)"] = round(sum(sc)/len(sc), 2) if sc else None
    # 稳定性：重复 URL 的两次输出摘要相似度
    if dup_urls:
        sims, t_same, num_conflict = [], 0, 0
        for u, v in dup_urls.items():
            a, b = v[0]["f"].get("核心摘要", ""), v[1]["f"].get("核心摘要", "")
            sims.append(sim(a, b))
            if v[0]["f"].get("新闻标题") == v[1]["f"].get("新闻标题"): t_same += 1
            # 数字冲突：两次摘要中的数字集合互相不覆盖的比例
            na, nb = numset(a), numset(b)
            if na and nb and len(na & nb) / len(na | nb) < 0.6: num_conflict += 1
        out["重复运行摘要相似度均值"] = round(sum(sims)/len(sims), 3)
        out["重复运行标题完全一致"] = f"{t_same}/{len(dup_urls)}"
        out["重复运行数字冲突(<0.6 Jaccard)"] = num_conflict
    return out, junk, eng_title, tmpl_leak, dup_urls

results = {}
for label, pts in R.items():
    out, junk, eng, leak, dups = analyze(pts, label)
    results[label] = out
    print(f"===== {label} =====")
    for k, v in out.items(): print(f"  {k}: {v}")
    if junk:
        print("  [垃圾页直摘案例]")
        for r in junk:
            print(f"    {r['u'][-55:]} | {r['f'].get('新闻标题','')[:40]}")
    if eng:
        print("  [英文标题]")
        for r in eng: print(f"    {r['f'].get('新闻标题','')[:70]}")
    if leak:
        print("  [模板泄漏]")
        for r in leak: print(f"    {r['f'].get('新闻标题','')[:90]}")
    print()

# 金标准自身契约合规（全部163条 unique gold）
g_sl = [len(g["summary"]) for g in gold.values() if g["summary"]]
g_tl = [len(g["title"]) for g in gold.values() if g["title"]]
print(f"===== DeepSeek 金标准自身（{len(gold)}条）=====")
print(f"  标题均长 {sum(g_tl)/len(g_tl):.1f}，15-30字合规 {sum(1 for x in g_tl if 15<=x<=30)}/{len(g_tl)}")
print(f"  摘要均长 {sum(g_sl)/len(g_sl):.1f}，≤120字合规 {sum(1 for x in g_sl if x<=120)}/{len(g_sl)}")

json.dump(results, open("_final_stats.json", "w"), ensure_ascii=False, indent=1)
