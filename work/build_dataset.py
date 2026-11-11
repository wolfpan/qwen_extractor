# -*- coding: utf-8 -*-
"""
build_dataset.py — 从 Qdrant ai_news_bge_orig 归档库构建 qwen3.5-0.8b 阶段1提取器 SFT 数据集

数据流：
  ai_news_bge_orig.payload.content        = v3.107-orig 工作流「拆包&数据清洗」后的正文（原始输入）
  ai_news_bge_orig.payload.metadata.title = RSS 条目的原始标题（C4A 抓取条目为空串）
  ai_news_bge_orig.payload.metadata.out_json = DeepSeek-V4.1-flash 的完整结构化输出（金标准）

关键保真步骤：
  1. 归档 content 只经过 v3.107 清洗；生产端 0.8B（v4.9-beta test 工作流）实际看到的是
     v4.9 清洗后的文本（markdown 符号/HTML 实体剥离 + APP 引导噪音清理 + 双锚点截取 + 3500 字截断）。
     本脚本把 test 工作流「拆包&数据清洗」节点的 v4.9 增量清洗逐条正则移植到 Python，
     保证训练输入分布 == 推理输入分布。
  2. instruction / input / output 与 v4.9 阶段1 LLM 节点逐字对齐：
     - system  = 「LLM阶段1·提取压缩(Qwen0.8B)」的 message（提取器提示词，2键契约）
     - user    = "输入：标题: {title} 正文: {desc}\n\n请严格输出包含 新闻标题、核心摘要 全部2个键的JSON："
     - output  = 紧凑 JSON {"新闻标题":...,"核心摘要":...}（仅2键；创建时间由管线生成，模型不输出）
  3. 增补合成 SKIP 样本，教会模型「空正文/404/访问受限/纯噪音 → SKIP」契约。
"""
import json, re, random, sys
from urllib.parse import urlparse
from collections import Counter

random.seed(3407)

WORK = r"D:\Codes\qwen3.5_0.8b_SFT\work"
WF   = r"D:\Codes\n8n-ai-stacks\exports\抓取每日科技新闻-test.json"

# ---------------------------------------------------------------- 1. 从工作流提取逐字 prompt
wf = json.load(open(WF, encoding="utf-8"))
stage1 = next(n for n in wf["nodes"] if n["name"] == "LLM阶段1·提取压缩(Qwen0.8B)")
SYSTEM_PROMPT = stage1["parameters"]["messages"]["messageValues"][0]["message"]
# user 模板（去掉 n8n 表达式前缀 '='，占位符换回）
USER_TMPL_RAW = stage1["parameters"]["text"]
assert USER_TMPL_RAW.startswith("=")
# "输入：标题: {{ $json.title || '' }} 正文: {{ $json.desc }}\n\n请严格输出..."
USER_HEAD, USER_TAIL = "输入：标题: ", " 正文: "
m = re.search(r"\}\}\s*\n\n(.*)$", USER_TMPL_RAW, re.S)
USER_SUFFIX = "\n\n" + m.group(1)  # "\n\n请严格输出包含 新闻标题、核心摘要 全部2个键的JSON："

def make_input(title: str, desc: str) -> str:
    return f"输入：标题: {title or ''} 正文: {desc}{USER_SUFFIX}"

# ---------------------------------------------------------------- 2. v4.9-beta 清洗移植（与 test 工作流逐条对齐）
RE_IMG      = re.compile(r"!\[.*?\]\(.*?\)")
RE_LINK     = re.compile(r"\[(.*?)\]\(.*?\)")
RE_HTML     = re.compile(r"<[^>]*>")
RE_EMPH     = re.compile(r"(\*\*|__|~~|^>\s*)", re.M)
RE_ENTITY   = re.compile(r"&(ldquo|rdquo|lsquo|rsquo|quot|nbsp|amp|#\d+);", re.I)
RE_APP_PAIR = re.compile(r"^[^\n]+\n\s*打开\s*APP\s*$", re.M | re.I)
RE_APP_LINE = re.compile(r"^\s*(APP\s*1?\s*X?\s*主页|扫码\s*打开\S{0,4}\s*APP|打开\s*APP|下载\S{0,6}APP)\s*$", re.M | re.I)
RE_H1       = re.compile(r"^#\s+\S.*$", re.M)
RE_DATETIME = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}\s*$", re.M)
RE_NL3      = re.compile(r"\n{3,}")
DESC_CAP    = 3500   # v4.9 阶段1 注意力适配截断

def clean_v49(text: str) -> str:
    text = RE_IMG.sub("", text)
    text = RE_LINK.sub(r"\1", text)
    text = RE_HTML.sub("", text)
    # v3.115: markdown 加粗/斜体/引用符号 + HTML 实体剥离
    text = RE_EMPH.sub("", text)
    text = RE_ENTITY.sub(" ", text)
    # v3.115: APP 引导/推荐位噪音行清理
    text = RE_APP_PAIR.sub("", text)
    text = RE_APP_LINE.sub("", text)
    # 双锚点截取：头部 600 字残留 APP/扫码 特征时，跳到 H1（优先）或独立日期时间行
    if re.search(r"APP|扫码", text[:600], re.I):
        win = text[:900]
        h1 = RE_H1.search(win)
        if h1 and h1.start() > 0:
            text = text[h1.start():]
        else:
            dt = RE_DATETIME.search(win)
            if dt and dt.start() > 0:
                text = text[dt.start():]
    text = RE_NL3.sub("\n\n", text)
    text = "\n".join(l for l in text.split("\n") if len(l.strip()) > 4 or len(l.strip()) == 0)
    text = text.strip()
    if len(text) > DESC_CAP:
        text = text[:DESC_CAP]
    return text

# ---------------------------------------------------------------- 3. 载入归档、构建金标准样本
pts = json.load(open(f"{WORK}\\orig_points.json", encoding="utf-8"))
samples, seen_urls, dropped = [], set(), []

for p in pts:
    pl = p["payload"]; m = pl.get("metadata", {})
    url = m.get("url", "")
    if url in seen_urls:
        dropped.append((url, "重复URL")); continue
    try:
        out = json.loads(m["out_json"]) if isinstance(m.get("out_json"), str) else m.get("out_json")
    except Exception as e:
        dropped.append((url, f"out_json解析失败:{e}")); continue
    t = str(out.get("新闻标题", "")).strip()
    s = str(out.get("核心摘要", "")).strip().replace("**", "")
    # 契约对齐：阶段1提示词要求「值内引号统一用「」」，DeepSeek 金标准里的 “” 机械归一
    for a, b in (("“", "「"), ("”", "」")):
        t, s = t.replace(a, b), s.replace(a, b)
    raw_content = pl.get("content", "")
    if not t or t == "SKIP" or len(s) < 20 or len(raw_content) < 60:
        dropped.append((url, f"低质 t={len(t)} s={len(s)} c={len(raw_content)}")); continue
    desc = clean_v49(raw_content)
    if len(desc) < 40:
        dropped.append((url, f"清洗后过短 {len(desc)}")); continue
    seen_urls.add(url)
    samples.append({
        "instruction": SYSTEM_PROMPT,
        "input": make_input(str(m.get("title", "") or ""), desc),
        "output": json.dumps({"新闻标题": t, "核心摘要": s}, ensure_ascii=False, separators=(",", ":")),
        "_meta": {"url": url, "domain": urlparse(url).netloc, "level": m.get("level", ""),
                  "type": m.get("type", ""), "gold": True},
    })

# ---------------------------------------------------------------- 4. 合成 SKIP 样本（教会空/404/受限/纯噪音契约）
SKIP_OUT = json.dumps({"新闻标题": "SKIP", "核心摘要": "SKIP"}, ensure_ascii=False, separators=(",", ":"))
skip_cases = [
    ("", ""),
    ("", "404 Not Found"),
    ("", "抱歉，您访问的页面不存在或已被删除。"),
    ("", "访问受限，请在客户端内查看全文"),
    ("", "打开APP 扫码打开虎嗅APP 下载APP"),
    ("", "安全验证 请完成滑动验证后继续访问"),
    ("", "Please enable JavaScript to continue reading."),
    ("", "Sign in to access this article. Subscription required."),
    ("", "Loading..."),
    ("", "首页 最新 栏目 订阅 登录 注册 关于我们 联系我们 广告合作"),
    ("", "验证码拦截：当前访问异常，请稍后重试。acw_sc__v2"),
    ("某公司动态", "无有效内容"),
]
for i, (t, d) in enumerate(skip_cases):
    samples.append({
        "instruction": SYSTEM_PROMPT,
        "input": make_input(t, d),
        "output": SKIP_OUT,
        "_meta": {"url": f"synthetic-skip-{i}", "domain": "synthetic", "level": "", "type": "SKIP", "gold": False},
    })

# ---------------------------------------------------------------- 5. token 预估与预算守卫（4096 上下文，答案不可被右截断）
def est_tokens(s):  # Qwen 系 tokenizer 粗估：中文≈0.75 tok/字，其余≈0.3 tok/字
    zh = sum(1 for c in s if "\u4e00" <= c <= "\u9fa5")
    return int(zh * 0.75 + (len(s) - zh) * 0.3)

TOKEN_BUDGET = 4000
shrunk = 0
for s in samples:
    total = est_tokens(s["instruction"]) + est_tokens(s["input"]) + est_tokens(s["output"]) + 40
    if total > TOKEN_BUDGET:
        # 压缩正文（保尾部输出完整）：按超出比例回切 desc
        over = total - TOKEN_BUDGET
        cut_chars = int(over / 0.75) + 50
        inp = s["input"]
        body_at = inp.find(" 正文: ") + len(" 正文: ")
        tail_at = inp.rfind(USER_SUFFIX)
        body = inp[body_at:tail_at]
        body = body[: max(200, len(body) - cut_chars)]
        s["input"] = inp[:body_at] + body + inp[tail_at:]
        shrunk += 1

lens = [est_tokens(s["instruction"]) + est_tokens(s["input"]) + est_tokens(s["output"]) + 40 for s in samples]

# ---------------------------------------------------------------- 6. 分层切分 train/val（按域名分层，SKIP 单独留 2 条进 val）
random.shuffle(samples)
gold   = [s for s in samples if s["_meta"]["gold"]]
syn    = [s for s in samples if not s["_meta"]["gold"]]
by_dom = {}
for s in gold:
    by_dom.setdefault(s["_meta"]["domain"], []).append(s)

val = syn[:2]
for dom, lst in by_dom.items():
    k = max(1, round(len(lst) * 0.1))
    val += lst[:k]
val_urls = {s["_meta"]["url"] for s in val}
train = [s for s in samples if s["_meta"]["url"] not in val_urls]

# ---------------------------------------------------------------- 7. 落盘
OUT = r"D:\Codes\qwen3.5_0.8b_SFT"
def strip_meta(rows):
    return [{k: v for k, v in s.items() if k != "_meta"} for s in rows]

json.dump(strip_meta(samples), open(f"{OUT}\\dataset.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(strip_meta(train),   open(f"{OUT}\\dataset_train.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(strip_meta(val),     open(f"{OUT}\\dataset_val.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(samples,             open(f"{WORK}\\samples_with_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- 8. 报告
print(f"金标准样本: {len(gold)}  SKIP合成: {len(syn)}  总计: {len(samples)}")
print(f"train {len(train)} / val {len(val)}")
print(f"丢弃: {len(dropped)}")
for d in dropped: print("   drop:", d)
print(f"超预算回切正文: {shrunk} 条")
print(f"token估算(含模板): mean {sum(lens)/len(lens):.0f} / p95 {sorted(lens)[int(len(lens)*0.95)]} / max {max(lens)}")
print("域名分布:", dict(Counter(s["_meta"]["domain"] for s in samples)))
print("等级分布:", dict(Counter(s["_meta"]["level"] for s in gold)))
sl = [len(json.loads(s["output"])["核心摘要"]) for s in gold]
tl = [len(json.loads(s["output"])["新闻标题"]) for s in gold]
print(f"金标准摘要长度: mean {sum(sl)/len(sl):.0f} min {min(sl)} max {max(sl)}")
print(f"金标准标题长度: mean {sum(tl)/len(tl):.0f} min {min(tl)} max {max(tl)}")
