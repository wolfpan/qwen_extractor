# -*- coding: utf-8 -*-
"""生成 Qwen3.5-0.8B-SFT训练笔记.ipynb（Google Colab T4 + Unsloth）"""
import json

def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# Qwen3.5-0.8B 新闻提取器 SFT 训练笔记（v3 数据集）

**目标**：把 n8n 抓取流「阶段1·提取压缩」的标题/摘要质量从 qwen3.5:0.8b base 提升到接近 DeepSeek-V4.1-flash 水平。

**数据**（本轮 v3，共 112 条 = 100 金标准 + 12 SKIP 合成，train 99 / val 13）：
- 教师信号：`ai_news_bge_orig` 归档库中 DeepSeek 生产真实输出（`out_json` 的 新闻标题/核心摘要 两键）
- 学生输入：同一篇文章经 **v4.9-beta 清洗管线**（实体剥离/APP噪音清理/双锚点截取/3500字截断）后的真实推理分布文本
- 覆盖 huxiu 41 / techcrunch 21（英译中）/ wallstreetcn 17 / tmtpost 14 / qbitai 4 / 36kr 3；含 12 条 SKIP 契约样本
- 契约：输出**恰好 2 键**紧凑 JSON `{"新闻标题":…,"核心摘要":…}`；创建时间由管线生成，模型不输出；值内引号统一「」

**运行环境**：Colab T4 (16GB) ｜ Unsloth LoRA r=16 ｜ fp16 ｜ max_seq_length 4096（与 Ollama numCtx 对齐）｜ response-only loss ｜ eval_loss 择优

**上传文件**：把 `dataset_train.json` 与 `dataset_val.json` 上传到 Colab 工作目录根部（左侧文件面板拖入即可）。

训练方向与数据构建的完整论证见仓库 `SFT方向分析.md`；构建脚本 `work/build_dataset.py` 可随归档库滚动重跑扩量。"""))

cells.append(code("""# 1. 验证 T4 GPU 是否已就绪
!nvidia-smi

# 2. 安装适配 Colab 的 Unsloth 及微调核心依赖
!pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps trl peft accelerate bitsandbytes datasets"""))

cells.append(md("""## 数据装载与体检

样本为 Alpaca 三键 `{instruction, input, output}`：
- `instruction` = test 工作流「LLM阶段1·提取压缩(Qwen0.8B)」节点的 system 提示词（逐字）
- `input` = `输入：标题: {title} 正文: {desc}\\n\\n请严格输出包含 新闻标题、核心摘要 全部2个键的JSON：`（与节点 user 模板逐字对齐）
- `output` = DeepSeek 金标准 2 键紧凑 JSON（或 SKIP）

体检项：output 必须可解析、键集合恰好 `{新闻标题, 核心摘要}`、SKIP 样本数正确。"""))

cells.append(code("""import json

for path in ["dataset_train.json", "dataset_val.json"]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # 体检：输出必须可解析且键集合正确
    bad = 0
    for item in data:
        try:
            o = json.loads(item["output"])
            assert set(o.keys()) == {"新闻标题", "核心摘要"}
        except Exception:
            bad += 1
    n_skip = sum(1 for it in data if '"SKIP"' in it["output"])
    lens = sorted(len(it["input"]) + len(it["output"]) for it in data)
    print(f"{path}: {len(data)} 条就绪 | 坏样本 {bad} | SKIP {n_skip} | 输入+输出字符数 p95={lens[int(len(lens)*0.95)]} max={lens[-1]}")
    assert bad == 0"""))

cells.append(md("""## 训练

要点（与上一轮已验证配方一致）：
1. **16-bit 全精度 LoRA**（不 4bit）——保护数字/百分比敏感度，摘要质量的核心是数字保真
2. **max_seq_length=4096**——与生产 Ollama `numCtx` 一致；v3 数据集 token 估算 max≈2718，答案不会被右截断
3. **ChatML 模板逐字对齐** Modelfile TEMPLATE（system/user/assistant 各带换行）
4. **response-only loss**——0.8B 阶段1 输入是数千字新闻正文，若不屏蔽 prompt 梯度，~85-90% 梯度浪费在复述正文上；此步把梯度集中到 2 键 JSON 输出
5. **eval_loss 择优 + load_best_model_at_end**——100 条小样本极易过拟合，验证集 13 条每 5 步评估一次"""))

cells.append(code("""import torch
import json
from datasets import Dataset
from unsloth import FastLanguageModel
# response-only loss 工具（若版本不符可改 from unsloth import train_on_responses_only）
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. 基础配置：0.8B 16-bit 载入，上下文 4096（与生产 Ollama numCtx 对齐，杜绝答案右截断）
max_seq_length = 4096
dtype = None          # T4 自动 float16
load_in_4bit = False  # 16-bit 全精度训练，保护数字/百分比敏感度
model_name = "Qwen/Qwen3.5-0.8B"

print(f">>> 加载基座: {model_name}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. LoRA：全线性层，r=16
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. ChatML 格式化（与 Modelfile TEMPLATE 逐字对齐：system/user/assistant 各带换行）
chatml_template = \"\"\"<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>\"\"\"

def build_dataset(path):
    with open(path, encoding="utf-8") as f:
        samples = json.load(f)
    texts = []
    for s in samples:
        out = s["output"]
        if isinstance(out, dict):
            out = json.dumps(out, ensure_ascii=False)
        full = chatml_template.format(s["instruction"], s["input"], out) + tokenizer.eos_token
        texts.append({"text": full})
    return Dataset.from_list(texts)

train_dataset = build_dataset("dataset_train.json")
eval_dataset  = build_dataset("dataset_val.json")
print(f">>> train {len(train_dataset)} / val {len(eval_dataset)}")

# 4. SFTTrainer —— 验证集周期评估 + 加载最优 checkpoint（防过拟合）
#    99 样本 × 有效批 8 ≈ 12 步/epoch × 3 epochs ≈ 36 步
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3,
        learning_rate = 2e-4,
        fp16 = True,
        bf16 = False,
        logging_steps = 2,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        eval_strategy = "steps",   # 旧版 transformers 用 evaluation_strategy
        eval_steps = 5,
        save_strategy = "steps",
        save_steps = 5,
        save_total_limit = 2,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
    ),
)

# 5. 只对 assistant 响应算 loss，屏蔽 system+user prompt
#    阶段1输入是数千字正文，不屏蔽则 ~85-90% 梯度耗在复述新闻上；此步把梯度集中到 JSON 输出
#    chr(10) 即换行符，等价于 "<|im_start|>user\\n"
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user" + chr(10),
    response_part = "<|im_start|>assistant" + chr(10),
)

print(">>> 开始微调（response-only loss + 验证集择优）...")
trainer.train()
print(">>> 完成，已加载验证集 eval_loss 最优 checkpoint。")"""))

cells.append(md("""## 离线验收（导出前必跑）

对 val 集抽样生成，对照金标准检查四项验收标准：
1. JSON 可解析、键恰好 2 个
2. 摘要 ≤200 字，无「背景：/数据支撑：/趋势：」脚手架，无「本文/作者」元语言
3. 数字保留（对照金标准里的数字个数）
4. SKIP 判定正确（垃圾文本→SKIP；真实正文→不误判）

生成参数与生产 Ollama 节点一致：temperature 0.2 / repeat_penalty 1.2 / max_new_tokens 500。"""))

cells.append(code("""import json, re
from unsloth import FastLanguageModel

FastLanguageModel.for_inference(model, tokenizer)

with open("dataset_val.json", encoding="utf-8") as f:
    val = json.load(f)

def generate(instruction, user_input, max_new_tokens=500):
    prompt = chatml_template.format(instruction, user_input, "")  # assistant 留空待生成
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    out = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        temperature = 0.2,
        repetition_penalty = 1.2,
        do_sample = True,
        pad_token_id = tokenizer.eos_token_id,
    )
    text = tokenizer.batch_decode(out)[0]
    return text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()

def check(pred_str):
    issues = []
    try:
        o = json.loads(pred_str)
        if set(o.keys()) != {"新闻标题", "核心摘要"}: issues.append("键集合错误")
        s = o.get("核心摘要", "")
        if len(s) > 200: issues.append(f"超长{len(s)}")
        if re.search(r"背景[：:]|数据支撑[：:]|趋势[：:]", s): issues.append("脚手架泄漏")
        if re.search(r"本文|这篇文章|文章指出", s): issues.append("元语言")
        nums = len(re.findall(r"\\d+(?:\\.\\d+)?", s))
        return o, issues, nums
    except Exception as e:
        return None, [f"JSON解析失败:{e}"], 0

n_pass = 0
for i, s in enumerate(val):
    pred = generate(s["instruction"], s["input"])
    gold = json.loads(s["output"])
    o, issues, nums = check(pred)
    ok = (not issues) and (gold["新闻标题"] == "SKIP") == (o is not None and o["新闻标题"] == "SKIP")
    n_pass += ok
    print(f"[{i}] {'PASS' if ok else 'FAIL'} {issues if issues else ''}")
    print(f"    预测标题: {o['新闻标题'][:40] if o else pred[:60]}")
    print(f"    金标题:   {gold['新闻标题'][:40]}")
    print(f"    预测摘要({len(o['核心摘要']) if o else 0}字,{nums}个数字): {(o['核心摘要'][:120] if o else '')}")
    print(f"    金摘要:   {gold['核心摘要'][:120]}")
print(f"\\n>>> 验收: {n_pass}/{len(val)} PASS")"""))

cells.append(md("""## 导出 Q4_K_M GGUF

Unsloth 的 `save_pretrained_gguf` 会自动给目录追加 `_gguf` 后缀，故用基础名导出后动态定位真实产物。正常大小 450~700 MB。"""))

cells.append(code("""import os, glob

export_base = "qwen08b_export"
model.save_pretrained_gguf(
    export_base,
    tokenizer,
    quantization_method = "q4_k_m",
)

def find_gguf(base):
    cands = []
    for d in [base + "_gguf", base]:   # Unsloth 实际目录通常带 _gguf
        if os.path.isdir(d):
            cands += glob.glob(os.path.join(d, "*.gguf"))
    q4 = [p for p in cands if "q4_k_m" in os.path.basename(p).lower()]
    return q4[0] if q4 else (cands[0] if cands else None)

target_file = find_gguf(export_base)
if target_file and os.path.exists(target_file):
    size_mb = os.path.getsize(target_file) / (1024 * 1024)
    print(f"GGUF 生成: {target_file}")
    print(f"大小: {size_mb:.2f} MB (正常 450~700 MB)")
    with open("gguf_path.txt", "w") as f:   # 记录真实路径供下一格使用
        f.write(target_file)
else:
    listing = os.listdir(export_base) if os.path.isdir(export_base) else "目录不存在"
    raise FileNotFoundError(f"未找到 GGUF；{export_base} 实际内容: {listing}")"""))

cells.append(code("""import shutil, os
from google.colab import drive

drive.mount('/content/drive')

# 读取上一格记录的真实 GGUF 路径（不再硬编码 _gguf 目录名）
with open("gguf_path.txt") as f:
    src_path = f.read().strip()
assert os.path.exists(src_path), f"源文件不存在: {src_path}"

dst_path = "/content/drive/MyDrive/qwen3.5_0.8b_news_extractor_v3.gguf"
print(f">>> 复制 {src_path} -> {dst_path}")
shutil.copy(src_path, dst_path)
size_mb = os.path.getsize(dst_path) / (1024 * 1024)
print(f">>> 已复制，Drive 端大小 {size_mb:.2f} MB")

# 强制刷盘，防止 FUSE 缓冲截断（204MB 坑）；刷盘后再卸载
print(">>> 强制刷盘中，请勿关闭页面...")
drive.flush_and_unmount()
print("刷盘完成！请前往 Drive 确认大小 500MB+ 后再下载。")"""))

cells.append(md("""## 部署（回到本地 n8n 环境）

1. GGUF 放入 Ollama models 目录，创建 Modelfile（TEMPLATE 必须与训练 ChatML 逐字一致）：

```
FROM ./qwen3.5_0.8b_news_extractor_v3.gguf
TEMPLATE \"\"\"<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"
PARAMETER temperature 0.2
PARAMETER num_ctx 4096
PARAMETER num_predict 500
PARAMETER repeat_penalty 1.2
```

2. `ollama create qwen3.5-0.8b-news-v3 -f Modelfile`
3. test 工作流「Ollama Qwen3.5-0.8B」节点：model 换成 `qwen3.5-0.8b-news-v3`，其余参数（think:false / format:json / temperature 0.2 / numCtx 4096 / numPredict 500 / repeatPenalty 1.2）**保持不变**
4. 在线 A/B 跑 1–2 天：对比 `ai_news_bge_test` 新增点 vs `ai_news_bge_orig` 同 URL DeepSeek 输出；监控「解析阶段1输出」的 `_need_cloud_retry` 率应显著下降

## 下一轮迭代（数据飞轮）

- `ai_news_bge_orig` 为滚动归档（生产每轮自动写入原始输入+DeepSeek输出）：每周重跑 `work/build_dataset.py` 增量扩充训练集后重训
- 失败回流：test 流触发云端重提的实录（`_fail_reason` 非空）+ DeepSeek 重提结果 = 天然的困难样本对，优先补入下一版
- 若摘要长度想收紧到 120 字契约：过滤金标准 >180 字样本重训，或同步改阶段1提示词预算"""))

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
out = r"D:\Codes\qwen3.5_0.8b_SFT\Qwen3.5-0.8B-SFT训练笔记.ipynb"
json.dump(nb, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("written:", out, len(cells), "cells")
