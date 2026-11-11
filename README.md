# qwen_extractor

把 **Qwen3.5-0.8B** 微调成一个「科技新闻标题/摘要提取器」，替换 n8n 抓新闻管线里的 DeepSeek——最主要的动因是**省钱**。

## 这个项目在干嘛

我有一条 n8n 管线，平均每天处理 150 条科技新闻，其中「读正文 → 写标题 + 摘要」这一步原来交给 DeepSeek：缓存命中率只有 50%~60%，开销以输出 token 为主，一天 1.5 元左右，一年下来 400 多元——而且这还都是按 DeepSeek 闲时折扣价跑的。于是我决定换成本地部署。

但负责跑模型的笔记本已经将近 10 年了：R5-2500U + 32GB DDR4-2400 内存，跑 3B 以上的模型无望。所以干脆训练一个**专门用于新闻提取和摘要的 0.8B 模型**，做 4bit 量化（Q4_K_M，约 517MB），部署在 Ollama 上——纯 CPU 也能跑出 30~50 token/s，完全满足管线长期在后台提取的性能要求。

这个项目教会这个小模型只干这一件事：

- **输入**：清洗过的新闻正文（可能带原标题）
- **输出**：只有 2 个键的 JSON——

```json
{"新闻标题":"15-30字核心事件标题","核心摘要":"120字内单段摘要，保留精确数字、单位、日期"}
```

- 正文是垃圾（空 / 404 / 付费墙 / 纯导航噪音）时，输出 `{"新闻标题":"SKIP","核心摘要":"SKIP"}`，让管线跳过。

小模型干粗活，干砸了管线的校验器会兜底转云端大模型重提。SFT 的目标就是把「事后兜底」内化成模型本能，降低重提率。

## 训练数据从哪来（这是本项目的核心）

**不是网上爬的，也不是手写的。** 生产管线每天跑，会自动把「原始正文 + DeepSeek 输出」归档进 Qdrant（`ai_news_bge_orig` 集合，append-only 滚动积累）。从归档库取真实配对当金标准，再合成少量 SKIP 样本：

| 版本 | 文件 | 条数 | 说明 |
|---|---|---|---|
| v3 | `data/dataset*.json` | 112 = 99 train / 13 val | 首版：100 条金标准 + 12 条合成 SKIP，训练出 v3 模型 |
| v4 | `data/dataset_v4*.json` | 167 = 150 train / 17 val | `--live` 在线全量拉归档库构建（飞轮已转一圈） |

格式是 Alpaca 三键 `{instruction, input, output}`。`instruction` 和 `input` 是从 n8n 生产节点里**逐字提取**的提示词，不是重新写的。

两个关键保真操作（细节见 `SFT方向分析.md`）：

1. **训练输入 == 推理输入**：归档正文只经过旧版清洗，而线上 0.8B 看到的是新版（v4.9）清洗后的文本。`work/build_dataset.py` 把生产工作流的清洗正则逐条移植了过来，保证模型「学的」和「用的」是同一个分布。
2. **输出契约归一**：金标准里的弯引号 `“”` 统一替换成提示词声明的 `「」`，教模型遵守自己声明的契约。

## 文件导览

| 文件 | 说明 |
|---|---|
| `SFT方向分析.md` | 项目的「为什么」：数据盘点、0.8B 六大失败模式诊断、7 条方向决策、验收标准。**建议先看这个** |
| `Qwen3_5_0_8B_SFT训练笔记V2.ipynb` | **权威训练笔记**，Colab 免费 T4 从头跑到尾（装环境 → 训练 → 验收 → 导出 GGUF） |
| `Qwen3.5-0.8B-SFT训练笔记.ipynb` | V1 初版（验收格有个 API bug，仅存档） |
| `data/` | 全部数据集：v3（冻结）与 v4（滚动），Alpaca 三键格式 |
| `work/build_dataset.py` | 数据构建脚本。默认用本地 dump 重建 v3；加 `--live` 参数在线全量拉 Qdrant 归档库构建 v4（不覆盖 v3 冻结文件）。**每周跑一次 `--live` 就能增量扩充训练集**（数据飞轮） |
| `work/orig_points*.json` | 从 Qdrant 拉的原始归档 dump（v3 离线版 / v4 在线版） |
| `work/samples_with_meta*.json` | 构建中间产物（带域名/等级元信息） |

## 训练配方一句话

Colab T4 + Unsloth + LoRA（r=16 全线性层，fp16 不量化——保护数字敏感度），`max_seq_length=4096` 与 Ollama `num_ctx` 对齐，**loss 只算在回答上**（不让模型学着复述正文），3 epochs ≈ 39 步，`eval_loss` 择优加载防过拟合，导出 Q4_K_M GGUF（约 517MB）。

## 效果怎么样

- **v3**：离线验收（13 条验证集 + 四项标准）**8 条 PASS**。3 条 FAIL 全是「输出纪律」问题（多吐了一个键 / JSON 结束后没停住）——v4 的主攻方向。
- **v4**：离线验收（17 条验证集）**15/17 PASS**。v3 的三类失败（SKIP 空输出、Extra data、JSON 骨架损坏）全部消失——复盘确认那是 repeat_penalty=1.2 惩罚 ChatML 骨架 token 造成的解码假象，模型本身没有此问题；剩余 2 条 FAIL 均为摘要超长（201 字 / 254 字），根源是金标准教师数据自带超长尾部，模型忠实学到了分布，属数据侧遗留。

## 部署（Ollama + n8n）

模型下载：[Phanai/qwen_extractor](https://huggingface.co/Phanai/qwen_extractor)（`qwen3.5_0.8b_news_extractor_v4.gguf`，Q4_K_M，约 517MB；上一版 v3 仍可下载）：

```bash
# 方式一：直接下载
curl -LO https://huggingface.co/Phanai/qwen_extractor/resolve/main/qwen3.5_0.8b_news_extractor_v4.gguf
# 方式二：Ollama 直接拉（拉完仍需下方 Modelfile 重套模板，模型没内嵌模板）
ollama pull hf.co/Phanai/qwen_extractor
```

⚠️ **必须手写 Modelfile**。Unsloth 导出时不会给这个模型内嵌对话模板，Ollama 默认模板对不上会导致输出崩坏。TEMPLATE 必须与训练时的 ChatML 逐字一致：

```
FROM ./qwen3.5_0.8b_news_extractor_v4.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.2
PARAMETER num_ctx 4096
PARAMETER num_predict 500
PARAMETER repeat_penalty 1.05
```

⚠️ **repeat_penalty 用 1.0–1.05，别沿用 v3 时代的 1.2**：1.2 会惩罚 ChatML 骨架 token，在 llama.cpp / Ollama 上可能诱发 JSON 骨架损坏或 SKIP 误判（`format:json` 只挡得住 Extra data）——这是 v4 验收复盘的结论。

```bash
ollama create qwen_extractor_v4 -f Modelfile
```

n8n 阶段1节点只换 model 名，`think:false`、`format:json` 保持不变。

## 怎么继续迭代（数据飞轮）

1. 生产管线每跑一轮，归档库自动积累新的「正文 ↔ 金标准」配对
2. 每周跑 `python work/build_dataset.py --live` 在线全量拉归档库 → 数据集自动变大（v3 → v4 已经从 112 涨到 167）
3. 线上触发云端重提的失败案例（`_fail_reason`），其重提结果天然是「小模型错 → 金标准对」的困难样本，优先补入下一版

## 已知局限

- v3 金标准只有 100 条（v4 已滚动到 167），长尾版式（如 36kr 快讯样本极少）泛化有限，靠飞轮慢慢补
- SKIP 样本全部是合成的，边界行为可能偏保守，上线后观察误判率再校准
- 归档窗口初期只有 3 天，题材有时效同质（比如黄仁勋事件占了 15 条），滚动再采集自然缓解
