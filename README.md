# qwen_extractor

把 **Qwen3.5-0.8B** 微调成一个「科技新闻标题/摘要提取器」，替换 n8n 抓新闻管线里的大模型，省钱提速。

## 这个项目在干嘛

我有一条 n8n 管线每天抓科技新闻，其中「读正文 → 写标题 + 摘要」这一步原来交给 DeepSeek，效果好但慢、烧 token。这个项目教会一个 0.8B 的小模型只干这一件事：

- **输入**：清洗过的新闻正文（可能带原标题）
- **输出**：只有 2 个键的 JSON——

```json
{"新闻标题":"15-30字核心事件标题","核心摘要":"120字内单段摘要，保留精确数字、单位、日期"}
```

- 正文是垃圾（空 / 404 / 付费墙 / 纯导航噪音）时，输出 `{"新闻标题":"SKIP","核心摘要":"SKIP"}`，让管线跳过。

小模型干粗活，干砸了管线的校验器会兜底转云端大模型重提。SFT 的目标就是把「事后兜底」内化成模型本能，降低重提率。

## 训练数据从哪来（这是本项目的核心）

**不是网上爬的，也不是手写的。** 生产管线每天跑，会自动把「原始正文 + DeepSeek 输出」归档进 Qdrant（`ai_news_bge_orig` 集合）。我从归档库取了 **100 条真实配对**当金标准，再合成 **12 条 SKIP** 样本，共 112 条：

| 文件 | 内容 |
|---|---|
| `dataset.json` | 全量 112 条 |
| `dataset_train.json` | 99 条训练集 |
| `dataset_val.json` | 13 条验证集（按域名分层抽样，seed=3407） |

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
| `work/build_dataset.py` | 数据构建脚本，可重跑。归档库是滚动的，**每周重跑一次就能增量扩充训练集**（数据飞轮） |
| `work/orig_points.json` | 从 Qdrant 拉的原始归档 dump |
| `work/samples_with_meta.json` | 构建中间产物（带域名/等级元信息） |

## 训练配方一句话

Colab T4 + Unsloth + LoRA（r=16 全线性层，fp16 不量化——保护数字敏感度），`max_seq_length=4096` 与 Ollama `num_ctx` 对齐，**loss 只算在回答上**（不让模型学着复述正文），3 epochs ≈ 39 步，`eval_loss` 择优加载防过拟合，导出 Q4_K_M GGUF（约 517MB）。

## 效果怎么样

离线验收（13 条验证集 + 四项标准）：**8 条 PASS**。3 条 FAIL 全是「输出纪律」问题（多吐了一个键 / JSON 结束后没停住），不是理解或摘要质量问题——下一版（v4）主攻方向就是它：加困难负例、收紧停止条件。

## 部署（Ollama + n8n）

模型下载：`qwen3.5_0.8b_news_extractor_v3.gguf`（HuggingFace 链接见下方 · 待上传）

⚠️ **必须手写 Modelfile**。Unsloth 导出时不会给这个模型内嵌对话模板，Ollama 默认模板对不上会导致输出崩坏。TEMPLATE 必须与训练时的 ChatML 逐字一致：

```
FROM ./qwen3.5_0.8b_news_extractor_v3.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.2
PARAMETER num_ctx 4096
PARAMETER num_predict 500
PARAMETER repeat_penalty 1.2
```

```bash
ollama create qwen_extractor_v3 -f Modelfile
```

n8n 阶段1节点只换 model 名，`think:false`、`format:json` 保持不变。

## 怎么继续迭代（数据飞轮）

1. 生产管线每跑一轮，归档库自动积累新的「正文 ↔ 金标准」配对
2. 每周重跑 `work/build_dataset.py` → 数据集自动变大
3. 线上触发云端重提的失败案例（`_fail_reason`），其重提结果天然是「小模型错 → 金标准对」的困难样本，优先补入下一版

## 已知局限

- 金标准只有 100 条，长尾版式（如 36kr 快讯只有 3 条样本）泛化有限，靠飞轮慢慢补
- SKIP 样本全部是合成的，边界行为可能偏保守，上线后观察误判率再校准
- 归档窗口只有 3 天，题材有时效同质（比如黄仁勋事件占了 15 条）
