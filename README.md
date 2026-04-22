# Deep-Research-Max

一个最小可用的 Gemini Deep Research 命令行客户端，直接调用 Gemini Interactions API。

## 要求

- Python 3.10+
- 已开通 Gemini Deep Research / Interactions API
- 环境变量 `GEMINI_API_KEY` 或 `GOOGLE_API_KEY`

## 用法

```bash
export GEMINI_API_KEY="your-api-key"

python gemini_deep_research.py "请调研 AI Agent 在企业知识库检索中的最佳实践"
```

可选参数：

```bash
python gemini_deep_research.py \
  --agent deep-research-pro-preview-12-2025 \
  --timeout 1800 \
  --poll-interval 15 \
  --json \
  "比较 Gemini Deep Research 和传统 RAG 的优缺点"
```

## 默认行为

- 创建后台 research interaction
- 轮询任务状态直到完成、失败或超时
- 输出最终报告
- 尽可能提取搜索查询、引用链接和推理步数

## 说明

- 默认 API 地址：`https://generativelanguage.googleapis.com`
- 默认 agent：`deep-research-pro-preview-12-2025`
- 如果任务超时，脚本会尝试取消该 interaction
