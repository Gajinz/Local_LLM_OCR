# 1. 模型的选择

- 显存16G可部署：12B、14B模型
- 视觉能力模型：qwen3.5全系
- 潦草手写字体识别方案：小尺寸OCR模型(PaddleOCR/GLMOCR)+小参数纯语言模型

# 2. 怎么保证小尺寸模型输出的结构性

- 核心工具/规范：JSON Schema
- 参考文档：LM Studio官方文档（[https://lmstudio.ai/docs/developer/openai-compat/structured-output）](https://lmstudio.ai/docs/developer/openai-compat/structured-output）)

# 3. 输出慢/显存不足怎么办

- 有N卡优化部署工具：VLLM、SG Lens、Tensor RT
- 无N卡部署工具：Ollama、LM Studio

