# OCR 使用说明

## 前置条件

1. 确保已在 LM Studio 中加载 Qwen3-VL-4B-Instruct 模型
2. 在 LM Studio 中启动本地服务器：
   - 打开 LM Studio
   - 点击左侧 "Local Server" 图标
   - 确保 "Server Running" 状态为开启
   - 默认端口为 1234

## 使用方法

1. 准备你要识别的图片，放在当前目录下（或修改脚本中的路径）

2. 修改 `ocr.py` 中的配置：
   ```python
   # 你的提示词
   PROMPT = "请识别图片中的所有文字内容，并按原样输出。"

   # 图片路径
   IMAGE_PATH = "image.jpg"
   ```

3. 运行脚本：
   ```bash
   python3 ocr.py
   ```

## 脚本结构

脚本核心部分只有三个配置项：

- `PROMPT` - 你对模型的提示词
- `IMAGE_PATH` - 要识别的图片路径
- 终端输出 - 识别结果会直接打印在终端

## 注意事项

- 如果 LM Studio 端口不是 1234，请修改 `LM_STUDIO_BASE_URL`
- 确保图片格式支持（jpg, png 等常见格式）
- M1 芯片 16GB 内存运行 Qwen3-VL-4B 应该足够，但可能需要关闭其他占用内存的应用
