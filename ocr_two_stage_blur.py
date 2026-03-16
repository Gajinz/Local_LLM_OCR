#!/usr/bin/env python3
"""
两阶段 OCR 脚本
- 第一阶段：使用 PaddleOCR-VL-1.5 提取图片中的文字
- 第二阶段：使用 Qwen3-VL-4B 将文字整理为结构化 JSON

使用说明：
1. 在 LM Studio 中加载 PaddleOCR-VL-1.5 模型
2. 在 LM Studio 中加载 Qwen3-VL-4B 模型（或使用其他小模型）
3. 启动 Local Server
4. 修改下方配置后运行：python3 ocr_two_stage.py
"""

import requests
import base64
import json
import re

# ============ 配置区域 ============

# LM Studio 本地 API 地址（OpenAI 兼容）
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

# 第一阶段：OCR 模型配置
OCR_MODEL = "paddleocr-vl-1.5"  # 请修改为你在 LM Studio 中看到的模型名称
OCR_PROMPT = ""

# 第二阶段：结构化模型配置
STRUCTURE_MODEL = "qwen/qwen3-vl-4b"  # 请修改为你在 LM Studio 中看到的模型名称

# 需要提取的字段（用于第二阶段的提示词）
FIELDS_TO_EXTRACT = [
    "土地使用者",
    "地址",
    "用地面积",
    "建筑占地",
    "用途",
    "四至"
]

# 图片路径（请修改为你的图片路径）
IMAGE_PATH = "test_blurred.png"

# ==================================


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_vlm(model: str, prompt: str, image_base64: str = None, temperature: float = 0) -> str:
    """调用 VLM 模型（OpenAI 兼容格式）"""
    # 构建消息内容
    content = [{"type": "text", "text": prompt}]

    # 如果有图片，添加图片内容
    if image_base64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        })

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": temperature,
        "max_tokens": 4096,
        #"enable_thinking": False,  # 关闭思考模式，注释掉此行可启用思考
    }

    response = requests.post(LM_STUDIO_API_URL, json=payload, timeout=180)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API 错误 {response.status_code}: {response.text}")


def build_structure_prompt(ocr_text: str) -> str:
    """构建结构化提取的提示词"""
    fields_str = "、".join(FIELDS_TO_EXTRACT)

    prompt = f"""请从以下OCR识别结果中提取以下字段：{fields_str}

OCR识别结果：
---
{ocr_text}
---

请按以下JSON格式输出（只输出JSON，不要其他内容）：
{{
    "土地使用者": 人名用逗号隔开,
    "地址": "xxx",
    "用地面积":"xxx",
    "建筑占地": "xxx",
    "用途": "xxx",
    "四至": "xxx",
}}

如果某个字段不存在或无法识别，请使用 null。
"""
    return prompt


def parse_json_response(response: str) -> dict:
    """尝试解析 JSON 响应"""
    # 尝试提取 JSON 块
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


def main():
    print("=" * 60)
    print("两阶段 OCR 识别，先paddle提取文字，然后qwen整理为结构化数据")
    print("=" * 60)

    # 编码图片
    print(f"\n[1/3] 正在读取图片: {IMAGE_PATH}")
    base64_image = encode_image_to_base64(IMAGE_PATH)

    # ========== 第一阶段：OCR 文字识别 ==========
    print(f"\n[2/3] 阶段一：使用 {OCR_MODEL} 识别文字...")
    print("-" * 40)

    try:
        ocr_result = call_vlm(OCR_MODEL, OCR_PROMPT, base64_image, temperature=0)
        print("OCR 识别结果：")
        print(ocr_result)
    except Exception as e:
        print(f"第一阶段失败: {e}")
        print("\n请确认：")
        print("1. PaddleOCR-VL-1.5 模型已在 LM Studio 中加载")
        print("2. 模型名称配置正确（查看 LM Studio Local Server 的模型列表）")
        return

    # ========== 第二阶段：结构化提取 ==========
    print("\n" + "=" * 60)
    print(f"[3/3] 阶段二：使用 {STRUCTURE_MODEL} 提取结构化数据...")
    print("-" * 40)

    structure_prompt = build_structure_prompt(ocr_result)

    try:
        # 第二阶段只需要传文本，不需要传图片
        # 因为 OCR 结果已经是文本了
        structure_result = call_vlm(
            STRUCTURE_MODEL,
            structure_prompt,
            image_base64=None,  # 第二阶段不需要图片
            temperature=0
        )
        print("结构化提取结果：")
        print(structure_result)
    except Exception as e:
        print(f"第二阶段失败: {e}")
        print("\n请确认：")
        print("1. Qwen3-VL-4B 模型已在 LM Studio 中加载")
        print("2. 模型名称配置正确")
        return

    # ========== 解析和验证 ==========
    print("\n" + "=" * 60)
    print("解析 JSON 结果...")
    print("-" * 40)

    parsed_data = parse_json_response(structure_result)

    if parsed_data:
        print("\n结构化数据：")
        print(json.dumps(parsed_data, ensure_ascii=False, indent=2))
    else:
        print("无法解析 JSON 格式的响应")
        print("原始响应：")
        print(structure_result)

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
