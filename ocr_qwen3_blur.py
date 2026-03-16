#!/usr/bin/env python3
"""
单阶段土地证 OCR 脚本 - 使用 LM Studio 本地 API 调用 Qwen3-VL 模型
直接从图片中提取土地证信息（用于对比两阶段方法的效果）
"""

import requests
import base64
import json
import re

# ============ 配置区域 ============

# LM Studio 本地 API 地址（OpenAI 兼容）
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

# 模型名称（请修改为你在 LM Studio 中看到的模型名称）
MODEL = "qwen3.5-4b-mlx"

# 图片路径（请修改为你的土地证图片路径）
IMAGE_PATH = "test_blurred.png"

# ==================================


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def build_land_certificate_prompt() -> str:
    """构建土地证识别提示词"""
    prompt = """请识别图片中的土地证信息，提取以下字段：土地使用者、地址、用地面积、建筑占地、用途、四至。

请按以下JSON格式输出（只输出JSON，不要其他内容）：
{
    "土地使用者": "xxx",
    "地址": "xxx",
    "用地面积": "xxx",
    "建筑占地": "xxx",
    "用途": "xxx",
    "四至": "xxx"
}

如果某个字段不存在或无法识别，请使用 null。"""
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
    # 编码图片
    print(f"正在读取图片: {IMAGE_PATH}")
    base64_image = encode_image_to_base64(IMAGE_PATH)

    # 构造请求数据（OpenAI 兼容格式）
    print("正在进行文字识别...")

    prompt = build_land_certificate_prompt()

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 4096
    }

    # 发送请求
    response = requests.post(LM_STUDIO_API_URL, json=payload, timeout=120)

    if response.status_code == 200:
        result = response.json()
        ocr_content = result["choices"][0]["message"]["content"]

        # 输出原始识别结果
        print("\n" + "="*50)
        print("原始识别结果：")
        print("="*50)
        print(ocr_content)

        # 解析 JSON
        print("\n" + "="*50)
        print("解析 JSON 结果...")
        print("="*50)

        parsed_data = parse_json_response(ocr_content)

        if parsed_data:
            print("\n结构化数据：")
            print(json.dumps(parsed_data, ensure_ascii=False, indent=2))
        else:
            print("无法解析 JSON 格式的响应")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
