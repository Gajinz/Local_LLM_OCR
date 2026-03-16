#!/usr/bin/env python3
"""
简单的 OCR 脚本 - 使用 LM Studio 本地 API 调用 Qwen3.5-4B-MLX 模型（关闭思考模式）
"""

import requests
import base64
import json
import re

# ============ 配置区域 ============

# LM Studio 本地 API 地址（OpenAI 兼容）
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

# 你的提示词
PROMPT = "请识别图片中的证书流水号（如41000600056）、不动产权证号（如豫（2017）正阳县-不动产证明0000046号）、证明权利或事项、权利人（申请人）、义务人、坐落、不动产单元号、面积、金额、其他、附记，并按json格式输出。"

# 图片路径（请修改为你的图片路径）
IMAGE_PATH = "test.jpg"

# ==================================

def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def validate_certificate_serial(serial: str) -> str:
    """硬编码规则验证证书流水号"""
    if not serial:
        return "格式异常"

    # 清理空格和换行符
    serial = serial.strip().replace(" ", "").replace("\n", "")

    # 尝试提取连续的数字序列（处理"No 41000600056"或"编号：41000600056"等情况）
    numbers = re.findall(r'\d+', serial)

    # 找到第一个11位或15位的数字
    for num in numbers:
        if len(num) == 11 or len(num) == 15:
            return "格式正常"

    return "格式异常"

def validate_unit_code(unit_code: str) -> str:
    """硬编码规则验证不动产单元号"""
    if not unit_code:
        return "格式异常"

    # 清理空格和特殊字符
    unit_code = unit_code.strip().replace(" ", "").replace("\n", "")

    # 规则：28位（四段式）
    # 通常格式：XXXXXX XXXXXX XX XXXXXXXX（段间可能有空格）
    # 去掉空格后应该是28位数字或字母
    if len(unit_code) == 28 and re.match(r'^[A-Z0-9]{28}$', unit_code):
        return "格式正常"
    else:
        return "格式异常"

def extract_fields(ocr_result: str) -> dict:
    """从OCR结果中提取关键字段"""
    fields = {
        "证书流水号": None,
        "不动产权证号": None,
        "不动产单元号": None
    }

    # 尝试解析 JSON 格式的 OCR 结果
    try:
        data = json.loads(ocr_result)
        fields["证书流水号"] = data.get("证书流水号") or data.get("证书编号")
        fields["不动产权证号"] = data.get("不动产权证号")
        fields["不动产单元号"] = data.get("不动产单元号")
    except:
        # 如果不是 JSON，尝试从文本中提取
        lines = ocr_result.split("\n")
        for line in lines:
            line = line.strip()
            if "证书流水号" in line or "证书编号" in line:
                parts = re.split(r'[:：]', line)
                if len(parts) > 1:
                    fields["证书流水号"] = parts[1].strip()
            elif "不动产权证号" in line or "不动产证明" in line:
                parts = re.split(r'[:：]', line)
                if len(parts) > 1:
                    fields["不动产权证号"] = parts[1].strip()
            elif "不动产单元号" in line:
                parts = re.split(r'[:：]', line)
                if len(parts) > 1:
                    fields["不动产单元号"] = parts[1].strip()

    return fields

def validate_certificate(ocr_result: str) -> dict:
    """验证证书格式"""
    # 提取字段
    fields = extract_fields(ocr_result)

    result = {}

    # 证书流水号：硬编码规则
    result["证书流水号"] = validate_certificate_serial(fields["证书流水号"])

    # 不动产单元号：硬编码规则
    result["不动产单元号"] = validate_unit_code(fields["不动产单元号"])

    return result

def main():
    # 编码图片
    print(f"正在读取图片: {IMAGE_PATH}")
    base64_image = encode_image_to_base64(IMAGE_PATH)

    # 构造请求数据（OpenAI 兼容格式）
    print("正在进行文字识别...")

    payload = {
        "model": "qwen3.5-4b-mlx",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT + " /no_think"},
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
        "max_tokens": 4096,
        # 关闭思考模式的正确方式：使用 chat_template_kwargs
        "chat_template_kwargs": {
            "enable_thinking": False
        },
    }

    # 发送请求
    response = requests.post(LM_STUDIO_API_URL, json=payload, timeout=120)

    if response.status_code == 200:
        result = response.json()
        ocr_content = result["choices"][0]["message"]["content"]

        # 输出识别结果
        print("\n" + "="*50)
        print("识别结果：")
        print("="*50)
        print(ocr_content)

        # 验证证书格式
        print("\n正在进行格式验证...")
        validation_result = validate_certificate(ocr_content)

        print("\n" + "="*50)
        print("格式验证结果：")
        print("="*50)
        print(json.dumps(validation_result, ensure_ascii=False, indent=2))
    else:
        print(f"错误: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
