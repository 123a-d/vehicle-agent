"""
Stage 5 Visual Tool Wrapper
文件名：vehicle_tool.py

作用：
把 vehicle_inference.py 中复杂的模型推理，
包装成 Agent 可以直接调用的简单工具函数：

    recognize_vehicle(image_path)
"""

from __future__ import annotations

import argparse
import json

from vehicle_inference import VehicleRecognizer


TOOL_NAME = "recognize_vehicle"

TOOL_DESCRIPTION = (
    "识别车辆图片，返回车型ID、车型名称、车型类型、"
    "置信度、Top-K候选和状态。"
)

# 全局缓存模型对象，避免每次调用都重新加载 checkpoint。
_recognizer: VehicleRecognizer | None = None


def get_recognizer() -> VehicleRecognizer:
    """
    第一次调用时加载模型；
    后续直接复用已经加载好的模型。
    """

    global _recognizer

    if _recognizer is None:
        _recognizer = VehicleRecognizer()

    return _recognizer


def recognize_vehicle(
    image_path: str,
    top_k: int = 3,
    uncertainty_threshold: float = 0.40,
) -> dict:
    """
    识别一张车辆图片。

    参数：
        image_path:
            图片路径。

        top_k:
            返回前K个候选，默认3。

        uncertainty_threshold:
            业务层置信度阈值，默认0.40。

    注意：
        threshold 不会篡改模型原始预测。
        如果 Top-1 confidence 低于阈值，
        只会把 status 标记为 uncertain，
        提醒 Agent 不要过度确定地回答。

    返回：
        一个结构化 dict，后续可直接交给 Agent。
    """

    recognizer = get_recognizer()

    result = recognizer.predict_safe(
        image_path=image_path,
        top_k=top_k,
        uncertainty_threshold=uncertainty_threshold,
    )

    # 底层推理失败时，直接返回结构化错误。
    if not result.get("success", False):
        return {
            "tool_name": TOOL_NAME,
            **result,
        }

    prediction = result["prediction"]

    # Tool 层只暴露稳定、清晰的字段。
    return {
        "tool_name": TOOL_NAME,
        "success": True,
        "status": result["status"],
        "prediction": {
            "class_index": prediction["class_index"],
            "vehicle_id": prediction["vehicle_id"],
            "vehicle_name": prediction["vehicle_name"],
            "vehicle_type": prediction["vehicle_type"],
            "confidence": prediction["confidence"],
        },
        "top_k": result["top_k"],
        "uncertainty_threshold": result["uncertainty_threshold"],
        "warning": result["warning"],
        "capability_note": (
            "该视觉模型主要支持项目定义的50个目标车型；"
            "对于范围外车型，unknown/拒识能力有限。"
            "若 status=uncertain，应避免给出过度确定的车型结论。"
        ),
    }


def main():
    """
    命令行测试：

        python vehicle_tool.py "D:\\car_project\\0009_0030.jpg"

    也可以：
        python vehicle_tool.py "D:\\car_project\\0009_0030.jpg" --top-k 5
        python vehicle_tool.py "D:\\car_project\\0009_0030.jpg" --threshold 0.50
    """

    parser = argparse.ArgumentParser(
        description="Stage 5 Visual Tool 测试"
    )

    parser.add_argument(
        "image_path",
        help="待识别图片路径",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="返回前K个候选，默认3",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="业务置信度阈值，默认0.40",
    )

    args = parser.parse_args()

    result = recognize_vehicle(
        image_path=args.image_path,
        top_k=args.top_k,
        uncertainty_threshold=args.threshold,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
