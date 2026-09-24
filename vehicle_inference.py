"""
Stage 5 核心推理模块：vehicle_inference.py

功能：
1. 加载 Stage 4 最终 ResNet18 checkpoint
2. 使用与验证阶段一致的图片预处理
3. 对单张图片进行推理
4. 返回 Top-K、confidence、车型 ID、车型名称/类型
5. 兼容当前 class_info.json 顶层为 list 的实际结构
6. 返回统一 dict，后续可直接包装成 Agent Tool
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms
from config import CHECKPOINT_PATH as CONFIG_CHECKPOINT_PATH


# ============================================================
# 1. 路径与固定配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = CONFIG_CHECKPOINT_PATH

CLASS_INFO_PATH = PROJECT_ROOT / "class_info.json"

NUM_CLASSES = 51
UNKNOWN_INDEX = 50
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


# ============================================================
# 2. 推理预处理
# ============================================================

def build_transform():
    """
    推理阶段必须和验证阶段保持一致。

    流程：
        Resize(256)
        -> CenterCrop(224)
        -> ToTensor()
        -> Normalize(ImageNet mean/std)
    """

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


# ============================================================
# 3. JSON 读取
# ============================================================

def load_json(path: Path, required: bool = False) -> Any:
    """
    读取 JSON。

    注意：不再要求顶层必须是 dict，
    因为当前 class_info.json 实际上是 list。
    """

    if not path.exists():
        if required:
            raise FileNotFoundError(f"找不到文件：{path}")
        return {}

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON 格式错误：{path}\n{exc}"
        ) from exc


# ============================================================
# 4. 整理 class_info.json
# ============================================================

def normalize_class_info(raw_class_info: Any) -> dict[str, dict]:
    """
    当前 class_info.json 实际结构大概是：

        [
            {"id": "0000", ...},
            {"id": "0001", ...},
            ...
        ]

    为了后续能直接：
        class_info["0009"]

    这里统一转换成：

        {
            "0000": {...},
            "0001": {...},
            ...
        }
    """

    if raw_class_info is None:
        return {}

    if isinstance(raw_class_info, list):
        normalized = {}

        for item in raw_class_info:
            if not isinstance(item, dict):
                continue

            vehicle_id = item.get("id")
            if vehicle_id is None:
                continue

            normalized[str(vehicle_id)] = item

        return normalized

    # 以后即使文件被改成 dict，也继续兼容。
    if isinstance(raw_class_info, dict):
        normalized = {}

        for key, value in raw_class_info.items():
            if isinstance(value, dict):
                normalized[str(key)] = value
            else:
                normalized[str(key)] = {"value": value}

        return normalized

    raise ValueError(
        "class_info.json 顶层必须是 list 或 dict。"
    )


# ============================================================
# 5. checkpoint 类别映射
# ============================================================

def build_index_to_class(
    class_to_idx: dict,
) -> dict[int, str]:
    """
    训练时 ImageFolder 保存的是：

        "0009" -> 0
        "0080" -> 1
        ...
        "unknown" -> 50

    推理时需要反过来：

        0 -> "0009"
        1 -> "0080"
        ...
        50 -> "unknown"
    """

    if not class_to_idx:
        raise ValueError(
            "checkpoint 中没有 class_to_idx。"
        )

    index_to_class = {
        int(index): str(class_name)
        for class_name, index
        in class_to_idx.items()
    }

    if len(index_to_class) != NUM_CLASSES:
        raise ValueError(
            "checkpoint 类别数量不是 51。"
        )

    if index_to_class.get(UNKNOWN_INDEX) != "unknown":
        raise ValueError(
            "checkpoint 中 index 50 不是 unknown。"
        )

    return index_to_class


# ============================================================
# 6. 车型名称清洗
# ============================================================

def extract_text_after_double_equal(
    value: Any,
) -> str | None:
    """
    class_info.json 里的部分字段类似：

        中文名称==Volvo_S90
        中文类型==Sedan####Upper Mid-size Sedan

    当前 PowerShell 中中文有乱码，
    但 == 后面的英文通常仍然可读。

    所以这里优先提取 == 后面的部分。
    如果没有 ==，就直接返回原字符串。
    """

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if "==" in text:
        _, right = text.split("==", 1)
        right = right.strip()
        if right:
            return right

    return text


def get_vehicle_info(
    vehicle_id: str,
    class_info: dict[str, dict],
) -> dict[str, Any]:
    """
    根据 vehicle_id 从 class_info 中获取车型名称和类型。

    当前真实字段：
        id
        name_from_new
        name_from_old
        type_info
    """

    if vehicle_id == "unknown":
        return {
            "vehicle_name": "unknown",
            "vehicle_type": "unknown",
        }

    info = class_info.get(vehicle_id)

    if info is None:
        return {
            "vehicle_name": vehicle_id,
            "vehicle_type": None,
        }

    # 优先使用 name_from_old，
    # 因为它通常保留了 == 后面的英文名称。
    vehicle_name = (
        extract_text_after_double_equal(
            info.get("name_from_old")
        )
        or extract_text_after_double_equal(
            info.get("name_from_new")
        )
        or vehicle_id
    )

    vehicle_type = extract_text_after_double_equal(
        info.get("type_info")
    )

    return {
        "vehicle_name": vehicle_name,
        "vehicle_type": vehicle_type,
    }


# ============================================================
# 7. 推理类
# ============================================================

class VehicleRecognizer:
    """
    车型识别推理器。

    创建一次对象时加载模型；
    之后可以连续预测很多张图片。
    """

    def __init__(
        self,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        class_info_path: str | Path = CLASS_INFO_PATH,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.class_info_path = Path(class_info_path)

        # 自动选择设备。
        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.transform = build_transform()

        # 读取并整理 class_info.json。
        raw_class_info = load_json(
            self.class_info_path,
            required=False,
        )

        self.class_info = normalize_class_info(
            raw_class_info
        )

        # 加载模型。
        (
            self.model,
            self.index_to_class,
            self.checkpoint,
        ) = self._load_model()

    # --------------------------------------------------------
    # 8. 加载模型
    # --------------------------------------------------------

    def _load_model(self):
        """
        checkpoint 主要保存训练好的参数。

        推理时仍然需要：
            创建 ResNet18
            -> 把 fc 改成 51 类
            -> load_state_dict()
            -> model.eval()
        """

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"找不到 checkpoint：{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if checkpoint.get("num_classes") != NUM_CLASSES:
            raise ValueError(
                "checkpoint 的 num_classes 不是 51。"
            )

        model = models.resnet18(
            weights=None
        )

        # 原始 ImageNet ResNet18 是 1000 类，
        # 当前项目需要 51 类。
        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(self.device)

        # 切换为推理模式。
        model.eval()

        index_to_class = build_index_to_class(
            checkpoint.get("class_to_idx", {})
        )

        return (
            model,
            index_to_class,
            checkpoint,
        )

    # --------------------------------------------------------
    # 9. 图片合法性检查
    # --------------------------------------------------------

    def _validate_image(
        self,
        image_path: str | Path,
    ) -> Path:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"图片不存在：{path}"
            )

        if not path.is_file():
            raise ValueError(
                f"不是文件：{path}"
            )

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"暂不支持该图片格式：{path.suffix}"
            )

        return path

    # --------------------------------------------------------
    # 10. 图片预处理
    # --------------------------------------------------------

    def _prepare_image(
        self,
        image_path: Path,
    ) -> torch.Tensor:
        """
        transform 后单张图片形状：
            [3, 224, 224]

        unsqueeze(0) 后：
            [1, 3, 224, 224]

        前面的 1 表示当前 Batch 只有 1 张图。
        """

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)

        except (
            UnidentifiedImageError,
            OSError,
        ) as exc:
            raise ValueError(
                f"图片无法读取或已损坏：{image_path}"
            ) from exc

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    # --------------------------------------------------------
    # 11. index -> 车型完整信息
    # --------------------------------------------------------

    def _class_result(
        self,
        class_index: int,
        confidence: float,
    ) -> dict:
        vehicle_id = self.index_to_class[
            class_index
        ]

        info = get_vehicle_info(
            vehicle_id=vehicle_id,
            class_info=self.class_info,
        )

        return {
            "class_index": class_index,
            "vehicle_id": vehicle_id,
            "vehicle_name": info["vehicle_name"],
            "vehicle_type": info["vehicle_type"],
            "confidence": round(
                float(confidence),
                6,
            ),
        }

    # --------------------------------------------------------
    # 12. 核心 predict()
    # --------------------------------------------------------

    def predict(
        self,
        image_path: str | Path,
        top_k: int = 5,
        uncertainty_threshold: float | None = None,
    ) -> dict:
        """
        单张图片推理。

        核心数据流：
            image
            -> model
            -> logits
            -> softmax
            -> topk
            -> index 映射
            -> dict
        """

        if not (1 <= top_k <= NUM_CLASSES):
            raise ValueError(
                "top_k 必须在 1~51 之间。"
            )

        if (
            uncertainty_threshold is not None
            and not (
                0.0 <= uncertainty_threshold <= 1.0
            )
        ):
            raise ValueError(
                "uncertainty_threshold 必须在 0~1 之间。"
            )

        path = self._validate_image(image_path)
        batch_tensor = self._prepare_image(path)

        # ====================================================
        # Stage 5 最核心的几行：
        # ====================================================
        with torch.inference_mode():
            # [1,3,224,224] -> [1,51]
            logits = self.model(batch_tensor)

            # 51 个 logits -> 51 个相对概率。
            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            # 找出概率最高的前 K 个类别。
            (
                top_confidences,
                top_indices,
            ) = probabilities.topk(
                k=top_k,
                dim=1,
            )

        # 去掉 Batch 维度并转成 Python list。
        top_confidences = (
            top_confidences[0]
            .cpu()
            .tolist()
        )

        top_indices = (
            top_indices[0]
            .cpu()
            .tolist()
        )

        # index -> 车型信息。
        top_k_results = [
            self._class_result(
                class_index=int(class_index),
                confidence=float(confidence),
            )
            for class_index, confidence
            in zip(
                top_indices,
                top_confidences,
            )
        ]

        # Top-1 是模型原始预测。
        prediction = top_k_results[0]

        # 业务层状态，不偷偷修改原始预测。
        status = "recognized"
        warning = None

        if prediction["vehicle_id"] == "unknown":
            status = "unknown"
            warning = "模型将该图片预测为 unknown。"

        elif (
            uncertainty_threshold is not None
            and prediction["confidence"]
            < uncertainty_threshold
        ):
            status = "uncertain"
            warning = (
                "最高置信度低于业务阈值，"
                "结果应谨慎使用。"
            )

        return {
            "success": True,
            "image_path": str(path.resolve()),
            "device": str(self.device),
            "status": status,
            "prediction": prediction,
            "top_k": top_k_results,
            "uncertainty_threshold": (
                uncertainty_threshold
            ),
            "warning": warning,
        }

    # --------------------------------------------------------
    # 13. 给 Agent / Tool 使用的安全版本
    # --------------------------------------------------------

    def predict_safe(
        self,
        image_path: str | Path,
        top_k: int = 5,
        uncertainty_threshold: float | None = None,
    ) -> dict:
        """
        predict() 出错会抛异常，适合开发调试。

        predict_safe() 会把错误也包装成 dict，
        更适合未来 Tool Calling。
        """

        try:
            return self.predict(
                image_path=image_path,
                top_k=top_k,
                uncertainty_threshold=(
                    uncertainty_threshold
                ),
            )

        except Exception as exc:
            return {
                "success": False,
                "image_path": str(image_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }


# ============================================================
# 14. 命令行测试入口
# ============================================================

def main():
    """
    示例：

        python vehicle_inference.py "D:/car_project/0009_0030.jpg"

        python vehicle_inference.py "D:/car_project/0009_0030.jpg" --top-k 3

        python vehicle_inference.py "D:/car_project/0009_0030.jpg" --threshold 0.40
    """

    parser = argparse.ArgumentParser(
        description="Stage 5 车型识别推理模块"
    )

    parser.add_argument(
        "image_path",
        help="待识别图片路径",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回前 K 个候选，默认5",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="可选业务置信度阈值，例如0.40",
    )

    args = parser.parse_args()

    recognizer = VehicleRecognizer()

    result = recognizer.predict_safe(
        image_path=args.image_path,
        top_k=args.top_k,
        uncertainty_threshold=args.threshold,
    )

    # dict -> JSON 风格输出。
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
