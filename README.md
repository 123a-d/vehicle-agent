# Vehicle Agent

Vehicle Agent 是一个面向细粒度车型识别的本地智能体项目。系统以 ResNet18 完成 50 个目标车型与 1 个 `unknown` 类的图像分类，并通过 DeepSeek Tool Calling 把视觉识别、车型资料查询、识别记录保存和历史查询组合成可对话的 Agent。FastAPI 提供 Session、图片上传和聊天接口，单页 Web 前端负责完整交互。

> 仓库只包含源码、说明和类别元数据。数据集、模型权重、上传图片、SQLite 运行数据和 API Key 均不会提交。

## 整体架构

```text
Browser (web/index.html)
        |
        v
FastAPI (api/main.py)
        |
        +-- Session 管理 (session_agent.py)
        |       |
        |       +-- DeepSeek Tool Calling
        |       +-- 会话消息与最近图片上下文
        |
        +-- 4 个 Agent Tools
                +-- recognize_vehicle
                +-- query_vehicle_info
                +-- save_recognition_record
                +-- query_recognition_history
                         |
                         +-- SQLite (database/vehicle_agent.db)

recognize_vehicle -> ResNet18 -> 51 类预测与 Top-K 结果
```

## 主要功能

- 上传车辆图片并校验格式、大小和可解码性。
- 对 50 个目标车型进行细粒度识别，同时保留 `unknown` 类和低置信度提示。
- 返回车型 ID、名称、类型、置信度和 Top-K 候选。
- 由 LLM 根据自然语言选择工具并组织最终回答。
- 以 Session 隔离多轮对话和最近上传图片。
- 使用 SQLite 查询车型资料、保存识别结果并读取会话历史。
- 通过 FastAPI 同时提供 API、交互文档和 Web 页面。

## 四个 Agent 工具

| 工具 | 作用 | 主要实现 |
|---|---|---|
| `recognize_vehicle` | 加载最终模型并识别车辆图片 | `vehicle_tool.py` |
| `query_vehicle_info` | 按车型 ID 查询名称、类型和子类型 | `vehicle_info_tool.py` |
| `save_recognition_record` | 把识别结果写入当前 Session 的历史 | `recognition_record_tool.py` |
| `query_recognition_history` | 查询当前 Session 最近的识别记录 | `recognition_history_tool.py` |

工具 Schema 和统一调度位于 `agent_tools.py`，多工具循环位于 `multi_tool_agent.py`，带上下文的会话封装位于 `session_agent.py`。

## 技术栈

- Python 3.10+
- PyTorch、TorchVision、ResNet18
- NumPy、scikit-learn、Pillow、Matplotlib
- OpenAI Python SDK（连接 DeepSeek 的 OpenAI 兼容接口）
- FastAPI、Uvicorn、Pydantic、python-multipart
- SQLite
- HTML、CSS、原生 JavaScript

## 模型训练结果

验证集采用 51 类 Macro-F1 作为主要指标：

| 实验 | Macro-F1 |
|---|---:|
| Baseline | 约 0.6017 |
| 解冻 Layer4 微调 | 约 0.8239 |
| Weighted Cross Entropy（最终模型） | 约 0.8417 |

最终 Weighted CE 模型的 `unknown` F1 约为 0.2056。项目曾尝试置信度阈值、更多样的 Unknown 样本、Hard Mining、特征原型距离和独立 Known/Unknown Gate。实验显示，提高 Unknown Recall 往往会增加目标车型误拒识，因此最终以整体 51 类 Macro-F1 最优的 Weighted CE 模型作为交付版本。

## 本地运行

### 1. 创建环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如需使用 CUDA，请根据本机 CUDA 版本从 PyTorch 官方渠道安装匹配的 `torch` 和 `torchvision`，再安装其余依赖。

### 2. 准备最终模型

模型权重不在 Git 仓库中。请把最终 checkpoint 放到：

```text
outputs/unknown_weighted/best_macro_f1.pth
```

checkpoint 需包含模型 `state_dict` 与训练时的 `class_to_idx`，共 51 类，并保证索引 50 对应 `unknown`。

### 3. 初始化 SQLite

```powershell
python database/init_database.py
python database/import_vehicle_info.py
```

这会在本地生成 `database/vehicle_agent.db`。数据库文件包含运行数据，已被 Git 忽略。

### 4. 配置 DeepSeek

在当前 PowerShell 会话中设置 API Key：

```powershell
$env:DEEPSEEK_API_KEY="你的 API Key"
```

可选配置：

```powershell
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-v4-flash"
```

如需重新运行数据构建或 Unknown 实验，请把原始数据根目录配置为：

```powershell
$env:VEHICLE_DATA_ROOT="D:\your_data_directory"
```

不要把真实 Key 写入源码或提交 `.env`。`.env.example` 仅展示变量名称；当前程序直接读取进程环境变量，并不会自动加载 `.env`。

### 5. 启动 FastAPI 与 Web

在项目根目录运行：

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

打开：

- Web：<http://127.0.0.1:8000/app>
- API 文档：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

## Session / SQLite / FastAPI / Web 流程

1. Web 调用 `POST /sessions` 创建内存 Session。
2. 用户上传图片后，API 校验文件并保存到本地 `uploads/<session_id>/`。
3. Web 发送聊天消息；Session 将最近图片路径加入 Agent 上下文。
4. DeepSeek 决定是否调用一个或多个工具。
5. 视觉工具加载本地 checkpoint 并返回结构化预测。
6. Agent 可继续查询车型资料、保存结果或读取历史。
7. SQLite 持久化车型资料和识别记录，最终自然语言回答返回 Web。

Session 本身保存在进程内存中；服务重启后 Session 会失效，但已写入 SQLite 的记录仍保留。

## 项目结构

```text
car_project/
|-- api/
|   `-- main.py                     # FastAPI、Session、上传与聊天接口
|-- database/
|   |-- init_database.py            # 建表
|   `-- import_vehicle_info.py      # 导入 50 个目标车型资料
|-- docs/                            # 项目文档
|-- scripts/
|   |-- stage_01_02_data/           # 数据集构建与类别选择
|   |-- stage_03_baseline/          # Baseline 训练与评估
|   `-- stage_04_optimization/      # 微调、Weighted CE 与 Unknown 实验
|-- web/
|   `-- index.html                  # 单页 Web 前端
|-- agent_tools.py                  # 四工具 Schema 与调度器
|-- multi_tool_agent.py             # 多轮 Tool Calling 循环
|-- session_agent.py                # Session 上下文封装
|-- vehicle_inference.py            # ResNet18 核心推理
|-- vehicle_tool.py                 # 视觉工具包装
|-- vehicle_info_tool.py            # 车型资料查询
|-- recognition_record_tool.py      # 识别记录写入
|-- recognition_history_tool.py     # 历史查询
|-- class_info.json                 # 车型公开元数据
|-- requirements.txt
`-- .env.example
```

以下本地目录由 `.gitignore` 排除，但现有运行目录中的内容不会被删除：`dataset_51/`、`unknown_train_v2/`、`unknown_train_v3_hard/`、`outputs/`、`uploads/`、`.venv/` 以及 SQLite 数据库。

## 已知局限

- 模型主要覆盖项目定义的 50 个目标车型；对范围外车型的 Unknown/拒识能力有限。
- `unknown` 是由多个非目标车型合并形成的高类内差异类别，F1 明显低于目标车型平均水平。
- 低置信度阈值只用于业务提示，不会修改模型原始 Top-1 预测。
- 首次视觉调用需要加载本地模型，CPU 环境可能较慢。
- Session 使用进程内存存储，不适合直接用于多进程或分布式生产部署。
- 当前 SQLite 适合单机演示；公开部署前还需补充身份验证、速率限制、持久化 Session 和更严格的上传隔离。
