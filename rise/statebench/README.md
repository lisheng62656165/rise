# StateTrace on StateBench

本目录包含 **Paper-aligned StateTrace-EDS-ECA**、**OAgents 官方 ORM Best-of-4（StateBench 适配）**、Vanilla 对照，以及完整 StateBench 代码、任务、初始环境和评分器。复制整个目录即可安装运行，不需要另行下载 StateBench、OAgents 或其他研究仓库。官方 ORM prompt 和许可证已包含在 `third_party/OAgents/`。

需要 Python **3.12**、安装 `requirements.txt`、联网访问模型 API，以及你自己的 API key。无需 GPU。“完整”指代码和数据自包含，并非免安装 Python 依赖或免费调用模型。本包仅包含 **StateBench，不包含 AppWorld**。

## 给 Codex 的指令

引用本 README 后可以说：

> 按 README 使用我的 OpenAI-compatible API 模型运行 StateBench。参考结果配置使用 deepseek-v4.1-flash；密钥已配置到 OPENAI_API_KEY，服务 base URL 已配置到 MIMO_BASE_URL。先安装依赖并跑三任务 smoke，再跑完整 Test，使用 generation seeds 20260921 20260925 20260929 20261003 20261007、selector seeds 77121 77122 77123 77124 77125、judge seeds 88021 88022 88023 88024 88025；完整运行 workers 100、selector-workers 100、score-workers 40。比较 Vanilla、修改后的官方 ORM Best-of-4 和 Paper-aligned StateTrace-EDS-ECA，汇总 Task Completion pass@1、pass^5 和 UX。不要改变算法，不要输出密钥；成功任务不重跑。

本包支持 ChatGPT/OpenAI API 以及其他 **OpenAI-compatible Chat Completions API**。历史参考表使用 `deepseek-v4.1-flash`；报告中的原服务 endpoint 未公开，精确 endpoint replay 需要原实验 provider 配置。其他模型也可以运行同一流程，但会产生新实验结果，不能当作复现下方 DeepSeek 结果。请提供确切的 API 模型 ID，而不只是“ChatGPT”；ChatGPT 网页订阅不代替 API 额度。模型需要支持 Chat Completions、工具调用和足够长的上下文；本包不保证任意模型都支持同一组请求参数。

### Codex 执行约定

如果用户只说“用 ChatGPT 模型跑 StateBench”，Codex 应读取本 README 和 `AGENTS.md`，然后执行以下流程：

1. 确认确切的 API 模型 ID、密钥来源和 base URL；已有配置则复用，缺少时只询问缺失项，不猜测模型，不让用户把密钥粘贴进对话。历史 endpoint 不在公开包中，不能据此推断用户的服务地址。
2. 在本目录建立 Python 3.12 虚拟环境，并安装 `requirements.txt`。**不要 git clone、下载数据集、下载模型权重或引用原作者的实验目录**；所需 StateBench 数据、代码、prompt 和评分器都在本目录。Python 和 pip 软件依赖不是随包的二进制运行环境，首次安装仍需要网络或本机已有的软件包缓存。
3. 按上面的用户授权范围运行：需要 `pass^5` 时使用五个不同 seed 的完整 Test；若用户只要求一组，只跑一组并将 `pass^5` 标为 N/A。不要因为要统计该指标就擅自追加付费批次。
4. 先在独立目录运行三题真实 API smoke；成功后执行下面对应的完整命令。完整 Test 不使用 `--tasks-per-domain 1`，不缩减150题，不改成 Train/Dev，不把模拟 API 测试结果当成真实模型结果。
5. 完成生成、选择、Task/UX 评分后，读取 `comparison.md`、`comparison.json` 和各方法 `*_metrics.json`，向用户报告三种方法的 `pass^5`、`Task Completion pass@1`、`UX`、完成数量和实际模型/seed。遇到错误按相同输出目录续跑，不跳过失败题后报告完整结果。

本 README 提供的是 **API 实验**入口，不会自动登录或操作 ChatGPT 网页。没有 API 密钥或所选模型权限时，应说明缺少的条件，不能声称实验已启动。

## 安装

在本目录打开终端。Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "YOUR_API_KEY"
$env:MIMO_BASE_URL = "YOUR_PROVIDER_BASE_URL"
$env:MIMO_TEMPERATURE = "0"
$env:MIMO_MAX_TOKENS = "8192"
$env:MIMO_SELECTOR_MAX_TOKENS = "8192"
$env:MIMO_STREAMING = "1"
$env:MIMO_ORM_SELECTOR_STREAMING = "0"
```

若系统没有 `py` 启动器，但 `python --version` 已是 3.12，则第一行改用 `python -m venv .venv`。PowerShell 禁止激活脚本时，不必改系统策略：直接以 `.\.venv\Scripts\python.exe` 替代后续命令中的 `python`。

Linux/macOS：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_API_KEY"
export MIMO_BASE_URL="YOUR_PROVIDER_BASE_URL"
export MIMO_TEMPERATURE="0"
export MIMO_MAX_TOKENS="8192"
export MIMO_SELECTOR_MAX_TOKENS="8192"
export MIMO_STREAMING="1"
export MIMO_ORM_SELECTOR_STREAMING="0"
```

密钥仅放环境变量或仓库外的私有文件，不要提交到 GitHub。`.env.example` 是变量说明，程序**不会自动加载** `.env`。如果同时存在 `MIMO_API_KEY`，它优先于 `OPENAI_API_KEY`；切换服务商时需清理旧变量。

## 一条命令运行 / Reproduce

以下命令展示历史参考配置。PowerShell 与 bash 的续行语法不同；README 给出 bash 兼容的单行命令。将 `MIMO_BASE_URL` 设置为你获准使用的 OpenAI-compatible 服务根地址，路径应止于 `/v1`，不要填 `/chat/completions`。

```bash
# 无付费 API 的本地测试：真实环境 + 本地模拟模型响应（含完整150题流程）
python -X utf8 -B -m pytest -q
python -X utf8 -B verify_release.py

# 三任务真实 API smoke：每领域 1 个，不是正式 Test
python -X utf8 run_experiment.py --model deepseek-v4.1-flash --base-url "$MIMO_BASE_URL" --seeds 20260921 --selector-seeds 77121 --judge-seeds 88021 --tasks-per-domain 1 --workers 3 --selector-workers 3 --score-workers 3 --output-dir outputs/smoke

# 五次完整 Test：与历史 DeepSeek 结果相同的 seed 配置，自动统计三项指标
python -X utf8 run_experiment.py --model deepseek-v4.1-flash --base-url "$MIMO_BASE_URL" --seeds 20260921 20260925 20260929 20261003 20261007 --selector-seeds 77121 77122 77123 77124 77125 --judge-seeds 88021 88022 88023 88024 88025 --workers 100 --selector-workers 100 --score-workers 40 --output-dir outputs/test_five
```

需要单次运行时传入任一 generation/selector/judge seed，`pass^5` 会标为 N/A。`--selector-seeds` 和 `--judge-seeds` 如使用，必须与 `--seeds` 一一对应；不传时分别复用 scalar seed。示例目录独立，不会自动从另一输出目录借用结果。请求预算为 generation/EDS selector 8192 tokens、ORM selector 2048 tokens、StateBench Task/UX judge 各 16384 tokens（由随包评分器显式设置）；Generation/EDS 用流式响应，ORM selector/评分用非流式响应。历史并发为生成/选择 100、评分 40；如果 provider 限额不足，应调低 `--workers`、`--selector-workers`、`--score-workers`，这会改变吞吐，云模型仍可能有输出差异。默认按 seed 顺序执行每一轮，每个阶段内部并发；不是同时启动五批。

程序依次执行：共享 Vanilla A → EDS-ECA → Best-of-4 → 三种方法离线评分 → 汇总。重复**相同命令和输出目录**可续跑；成功轨迹、阶段 checkpoint、评分不会重复请求。修改模型、seed 或评分设置请用新目录，不要同时向同一目录启动多个进程。失败会以非零状态退出；修好连接后再执行原命令。

### 历史参考结果

以下为五轮 `deepseek-v4.1-flash` Test 的已测结果。ORM 行来自对已有相同四候选池执行官方 ORM prompt 的重选；不是旧的自定义 Best-of-4 selector 数字。

| 方法 | Task Completion pass@1 | pass^5 | 平均 UX |
| --- | ---: | ---: | ---: |
| Vanilla | 489/750 = 65.20% | 58/150 = 38.67% | 3.9709 |
| 修改后的官方 ORM Best-of-4 | 541/750 = 72.13% | 70/150 = 46.67% | 4.1428 |
| Paper-aligned StateTrace-EDS-ECA | 545/750 = 72.67% | 73/150 = 48.67% | 4.1499 |

Generation seeds 为 `20260921, 20260925, 20260929, 20261003, 20261007`；selector seeds 为 `77121–77125`；judge seeds 为 `88021–88025`。这些是历史观察值，不是测试夹具输出，也不保证云端模型重跑逐位相同。报告未公开原服务 endpoint；换模型、judge 或 provider 后应作为新实验报告。

## 数据与算法

三个领域是 **customer_support、shopping_assistant、travel**。每领域 train/dev/test 为 **70/30/50**，总计 **210/90/150**，全部 450 个任务及初始环境已包含。`--split test` 是入口默认值；不要为结果更好而删任务。划分文件见 `benchmark/state_bench/domains/*/splits/`。

```text
Paper-aligned StateTrace-EDS-ECA:
Vanilla A -> accepted incumbent
  -> 公开事件分层 + frontier/事件信用指导 -> fresh B'
  -> 存在 material disagreement 时模型二选一 -> accepted B
  -> 更新确定性事件信用 -> fresh C' -> 二选一 -> accepted C
  -> 更新确定性事件信用 -> fresh D' -> 二选一 -> 最终 incumbent

OAgents 官方 ORM Best-of-4:
4 条独立 fresh 轨迹 -> 固定 c1/c2/c3/c4 顺序
  -> 原文 ORM_list_wise.yaml + 公开完整对话
  -> 普通 JSON {index, analysis, ...} 四选一 -> 最终轨迹
```

EDS-ECA **不把四条一起选**，最终可能来自 A/B'/C'/D'；无实质公开事务分歧则保留 incumbent。事件信用来自公开事件与选中/未选中轨迹的差异，不是隐藏任务得分，也不是额外训练的 reward model。两种 selector 都调用所配置的模型，而非关键词加权打分器。

同一 seed 为 `s` 时：Vanilla A 用 `s`；EDS 三个 proposal 用 `s,s+1,s+2`；Best-of-4 用 `s,s+1,s+2,s+3` 并复用同一 A 作为首候选。所有新候选均从独立初始环境执行，不拼接数据库状态。Selector 基础 seed 默认 `77113`；EDS 使用 `base + split任务序号*3 + 阶段号(1..3)`，Best-of-4 使用 `base + 完整split任务key排序序号(从0起)`。judge seed 默认 `88002`。云 API 的 seed 不保证逐字复现。

在线生成、事件分析和选择只读取公开轨迹、公开工具 schema，不读取 hidden requirements、gold actions、最终 state diff 或评分。初始环境与隐藏要求只由 benchmark harness/离线 evaluator 使用。

**复现口径**：Best-of-4 现为本项目修正后的 `oagents_official_orm_listwise_statebench_v1`，不是旧版自定义 prompt。官方原文 prompt 不附加领域 system prompt、不注入 EDS 事件/信用、不随机打乱候选；selector 使用 temperature=0、max_tokens=2048、普通 Chat Completion，无 selector tool call。格式失败至多3次同输入重试，耗尽后报错，不默认选 c1。StateBench 对话序列化和 API 后端属于适配，未声称复现 OAgents 原论文的 CodeAgent 环境或表格。EDS 的“Paper-aligned”指本项目的二选一＋确定性公开事件信用版本，**不是原始 MiMo joint-selection/event-credit 版本**。详见 [来源说明](THIRD_PARTY_NOTICES.md)。

**旧输出迁移**：不要将旧 Best-of-4 已选结果或其评分放进新运行目录。配置记录 selector 协议版本，直接 runner 也会拒绝旧 selector final，避免将旧结果当成本版结果。历史目录不修改；需要重选已有候选时见 [REPRODUCE.md](REPRODUCE.md)。

## 评分与结果

默认 agent、user simulator、selector、离线 Task/UX judge 使用同一个配置模型；三种方法使用相同评分设置。使用随包 StateBench **v0.8.1 / UX v27** 评分代码；更换 judge 模型会改变数值，不等同于官方排行榜固定模型协议，也不保证重现历史 MIMO/Nemotron 数字。

每个输出目录生成 `comparison.md`、`comparison.json` 和 `{vanilla,eds_eca,best4}_metrics.json`，后者包含分 seed 结果。

| 指标 | 统计定义 |
| --- | --- |
| Task Completion pass@1 | 每任务最终选中轨迹的完成度；五次运行时为全部 750 条评分的均值 |
| pass^5 | 同一方法、同一任务五次**全部**成功的任务数 / 150；不是 pass@5，也不是 Best-of-4 的候选 oracle |
| UX | 官方 `ux_score` 的均值，范围 1–5；v27 为 `max(1, 0.5*user_control + 0.3*user_effort + 0.2*response_density - resource_penalty)` |
| State Req. / Task Req. | 对应官方二值 requirement 指标的均值，详见各方法 JSON |

只有恰好五次完整同任务集评分才输出 `pass^5`。缺一个任务不会用剩余交集做正式分母。Smoke 明确标为 `partial`，不输出 `pass^5`。API 错误不当成已完成的算法失败；任务正常执行但没有完成目标则计 0。

已评分目录也可单独汇总，**每次只能传同一种方法的多个 seed**：

```bash
python -X utf8 aggregate_metrics.py --method eds_eca --run 42=outputs/test_five/seed_42/scores/eds_eca --run 142=outputs/test_five/seed_142/scores/eds_eca --run 242=outputs/test_five/seed_242/scores/eds_eca --run 342=outputs/test_five/seed_342/scores/eds_eca --run 442=outputs/test_five/seed_442/scores/eds_eca --output outputs/test_five/eds_eca_metrics.json
```

```text
outputs/test_five/
  config.json                       # 配置，不含密钥
  seed_42/
    vanilla/                        # A
    eds_eca/                        # 最终 incumbent + stage_checkpoints/
    best4/candidates/candidate_0..3/ # 四个原始候选
    best4/final/                    # 四选一结果
    scores/vanilla|eds_eca|best4/    # 离线评分
  seed_142/ ...
  comparison.md                     # 最终三方法指标对比
```

一次完整对比共享 A，通常需 **150 × 7 = 1050 条完整 rollout**；五次为 5250 条，另有模拟用户、selector 和评分调用。实际成本与时长取决于模型、轨迹长度、重试和服务商吞吐，可能需要数小时至数天。单独看每种方法预算为每任务 4 条，不是 7 条。Token 成本不能由 rollout 数直接换算。

## 模型兼容与源码

历史变量名 `MIMO_*` 不限制模型品牌。某些服务不支持 `seed`、`temperature` 或 `max_tokens`；仅在确认服务要求后显式设置 `MIMO_SEND_SEED=0`、`MIMO_SEND_TEMPERATURE=0`、`MIMO_TOKEN_PARAMETER=max_completion_tokens`。这些配置会记录到运行配置中，关闭 seed 时不能声称服务端 seeded determinism。不使用自动降级悄悄换模型/参数。长 selector 响应可配置 `MIMO_SELECTOR_MAX_TOKENS`；不支持工具调用的模型不能运行此协议。

可选 `MIMO_API_KEYS_FILE` 指向仓库外每行一个 key 的私有文件，新客户端轮流取 key。瞬时网络错误有界重试，但**不会自动更换失效账号，也不能绕过服务商额度限制**。完整变量说明见 [.env.example](.env.example)。

| 文件 | 职责 |
| --- | --- |
| `run_experiment.py` | 全流程入口，三方法生成、评分、聚合 |
| `run_statetrace_eds_eca_paper.py` | A/B'/C'/D'、二选一、checkpoint |
| `state_trace_eds_ec.py` | ECA 信用、frontier、二选一 prompt/解析 |
| `state_trace_event_scaling.py`, `state_trace_eds_r.py`, `trace_ir.py` | 公开事件、层级表示、事务分歧 |
| `run_oagents_best_of4.py`, `oagents_official_orm.py` | 独立候选、官方 ORM 四选一与同输入重试 |
| `third_party/OAgents/ORM_list_wise.yaml` | 官方原文 prompt（无需联网下载） |
| `state_trace_pool_select.py` | EDS 二选一所用公开事件 packet；旧 Best4 helper 不再由正式入口调用 |
| `phase0_runner.py`, `phase0_score.py`, `aggregate_metrics.py` | Vanilla、官方离线评分、统计 |
| `benchmark/` | 随包 StateBench 代码/数据/原始许可证 |

验证范围见 [VERIFICATION.md](VERIFICATION.md)。测试通过不代表新模型科学结果已复现，更不保证 EDS-ECA 高于 Best-of-4。发布前不要上传 `outputs/`、私有日志、密钥或虚拟环境。许可证见 [LICENSE](LICENSE) 和 [第三方说明](THIRD_PARTY_NOTICES.md)。
