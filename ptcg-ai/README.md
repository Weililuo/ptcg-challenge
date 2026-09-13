# ptcg-ai

PTCG 对局数据处理、行为克隆训练与 CABT 对局运行工具。仓库统一管理代码、牌组、训练配置和发布模型；原始数据、处理后数据及训练过程文件均保留在本地。

## 目录

~~~text
ptcg-ai/
├─ configs/              可选训练配置
│  ├─ base/
│  └─ experts/
├─ datasets/             本地数据入口与说明
├─ decks/                牌组文件
├─ models/               可直接使用的发布模型
├─ src/                  数据、模型、训练与测试实现
├─ workspace/            可删除的过程文件，不进入 Git
├─ base.py               通用模型训练启动器
├─ expert.py             专家模型训练启动器
└─ run.py                单场对局启动器
~~~

## 安装

建议使用 Python 3.11。在仓库根目录打开 PowerShell：

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

<code>requirements.txt</code> 固定了本项目当前使用的 PyTorch 与 Kaggle Environments 版本。若需要特定 CUDA 版本，请先按 PyTorch 官方安装方式安装对应的 <code>torch</code>，再安装其余依赖。

## 准备数据

1. 在 <code>datasets</code> 下创建 <code>raw</code> 文件夹。
2. 将 Kaggle CABT 回放 ZIP 放入 <code>datasets/raw</code>；程序会递归查找 ZIP，无需手动解压。
3. 生成统一数据集：

~~~powershell
python src/base_data.py --workers 4
~~~

<code>--workers</code> 控制并行处理的压缩包数量，默认值为 <code>2</code>；<code>--shard-size</code> 控制每个分片包含的对局数，默认值为 <code>2048</code>。首次运行会补齐所需数据目录。生成结果保存在 <code>datasets/processed</code>、<code>datasets/manifests</code> 和 <code>datasets/metadata</code>。

数据目录、产物和专家筛选方式详见 [datasets/README.md](datasets/README.md)。

## 训练通用模型

最简命令：

~~~powershell
python base.py base_v1
~~~

训练器读取统一数据集，训练结束后自动在测试集上评估，并写入：

~~~text
models/base/base_v1/model.pt
models/base/base_v1/record.json
~~~

可以在 <code>configs/base/base_v1.json</code> 保存可复用配置，也可以通过 <code>--config</code> 指定其他 JSON：

~~~json
{
  "device": "cuda",
  "model": {
    "d_model": 192,
    "nhead": 6,
    "layers": 4,
    "dim_feedforward": 768,
    "dropout": 0.1
  },
  "training": {
    "batch_size": 32,
    "accumulate": 4,
    "epochs": 3,
    "learning_rate": 0.0003
  },
  "test": {
    "batch_size": 64
  }
}
~~~

~~~powershell
python base.py base_v1 --config configs/base/base_v1.json
~~~

未写出的配置项使用程序默认值。

## 训练专家模型

专家模型从一个通用模型继续训练。只需提供专家模型名称、牌组、通用模型、专家数据名称及核心卡牌条件：

~~~powershell
python expert.py dragapult_v1 `
  --deck decks/dragapult.csv `
  --base base_v1 `
  --data dragapult `
  --card 119:2 `
  --card 121:2
~~~

<code>--card 卡牌ID:最少数量</code> 可以重复使用。若对应专家数据视图不存在，启动器会先根据统一数据集创建视图；视图只保存索引，不复制基础训练数据。

也可以将参数写入 <code>configs/experts/dragapult_v1.json</code>：

~~~json
{
  "deck": "decks/dragapult.csv",
  "base": "base_v1",
  "data": "dragapult",
  "cards": {
    "119": 2,
    "121": 2
  },
  "device": "cuda",
  "training": {
    "epochs": 1,
    "learning_rate": 0.00005
  }
}
~~~

~~~powershell
python expert.py dragapult_v1
~~~

专家模型保存到 <code>models/experts/dragapult_v1</code>，并在 <code>record.json</code> 中记录牌组、基础模型、筛选条件、数据集和评估结果。

## 第二轮强化学习

v2 模型从已经完成行为克隆训练的 v1 模型开始，通过多进程 CABT 对局和 PPO Actor-Critic 继续学习。奖励包含胜负奖励及小幅奖赏卡差增量奖励。

强化学习配置是扁平 JSON。例如 <code>configs/base/base_v2.json</code>：

~~~json
{
  "rollout_workers": 4,
  "games_per_iteration": 32,
  "batch_size": 256,
  "update_epochs": 4
}
~~~

训练通用 v2 模型：

~~~powershell
python src/learn.py base_v2 models/base/base_v1 `
  --config configs/base/base_v2.json `
  --device cuda
~~~

训练专家 v2 模型：

~~~powershell
python src/learn.py froslass_v2 models/experts/froslass_v1 `
  --config configs/experts/froslass_v2.json `
  --device cuda
~~~

通用模型会在仓库牌组间训练；专家模型及专家对手始终使用各自 <code>record.json</code> 中指定的牌组。Rollout 进程使用 CPU，主进程使用 GPU 更新模型。过程检查点保存在 <code>workspace/learn</code>，完成后的模型仍发布到 <code>models/base</code> 或 <code>models/experts</code>。

## 继续中断的训练

~~~powershell
python base.py base_v1 --resume
python expert.py dragapult_v1 --resume
python src/learn.py base_v2 models/base/base_v1 `
  --config configs/base/base_v2.json `
  --device cuda `
  --resume
~~~

继续训练依赖 <code>workspace/train/.../last.pt</code> 或 <code>workspace/learn/.../last.pt</code>。删除 <code>workspace</code> 不影响已经发布的 <code>model.pt</code> 及对局运行，但会失去继续未完成训练的检查点。

## 运行对局

每个牌组文件应包含恰好 60 行整数卡牌 ID。模型参数既可以是模型目录，也可以直接指向 <code>model.pt</code>：

~~~powershell
python run.py `
  models/experts/dragapult_v1 decks/dragapult.csv `
  models/base/base_v1 decks/opponent.csv
~~~

程序自动选择 CUDA 或 CPU，完成一场 CABT 对局并在仓库根目录生成 <code>result.html</code>。也可以显式指定设备：

~~~powershell
python run.py model_a deck_a.csv model_b deck_b.csv --device cuda
~~~

批量模拟不会生成 HTML，而是将 player1 的胜、负、平和胜率写入其模型目录中的 <code>record.json</code>：

~~~powershell
# 模拟 100 局
python run.py model1 deck1.csv model2 deck2.csv --batch

# 模拟 1000 局
python run.py model1 deck1.csv model2 deck2.csv --batch --large
~~~

批量模拟会交替双方座位；<code>--large</code> 单独使用时也会模拟 1000 局。

模型目录与记录格式详见 [models/README.md](models/README.md)。

## 文件管理约定

- <code>datasets/raw</code>、<code>processed</code>、<code>manifests</code>、<code>metadata</code> 均为本地数据，不进入 Git。
- <code>workspace</code> 只存放分片中间结果、训练检查点等过程产物；发布模型不依赖它。
- 每个发布模型只保留 <code>model.pt</code> 和 <code>record.json</code>。
- <code>result.html</code> 是可随时重新生成的单场对局回放，不进入 Git。
