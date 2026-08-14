# config/

對手定義與凍結的評估對手池。**進版控。**

```
config/
├── ladder.json         凍結的對手池（docs/division-of-labor.md §6）
├── sweep-hire.json     參數掃描的例子（對手直接寫在檔案裡）
└── opponents/
    └── <name>.json     一個對手一個檔
```

## 對手檔格式

兩種寫法。**引擎內建的**：

```json
{
  "name": "starter",
  "builtin": "starter",
  "note": "隨便寫，只給人看"
}
```

**我們自己的 agent**：

```json
{
  "name": "gen0-no-hire",
  "entry": "agents.gen0:act",
  "params": {"max_hands": 0}
}
```

- `entry` 是 `module:attr`，路徑從 repo 根目錄算
- `params` 會當第三個位置引數傳給 `entry`，也就是 `act(obs, config, params)`。
  空的話直接呼叫 `entry(obs, config)`
- `name` 省略時用檔名

## 對手池檔格式

`opponents` 的每個元素可以是**名字**（去 `opponents/` 查檔），也可以是**完整的
spec dict**（直接寫在檔案裡，不用另外開檔）。兩種可以混用：

```json
{
  "name": "hire-sweep",
  "opponents": [
    "gen0",
    {"name": "hire-2",  "entry": "agents.gen0:act", "params": {"max_hands": 2}},
    {"name": "hire-12", "entry": "agents.gen0:act", "params": {"max_hands": 12}}
  ]
}
```

一個檔就是一組參數掃描。名字重複會直接報錯 —— 不然結果表分不出誰是誰。

## 怎麼用

```bash
python -m eval.runner --a gen0 --b starter                  # config/opponents/starter.json
python -m eval.runner --a gen0 --b config/opponents/x.json  # 直接給路徑
python -m eval.runner --a gen0 --ladder                     # 打整個 config/ladder.json
python -m eval.runner --a gen0 --ladder config/sweep-hire.json   # 換一個池
```

`--a` / `--b` 的解析順序：檔案路徑 → `config/opponents/<name>.json` → 內建後備表
→ `module:attr`。

## 規則

🚨 **`ladder.json` 裡既有對手的參數不准改。** 改了之後新舊結果就沒得比，
`tests/baselines.json` 也跟著作廢。要試新參數就**新增**一個對手檔。

對手池要凍結 4~6 個（`docs/division-of-labor.md` §6）。`frozen` 欄位記凍結日期。
