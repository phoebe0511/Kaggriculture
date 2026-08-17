# k_cells — 貼進 Kaggle notebook 的驗證格

每支檔案就是一個 notebook cell，照編號順序貼、順序跑。
目的**只有驗證**，不是產生 submission —— submission 是本機
`python -m serving.build_submission` 產生的 `submission/`。

## 流程

```
本機  python -m serving.build_submission     →  submission/（四支平面模組）
      把 submission/ 整包上傳成 Kaggle Dataset
Kaggle  cell 01 → 07 → 10 → 20
```

| cell | 做什麼 | 失敗代表 |
|---|---|---|
| `01_env_check.py` | 引擎版本 + 12 條規則指紋 + 對局設定 | Kaggle 的引擎跟本機不同，本機所有測試數字作廢 |
| `07_assemble_agent.py` | 從 `/kaggle/input` 找出四支檔案複製到 `/kaggle/working`，並擋掉未攤平版本 | Dataset 沒掛上，或上傳到 repo 版而不是 `submission/` 版 |
| `10_notebook_smoke.py` | **用 Kaggle 實際的載入方式**跑三局完整對局 | agent 在真實載入路徑下起不來 |
| `11_diag_trace.py` | cell 10 是 DONE 但現金異常時才跑：逐日狀態 vs 送出的動作 | — |
| `20_presubmit_check.py` | 單回合耗時尖峰 vs `actTimeout` | 會超時被判失敗 |

## 為什麼 cell 10 傳檔案路徑而不是 import

`env.run([agent_fn, "starter"])` 用的是 notebook 裡已經 import 好的物件，
走的是 notebook 的 `sys.path`。評分時走的是另一條：

```python
# kaggle_environments/agent.py:51-59
exec_dir = os.path.dirname(path)
sys.path.append(exec_dir)          # 接在最後面，不是最前面
exec(compile(raw, path, "exec"), {})
```

兩條路徑會給出不同結果。本機就是因為改用傳路徑才抓到
**`agents` 這個 top-level 名字被佔走**：
`site-packages/kaggle_environments/envs/lux_ai_s3/` 底下有一支 `agents.py`，
它在 `sys.path` 的位置比我們的目錄前面（實測 index 9 vs 10），
跨 entry 是前面的贏，所以 `import agents` 抓到它，然後炸在它自己的相對 import。

錯誤訊息還會被包成看不出原因的樣子：

```
InvalidArgument: Invalid raw Python:
  ImportError('attempted relative import with no known parent package')
```

`agents/__init__.py` 救不了 —— 「正規 package 優先於同名 `.py`」只在
**同一個 `sys.path` entry 內**成立。所以 submission 攤平成四支平面模組。

## 本機基準（引擎 1.32.7，對手 `starter`）

cell 10 會拿這組數字對照。**同 seed、同引擎版本，結果應該完全一致** ——
Kaggle 的 CPU 只影響速度。對不上就是引擎版本或檔案版本不同。

| seed | 期末現金 |
|---|---|
| 41001 | 91,397 |
| 41002 | 94,200 |
| 41003 | 78,386 |

本機時間（cell 20 的量法，seed 41003）：

```
平均 5.4 ms   p99 12.4 ms   尖峰 52.8 ms   餘裕 19.0x（actTimeout = 1 秒）
整局 6.0 s                                 餘裕 198.5x（runTimeout = 1200 秒）
```

## `_template_from_other_project/`

另一個專案（MCTS 卡牌）的 cell，留著當風格參考，**跟 Kaggriculture 無關**，
不要貼進 notebook。
