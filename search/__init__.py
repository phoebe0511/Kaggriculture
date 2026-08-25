"""離線 search —— `docs/CLAUDE.md` 路線圖「網路 + search」那一步。

🩸 **這裡的東西上不了場。** 每個決策要跑幾十次「打到底」的 rollout，而
`actTimeout` 是 1 秒（journal 08-24 §6 實測 8 個真實 episode config 都一樣）。
它的用途是**離線產出比 expert 更好的動作**，再靠 DAgger 蒸餾回網路。

所以 `agents/` 底下任何檔案都不該 import 這個套件。
"""
