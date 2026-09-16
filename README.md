# Super Mario AI: Genetic Algorithm

基於遺傳演算法與神經網路的超級瑪利歐自動遊玩代理人
**Automated Super Mario Agent using Genetic Algorithm and Neural Networks**

本專案從零開始 (From Scratch) 建構神經網路與遺傳演算法 (Genetic Algorithm, GA)，讓 AI 在 `gym-super-mario-bros` 環境中透過不斷的世代交替、基因突變與物競天擇，自我學習並找出能讓瑪利歐跑得最遠的最佳操作權重。

---

## Overview

**開發流程：**

`Environment Setup` ➔ `Neural Network Initialization` ➔ `Genetic Algorithm (Selection, Crossover, Mutation)` ➔ `Fitness Evaluation` ➔ `Model Export`

| 開發階段 | 核心技術 | 對應程式碼 |
| :--- | :--- | :--- |
| **1. 演算法訓練** | 結合神經網路前向傳播與 GA 演化迭代邏輯 | [`train.py`](./train.py) |
| **2. 模型展示** | 讀取最佳基因權重，並實際渲染遊戲畫面 | [`play.py`](./play.py) |
| **3. 權重儲存** | 儲存訓練 5000 代後表現最佳的神經網路權重 | [`best_genome.npy`](./best_genome.npy) |

---

## 專題成果

###  [點擊觀看 AI 遊玩實機演示 (YouTube)](https://youtu.be/eV816k5ZiEE)
[![Demo Video](https://img.youtube.com/vi/eV816k5ZiEE/0.jpg)](https://www.youtube.com/watch?v=eV816k5ZiEE)

###  [專題簡報](./Presentation.pdf)

---

## 專題簡介

市面上許多遊戲 AI 專案依賴高度封裝的深度學習框架進行強化學習 (Reinforcement Learning)。然而，為了深入探究演算法底層邏輯，本專案決定不依賴現成框架，直接**手刻單層神經網路與完整的遺傳演算法機制**。

系統以瑪利歐的遊戲畫面作為輸入，透過每一代 100 個 AI 個體的試錯，以「橫向移動距離」為適應度標準進行物競天擇，最終演化出能夠自主通關的最佳權重模型。

---

## 開發目標

* 在 `gym-super-mario-bros` 環境中建構客製化的神經網路控制器
* 完整實作遺傳演算法的五大階段：編碼、適應度、選擇、交配、突變
* 實作防卡死機制 (Short-term Memory)，優化訓練效率
* 比較單層神經網路與多層感知器 (MLP) 在 GA 架構下的收斂表現
* 測試並調校突變率 (Mutation Rate) 與交配權重 (Alpha) 對演化的影響

---

## 系統架構

本專案的演算法架構與資料流傳遞過程如下圖所示：

<pre>
       Game Observation (15x16x3 RGB)
                  │
                  ▼
         Flatten to 1D Array (720)
                  │
                  ▼
   ┌─────────────────────────────┐
   │ Single Layer Neural Network │
   │   Weight Shape: (720, 7)    │ ◀ 基因權重由 GA 演化更新
   └──────────────┬──────────────┘
                  │
                  ▼
         Action Prediction (7)
           (Sigmoid + Argmax)
                  │
                  ▼
       Execute in Super Mario
</pre>

---

## 方法：遺傳演算法實作

### 1. Encode (基因編碼與環境輸入)
* **神經網路架構**：將遊戲畫面縮放至 `15x16` 並提取 RGB 通道，展平為 `720` 維的特徵向量。
* **輸出層**：輸出對應至 7 種基本的瑪利歐動作按鈕，最終選擇激勵值（Sigmoid）最大的動作執行。
* **GA 參數設定**：族群大小 (Population) 設為 100，共訓練 5000 代，突變率設為 0.08。

### 2. Fitness Function (適應度函數)
* **距離指標**：以系統回傳的橫向移動距離 `info['x_pos']` 作為個體的適應度分數，跑得越遠分數越高。
* **Short-term Memory (防卡死機制)**：系統會記錄過去 100 步的 `x_pos`。若 100 步內距離皆無增加，則判定卡死並提前結束該回合，大幅節省無效的訓練時間。

### 3. Selection (物競天擇)
每一代訓練結束後，對 100 個個體的適應度進行排序。保留表現最好的前 50% 菁英個體作為下一代的父母本，淘汰後 50%。

### 4. Crossover (基因交配)
* 隨機生成隨機數決定交配段落，將父母代的基因陣列進行切分與重新組合（單點與多點交配）。
* 引入**混合交配 (Blended Crossover)**，加入 `alpha = 0.3` 的權重，讓子代基因按比例融合父母雙方的特徵，提升收斂穩定性。

### 5. Mutation (基因突變)
* 針對子代基因，以 8% 的機率觸發突變。
* 突變方式採用**高斯分布 (Gaussian Distribution)**，隨機賦予均值為 0、標準差為 1 的新數值更新權重，以維持物種多樣性並跳出局部最佳解。

---

## 開發過程與分析

在專案開發過程中，團隊透過調整多項參數與模型架構來尋求最佳解：

1. **參數調整測試**：
   增加 Population size 與 Generations 確實能讓瑪利歐有更高機率探索出更好的破關路線，但相對所需的矩陣運算時間呈指數級上升，需在效能與成效間取得平衡。
2. **多層神經網路 (MLP) 嘗試**：
   團隊曾嘗試引入隱藏層處理更複雜的畫面特徵，期望優化決策。然而實驗結果顯示，在此單純的 GA 架構下，增加網路深度反而導致權重難以收斂、表現不佳，最終決定回歸輕量且高效的單層架構。
3. **動作權重人為干預**：
   曾嘗試人為介入調整特定動作（如強迫向右跳躍）的初始機率，但發現過度干預反而限縮了 GA 自由探索新路徑的潛力，最終交由演算法透過適應度自行學習。

---

## 未來發展

* 嘗試實作 NEAT (NeuroEvolution of Augmenting Topologies) 演算法，讓 AI 不只演化權重，也自動演化網路結構。
* 將適應度函數 (Fitness Function) 加入破關時間、獲得金幣數與擊敗敵人數量等複合指標。
* 引入多執行緒 (Multiprocessing) 同時運行多個遊戲環境，加速演化過程。

---

## 使用技術

| 類別 | 技術 |
| :--- | :--- |
| Programming | Python |
| Game Environment | `gym-super-mario-bros`, `nes-py` |
| Image Processing | OpenCV (`cv2`) |
| Data Computation | NumPy |

---

## 專案結構

<pre>
Super-Mario-GA-Bot/
│
├── train.py                 # GA 演算法與神經網路訓練主程式
├── play.py                  # 讀取模型權重並展示遊玩畫面
├── best_genome.npy          # 最佳神經網路權重檔
│
├── Presentation.pdf         # 演算法期末實驗報告
└── README.md
</pre>

---

## 個人貢獻

在本次五人團隊專題中，我主要參與以下工作：
* 協助建構單層神經網路的矩陣運算與前向傳播機制。
* 參與設計與優化遺傳演算法流程（交配、突變機率調整）。
* 測試不同 Fitness Function (如引入 Short-term Memory 防卡死機制) 對收斂速度的影響。
* 分析單層網路與多層神經網路 (MLP) 在演化過程中的表現差異。

---

## 專題資訊

**專題名稱：** 演算法期末報告 - Super Mario AI
**專題成員：**
* 余沛達
* 温苡均
* 蘇靖文
* 黃妤涵
* 楊芷璇
