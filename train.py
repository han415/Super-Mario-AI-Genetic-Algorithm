import cv2
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np

# 磚塊大小
TileSize = 16
TileH = 15 
TileW = 16

# GA 參數設置
PopulationSize = 100  # 生物圈所能承受的生物個體數 要4的倍數
MutationRate = 0.08  # 基因突變率 10%有點多
alpha = 0.3

dim = TileH * TileW * 3  # observation, 3 => RGB

# 神經節是一矩陣 #7個按鈕(神經元)
fc = np.random.randn(PopulationSize * dim * 7).reshape([PopulationSize, dim, 7])

# 環境
env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

A = -1
k = 20
key = 0

# 初始化一些變量
highest_x_pos = -float('inf')  # 記錄最高的x_pos

for ger in range(5000):
    rec = []  # 記錄每次訓練中的所有x_pos值
    for ai in range(PopulationSize):
        observation = env.reset()
        observation = observation[0]
        done = False
        stm = []  # 短期記憶
        total_x_pos = 0  # 記錄這個回合的總x_pos
        for i in range(5000):  # step number            
            if done:
                break
            # 把observation縮小並轉浮點數
            x = cv2.resize(observation, (TileH, TileW)).astype(float).reshape([-1]) / 255 
            # 輸出動作
            z = np.dot(x, fc[ai])
            y = 1 / (1 + np.exp(-z))
        
            A = np.argmax(y)           
            observation, reward, done, _, info = env.step(A)  # 執行動作
            cv2.imshow('Game_alpha', cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            total_x_pos = info['x_pos']  # 更新x_pos
            stm.append(info['x_pos'])
            # 如果短期記憶中記錄的 x_pos 超過了 100 個
            if len(stm) > 100:
                # 如果最近的 x_pos 與 100 步驟之前的 x_pos 之間沒有增加
                if stm[-1] - stm[-99] <= 0:
                    break  # 提前結束這個個體的模擬
        
        rec.append(info['x_pos'])  # 記錄每次訓練中的x_pos值

    print("最高的x_pos:", highest_x_pos)

    # 排序
    fitness = np.array(rec)

    
    best_genome = None
    best_fitness = -np.inf
    
    fitness = np.array(rec)
    max_fitness = np.max(fitness)
    print('generation ' + str(ger) + ' dist=', str(np.max(fitness)))
    
    if max_fitness > best_fitness:
        best_fitness = max_fitness
        best_genome = fc[np.argmax(fitness)]
        
        # Save the best genome to a file after each generation
        np.save('best_genome_alpha.npy4', best_genome)
        print(f'New best fitness: {best_fitness} - genome saved.')
        
    selection = np.argsort(-fitness)[0:PopulationSize // 2]
    
    fc = fc[selection]
    shuffle = np.random.choice(PopulationSize // 2, PopulationSize // 2, replace=False)
    fc = fc[shuffle]
    
    group = []
    
    flatChromosome = fc.reshape([len(fc), -1]).copy()
    crossOverPoint = np.random.randint(0, len(flatChromosome[0]), PopulationSize)
    chooseAllele = np.random.randint(0, 2, PopulationSize)
    chooseMutation = np.random.uniform(0, 1, PopulationSize // 2 * len(flatChromosome[0])).reshape(np.shape(flatChromosome))
    
    for j in range(PopulationSize//4): #保證每個後代有父母參與
        j=j*2
        offSprint=flatChromosome.copy() #基因攤平
        
        if chooseAllele[j] == 0: #隨機選擇
            offSprint[j, 0:crossOverPoint[j]] = alpha * flatChromosome[j+1, 0:crossOverPoint[j]] + (1 - alpha) * flatChromosome[j, 0:crossOverPoint[j]]
            offSprint[j+1, 0:crossOverPoint[j]] = alpha * flatChromosome[j, 0:crossOverPoint[j]] + (1 - alpha) * flatChromosome[j+1, 0:crossOverPoint[j]]
        else:
            offSprint[j, crossOverPoint[j]:] = alpha * flatChromosome[j+1, crossOverPoint[j]:] + (1 - alpha) * flatChromosome[j, crossOverPoint[j]:]
            offSprint[j+1, crossOverPoint[j]:] = alpha * flatChromosome[j, crossOverPoint[j]:] + (1 - alpha) * flatChromosome[j+1, crossOverPoint[j]:]
    #基因突變
    mutation = offSprint[chooseMutation < MutationRate]
    #高斯分佈
    offSprint[chooseMutation < MutationRate] = np.random.randn(len(mutation))
    #更新fc 合併父母跟孩子
    fc = np.concatenate([flatChromosome, offSprint], 0).reshape([PopulationSize, -1, 7])
    
env.close()
