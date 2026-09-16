# Import the game
import cv2
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

#建立環境
# env = gym_super_mario_bros.make('SuperMarioBros-v0',apply_api_compatibility=True, render_mode="rgb_array")
env = gym_super_mario_bros.make('SuperMarioBros-v0',apply_api_compatibility=True)

env = JoypadSpace(env, SIMPLE_MOVEMENT)


# restart or not
done = True
# Loop through each frame in the game
A=-1
k=20


for _ in range(10000):

    # if done | truncated:
    if done: #先讓遊戲結束可以reset環境

        # start the game
        observation = env.reset()
    # do random actions
    # observation, reward, done,truncated, info = env.step(env.action_space.sample())
    
    if A==-1: #有0~6七種按鍵
        #A=env.action_space.sample()
        A=0
    observation, reward, done,_,info = env.step(A) #執行動作
    #螢幕圖片(神經網路看到的),拉竿子,結束(死亡&成功),info(天擇，雜湊函數):fitnessfunction(exposition,score(金幣)

    # show the game on screen RGB=>BGR
    x=observation[:,:,0].copy()
    observation[:,:,0]=observation[:,:,2].copy()
    observation[:,:,2]=x
    
    #tileFormat=observation.reshape([TileH,TileSize,TileW,TileSize,3]).transpose([0,2,1,3,4]).reshape([])
    #downScale=np.mean(tileFormat,(1,2)).reshape([TileH,TileSize,3]).astype(np.unit8)
    #vis=cv2.resize(downScale(100,100))
    cv2.imshow('Game',observation)
    key=cv2.waitKey(30)
    

    if key== 27 or key==113: #esc離開
        print(info['x_pos'])
        break
    elif key==ord('e'):
        A=2
    elif key==ord('w') :
        A=5
    elif key==ord('s'):
        A=0
    elif key==ord('d') :
        A=1
    elif key==ord('a') :
        A=6
    else:
        k-=1
        if k<0:
            k=20
            A=0
    print(info['x_pos']) #距離
# close the game
env.close()
cv2.destroyAllWindows()