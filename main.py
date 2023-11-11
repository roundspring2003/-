# -*- coding: Big5 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functools import partial
from PIL import Image, ImageTk
from tkinter import ttk
import matplotlib.animation as animation
import time


def load_file(path): #讀取資料 
    load=np.loadtxt(path)
    data=load[:,:-1]
    target=load[:,-1]
    target=target.reshape(-1,1)
    return data,target

def car_move(x,y,car_degree,wheel_degree): #車移動公式
    car_radians=math.radians(car_degree)
    wheel_radians=math.radians(wheel_degree)
    
    x=x+math.cos(wheel_radians+car_radians)+math.sin(wheel_radians)*math.sin(car_radians)
    y=y+math.sin(wheel_radians+car_radians)-math.sin(wheel_radians)*math.cos(car_radians)
    car_radians=car_radians-math.asin(math.sin(wheel_radians)/3)
    car_degree=math.degrees(car_radians)
    x=round(x,4)
    y=round(y,4)
    return x,y,car_degree

#定義活化函數sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

#計算loss
def criterion(data,target,ih_weight,ho_weight):
    y1=sigmoid(np.dot(data, ih_weight[1:,:])-ih_weight[0,:])
    y2=sigmoid(np.dot(y1,ho_weight[1:,:])-ho_weight[0,:])
    loss=(target-y2)**2
    loss=loss.sum()/len(target)
    return loss

#倒傳遞演算法
def run_epoch(train_data, target,epoch):
    # 三個權值，隨機
    np.random.seed(0)
    learing_rate=0.1
    input_size=len(train_data[0])
    output_size=len(target[0])
    hidden_size=20
    i_h_weight_key = np.random.uniform(-1, 1, size=(input_size+1,hidden_size))
    h_o_weight_key = np.random.uniform(-1, 1, size=(hidden_size+1,output_size))
    loss=[]
    for i in range(epoch):
        bar['value'] = (i+1)*100/1000
        val.set(f'{int((i+1)*100/1000)}%')
        win.update()
        
        for number in range(len(train_data)):
            #正向
            y = sigmoid(np.dot(train_data[number].reshape(1,input_size), i_h_weight_key[1:,:])-i_h_weight_key[0,:])  # 做內積和扣掉神經閥值            
            o = sigmoid(np.dot(y, h_o_weight_key[1:,:])-h_o_weight_key[0,:])  # 做內積和扣掉神經閥值
            #反向
            delta_output=(target[number]-o)*o*(1-o)
            delta_hidden=y*(1-y)*np.dot(delta_output,h_o_weight_key[1:,:].T)
            
            #更改權重值
            i_h_weight_key[0]=i_h_weight_key[0]-learing_rate*delta_hidden#更改bias
            i_h_weight_key[1:,:]=i_h_weight_key[1:,:]+learing_rate*np.dot(train_data[number].reshape(input_size,1),delta_hidden)

            h_o_weight_key[0]=h_o_weight_key[0]-learing_rate*delta_output#更改bias
            h_o_weight_key[1:,:]=h_o_weight_key[1:,:]+learing_rate*np.dot(y.reshape(hidden_size,1),delta_output)
         
        c_loss=criterion(train_data,target,i_h_weight_key,h_o_weight_key)
        loss.append(c_loss)    
        if((i+1)%10==0):
            print(f"{i+1} times,loss= {c_loss: .4f}")
    
    return i_h_weight_key,h_o_weight_key,loss

#感知器計算距離
def sentive_distance(x,y,car_degree):
    file_name = "軌道座標點.txt"
    position=np.loadtxt(file_name,delimiter=',',skiprows=1)
    position=position[2:]
    radians=np.radians(car_degree)
    l_x=100*math.cos(radians)+x
    l_y=100*math.sin(radians)+y
    start=[x,y]
    end=[l_x,l_y]
    distances=[]
    input_path=LineString([start,end])
    for i in range(len(position)):
        segment=(position[i],position[(i+1)%len(position)])
        segment_path = LineString(segment)
        
        if input_path.intersects(segment_path):
            start=Point(start)
            inter_points = input_path.intersection(segment_path)
            distance=inter_points.distance(start)
            distances.append(distance)
    return round(min(distances),4)

#讀取感知器距離計算方向盤角度
def caulate_wheel(merge):
    y = sigmoid(np.dot(merge.reshape(1,3), i_h_weight_key[1:,:])-i_h_weight_key[0,:])  # 做內積和扣掉神經閥值            
    o = sigmoid(np.dot(y, h_o_weight_key[1:,:])-h_o_weight_key[0,:])  # 做內積和扣掉神經閥值
    wheel_degree=o*(target_max-target_min)+target_min    
    return wheel_degree[0][0]

#訓練
def train(train_data, target,epoch):
    global i_h_weight_key,h_o_weight_key

    but_train['state']='disabled'

    newWindow = tk.Toplevel(win)
    newWindow.title("等待訓練")
    newWindow.geometry("450x450")
    gif = Image.open(r"C:\\Users\\User\Desktop\\giphy.gif")
    canvas = tk.Canvas(newWindow, width=500, height=500)
    canvas.pack()
    frames = []
    for frame in range(0, gif.n_frames):
        gif.seek(frame)
        frames.append(ImageTk.PhotoImage(gif))
    def animate_gif(frame=0):
        canvas.itemconfig(image_item, image=frames[frame])
        newWindow.after(50, animate_gif, (frame+1) % len(frames))

    image_item = canvas.create_image(200, 200, image=frames[0])
    
    animate_gif()
    time.sleep(0.01)
    i_h_weight_key,h_o_weight_key,_=run_epoch(train_data, target,epoch)

    but_start['state']='normal'
    newWindow.destroy()

#更新車子路徑
def draw():
    lb_direct_value.place(x=100,y=550)
    lb_left_value.place(x=380,y=550)
    lb_right_value.place(x=650,y=550)
    x=0
    y=0
    car_degree=90
    circle = plt.Circle((x, y), radius=3, color='g', fill=False)
    position_x=[x]
    position_y=[y]
    wheel_degrees=[]
    merges=[]

    track4D_path = "track4D.txt"
    track6D_path = "track6D.txt"
    while True:
        direct=sentive_distance(x,y,car_degree)
        left=sentive_distance(x,y,car_degree-45)
        right=sentive_distance(x,y,car_degree+45)
        lb_direct_value.config(text=direct)
        lb_left_value.config(text=left)
        lb_right_value.config(text=right)
        
        merge=np.hstack((direct,right,left))
        merges.append(merge)
        merge=(merge-data_min)/(data_max-data_min)
        wheel_degree=caulate_wheel(merge)
        x,y,car_degree=car_move(x,y,car_degree,-wheel_degree)
        wheel_degrees.append(-wheel_degree)
        circle.set_visible(False)
        circle = plt.Circle((x, y), radius=3, color='g', fill=False)
        ax.add_patch(circle)
        ax.plot(position_x, position_y, color='blue')
        canvas1.draw()
        win.update()
        time.sleep(0.25)
        if 40-y<=3:
            break
        position_x.append(x)
        position_y.append(y)

    
    np.savetxt(track4D_path, np.column_stack((merges, wheel_degrees)), fmt="%.7f %.7f %.7f %.7f")
    np.savetxt(track6D_path, np.column_stack((position_x,position_y,merges, wheel_degrees)), fmt="%.7f %.7f %.7f %.7f %.7f %.7f")
    
    #win.update()
    #my_animation = animation.FuncAnimation(fig,fun,interval=500,frames=300,init_func=init)
    #canvas1.draw()

       

#抓取資料
data,target=load_file("train4dAll.txt")
global target_min,target_max
epoch=1000
#資料前處理
data_max=np.max(data,axis=0)
data_min=np.min(data,axis=0)
data=(data-data_min)/(data_max-data_min)
target_max=np.max(target,axis=0)
target_min=np.min(target,axis=0)
target_std=(target-target_min)/(target_max-target_min)


#GUI介面顯示
win=tk.Tk()
win.config(bg="gainsboro")
win.title("Self-car")
win.geometry("900x600")

#地圖製作
fig = plt.figure()
fig.set_size_inches(4.5, 3.5)
position=np.loadtxt("軌道座標點.txt",delimiter=',',skiprows=1)

ax = fig.add_subplot(111, aspect='equal')
canvas1 = FigureCanvasTkAgg(fig, master=win)
canvas1.draw()
canvas1.get_tk_widget().place(x=450,y=10)

polygon1 = Polygon(position[2:], closed=True, fill=True,alpha=0.3) #地圖
ax.add_patch(polygon1)
vertices2 = [[18, 37], [18, 40], [30, 40], [30, 37]] #終點
polygon2 = Polygon(vertices2, closed=True, fill=True,color='red')
ax.add_patch(polygon2)
ax.axhline(y=0,color='green',xmin=3/32, xmax=12/32)#起點         
ax.set_xlim((-10, 32))
ax.set_ylim((-5, 53))

# 設置字體
font_size = 20  # ?置字体大小
font_style = "bold"  # ?置字体?式
custom_font = ("微軟正黑體", font_size, font_style)
value_font = ("微軟正黑體", 15)

lb_direct=tk.Label(text="前方距離：", font=custom_font)
lb_left=tk.Label(text="左邊距離：", font=custom_font)
lb_right=tk.Label(text="右邊距離：", font=custom_font)
lb_direct_value=tk.Label(text="", font=value_font)
lb_left_value=tk.Label(text="", font=value_font)
lb_right_value=tk.Label(text="", font=value_font)
but_train=tk.Button(win,text="開始訓練",command=partial(train, data, target_std, epoch),font=value_font,width=8)
but_start=tk.Button(win,text="開車",command=draw,font=value_font,width=8)
bar = ttk.Progressbar(win, length=200,mode='determinate')
val = tk.StringVar()
val.set('0%')
bar_label = tk.Label(win, textvariable=val,font=value_font)

but_start['state']='disabled'

lb_direct.place(x=100,y=500)
lb_left.place(x=380,y=500)
lb_right.place(x=650,y=500)
but_train.place(x=150,y=200)
bar.place(x=100,y=300)
bar_label.place(x=180,y=325)
but_start.place(x=600,y=400)

win.mainloop()