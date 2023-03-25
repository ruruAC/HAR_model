import streamlit as st
import pandas as pd
from scipy import stats
import numpy as np
#import plotly.graph_objs as go
# https://blog.csdn.net/u013719780/article/details/53178363?locationNum=8&fps=1
np.random.seed(813306)
import plotly.graph_objs as go
import tensorflow.keras as keras
import os
from tensorflow.keras.utils import to_categorical
def windows(a, size) :
    len1 = len(a)
    num = len1 // size
    res = []
    for i in range(num) :
        start = i * size
        end = (i + 1) * size
        res.append((start, end))
    return res

def read_data_test(file_path) :
    column_names = ['timestamp', 'x-axis', 'y-axis', 'z-axis']
    # data = pd.read_csv(file_path, header=None, names=column_names)
    data = pd.read_csv(file_path, header=0)
    return data


def segment_signal_test(data, window_size=90) :
    
    segments = np.empty((0, window_size, 3))
    labels = np.empty((0))
    # print len(data['timestamp'])
    count = 0
    for (start, end) in windows(data['timestamp'], window_size) :
        # print count
        count += 1
        x = data["x-axis"][start :end]
        y = data["y-axis"][start :end]
        z = data["z-axis"][start :end]
        if (len(data['timestamp'][start :end]) == window_size) :
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["act"][start :end])[0][0])

    return segments, labels


def load_data_test() :

    data = read_data_test(filepath)
    segments, labels = segment_signal_test(data, window_size=128)

    for i in range(segments.shape[0]) :
        for j in range(segments.shape[1]) :
            for k in range(segments.shape[2]) :
                if type(segments[i][j][k]) == str :
                    segments[i][j][k] = segments[i][j][k].replace(';', '')
    a = segments.astype('float32')
    # b=[0 if i =='Downstairs' else i for i in labels]
    # b=[1 if i =='Jogging' else i for i in b]
    # b=[2 if i =='Sitting' else i for i in b]
    # b=[3 if i =='Standing' else i for i in b]
    # b=[4 if i =='Upstairs' else i for i in b]
    # b=[5 if i =='Walking' else i for i in b]
    b = [0 if i == 'sit' else i for i in labels]
    b = [1 if i == 'stand' else i for i in b]
    b = [2 if i == 'walk' else i for i in b]
    b=[3 if i =='Jogging' else i for i in b]
    b=[4 if i =='Upstairs' else i for i in b]
    b=[5 if i =='Downstairs' else i for i in b]
    a = a[:, :, np.newaxis, :]
    #b = to_categorical(b, 3)
    # x_train,x_val,y_train,y_val = train_test_split(a, b,test_size = 0.3,random_state = 2019)
    return data,a
def segment_signal_y(y_label,y_pre,window_size=90):
    data = read_data_test(filepath)
   
    #labels = np.empty((0))
    labels=[]
    labels2=[]
    # print len(data['timestamp'])
    count = 0
    
    for (start, end) in windows(data['timestamp'], window_size):
        # print count

        if (len(data['timestamp'][start:end]) == window_size):
            for i in range(128):
                labels = np.append(labels,y_pre[count])
                labels2=np.append(labels2,y_label[count])
        count=count+1
    df={"timestamp":data['timestamp'].index,"activate":labels[0:],"act_num":labels2[0:]}
    print(df)
    return labels,df
import time


import cartoon_html
st.set_page_config(
    page_title="HAR Web",
    page_icon="浙江大学.png",
     initial_sidebar_state="expanded", )
cartoon_html.cartoon_html()
if __name__ == '__main__':
    placeholder = st.empty()
    #st.write(time.ctime())
    for i in range(1):
        
        #my_bar = st.progress(0)
        placeholder10=st.empty()
        
        placeholder.caption(time.ctime())

        placeholder10.info('**请首先采集数据** :wave:')
        percent_complete =0
        data,x_test= load_data_test()
        model = keras.models.load_model("model_ql.h5")  # 500epoch
        y_p = model.predict(x_test)
        y_label = np.argmax(y_p, axis=-1)
        print(y_label)
        y_pre=[]
        #my_bar.progress(percent_complete + 20)

        
        for ans in y_label :
            if ans == 0 :
                y_pre.append("Sitting!")
                print("Sitting!")
            elif ans == 1 :
                y_pre.append("Standing!")
                print("Standing!")
            else :
                y_pre.append("Walking!")
                print("Walking!")
        labelsy,df=segment_signal_y(y_label,y_pre,window_size=128)
        #my_bar.progress(percent_complete + 50)
        st.sidebar.subheader('**数据采集** :wave:')
        agree=st.sidebar.button("Begin")
        #print(labelsy)
        #cartoon_html.cartoon_html()
        data,x_test= load_data_test()
        placeholder4=st.sidebar.empty()
        placeholder5=st.sidebar.empty()
        placeholder6=st.sidebar.empty()

        
        st.sidebar.subheader('**姿态识别** :wave:')
        
        placeholder2 = st.sidebar.empty()
        placeholder3=st.sidebar.empty()
        #st.sidebar.write("0:Sitting :bow:  1:Standing :shoe: 2:Walking :feet:  3:Jogging :running:  4:Upstairs :arrow_up: 5:Downstairs  :arrow_down:")
        placeholder3 = st.sidebar.empty()
        #my_bar2 = st.sidebar.progress(0)
        agree2=st.sidebar.button("Stop")
        #my_bar.progress(percent_complete + 70)
        if agree :
            
            for i in range(len(df['act_num'])):
                placeholder.caption(time.ctime())
                time.sleep(0.05)  
                placeholder4.write("x-axis:"+str(data['x-axis'][i]))
                placeholder5.write("y-axis:"+str(data['y-axis'][i]))
                placeholder6.write("z-axis:"+str(data['z-axis'][i]))
                if df['act_num'][i]==0:
                    #str=str+":bow:"
                    
                    placeholder2.write("Sitting :bow:")
                    placeholder3.write(str(i/20)+"s"+"...")
                # my_bar2.progress(int((100/len(df['act_num']))*i))
                elif df['act_num'][i]==1:
                    #str=str+":shoe:"
                    
                    placeholder2.write("Standing :shoe:")
                    placeholder3.write(str(i/20)+"s"+"...")
                # my_bar2.progress(int((100/len(df['act_num']))*i))
                else:
                    #str=str+":feet:"
                
                    placeholder2.write("Walking :feet:")
                    placeholder3.write(str(i/20)+"s"+"...")
                    #my_bar2.progress(int((100/len(df['act_num']))*i))
        # st.sidebar.write(str)
        if agree2:
            placeholder10.empty()
                


            placeholder4.write("x-axis:"+"-")
            placeholder5.write("y-axis:"+"-")
            placeholder6.write("z-axis:"+"-")
            placeholder3.write("Ending!")
            st.subheader('**数据采集** :wave:')
            
            st.line_chart(data['x-axis'])
            st.line_chart(data['y-axis'])
            st.line_chart(data['z-axis'])
            st.subheader('**姿态识别** :wave:')
            st.write("0:Sitting :bow:  1:Standing :shoe: 2:Walking :feet:  3:Jogging :running:  4:Upstairs :arrow_up: 5:Downstairs  :arrow_down:")

            st.line_chart(df['act_num'])
            st.sidebar.line_chart(df['act_num'])
            
            flag=0

            #my_bar.progress(percent_complete + 100)
            st.success('This is a success message!')
            st.balloons()
            placeholder.caption(time.ctime())
        

            

        st.sidebar.caption("Made by Group 4@zju :ribbon:")
        
        