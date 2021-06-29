# -*- coding: utf-8 -*-

## Used Imports
import os
import tarfile
import pickle
import math
import random
import glob
import torch
import torchvision
import subprocess
import cv2
import shutil
import os.path
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.nn.functional as F
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import streamlit as st
import time
import clip

# import urllib, cStringIO    # imatges
from io import BytesIO
from os import path
from matplotlib import pyplot
from PIL import Image
from zipfile import ZipFile 
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,plot_confusion_matrix,precision_score,accuracy_score,roc_auc_score,f1_score,recall_score
from tensorflow.keras import layers,regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Activation, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, VGG19, ResNet50
from urllib.request import urlopen
# from google.colab import drive  # for google colab drive mount
from numpy.random import seed
from scipy.special import softmax

# %matplotlib inline
# used_seed=seed(42)

## NOTES:
## - Delete results_vs_one from functions "Predicciones..." (and program?)

## --------------- FUNCTIONS ---------------
def Predict_1_vs_0(prediccion_probs,results_vs_one,mult_coef):
    current_result=[]
    for i in range(len(prediccion_probs[:,0])):
        if prediccion_probs[i,1]>prediccion_probs[i,0]:
            current_result.append(1)
        else:
            current_result.append(0)

    return np.array(current_result)
    
def Predict_0_vs_1(prediccion_probs,results_vs_one,mult_coef):
    current_result=[]
    for i in range(len(prediccion_probs[:,0])):
        if prediccion_probs[i,0]>prediccion_probs[i,1]:
            current_result.append(1)
        else:
            current_result.append(0)

    return np.array(current_result)
    
def Predict_1_vs_2(prediccion_probs,results_vs_one,mult_coef):
    current_result=[]
    for i in range(len(prediccion_probs[:,0])):
        if prediccion_probs[i,1]>prediccion_probs[i,2]:
            current_result.append(1)
        else:
            current_result.append(0)

    return np.array(current_result)
    
def Predict_bald(prediccion_probs,results_vs_one,mult_coef):
    current_result=[]
    for i in range(len(prediccion_probs[:,0])):
    
        if prediccion_probs[i,1]>prediccion_probs[i,2]:
            if prediccion_probs[i,3]>prediccion_probs[i,0]:
                current_result.append(1)
            else:
                current_result.append(0)
        else:
            if prediccion_probs[i,4]>prediccion_probs[i,0]:
                current_result.append(1)
            else:
                current_result.append(0)    

    return np.array(current_result)

## Test image wiht a clip model
def Token_img(n_images,n_tokens,index_token_change,current_images,clip_text, clip_model, clip_transform, clip_device):
    prediccion_probs=np.zeros((n_images,n_tokens))
    results_vs_one=np.zeros((n_images,n_tokens-1))
    for i in range(n_images):
        prediccion_probs[i,:]=CLIP_get_probs_only(current_images[i], clip_text, clip_model, clip_transform, clip_device)
        results_vs_one[i,:]=Predicciones_CLIP_vs_one(prediccion_probs[i,:],index_token_change)
    
    return prediccion_probs, results_vs_one
    
def CLIP_get_probs_only(img_file, img_txt, img_model, img_transf, img_device):
    img_proeprocessed = img_transf(Image.fromarray(img_file)).unsqueeze(0).to(img_device)
    img_features = img_model.encode_image(img_proeprocessed)
    txt_features = img_model.encode_text(img_txt)
    img_logits, img_logits_txt = img_model(img_proeprocessed, img_txt)
    # image_p=softmax(img_logits.detach().numpy()[0])
    image_p=img_logits.detach().numpy()[0]

    return np.round(image_p,2)

def Predicciones_CLIP_vs_one(prob,index_token_change):
    # Bold 
    Prediction_tokens=[]
    for i in range(1,len(prob)):
        if i<index_token_change:
            Prediction_tokens.append(prob[i]>prob[0])
        else:
            Prediction_tokens.append(prob[i]<prob[0])

    return np.array(Prediction_tokens)

def Get_features(path_info):
    ## Read descriptions
    description_labels = np.array(open(path_info+'list_attr_celeba.txt').readlines()[1:2][0].split())
    
    n_labels = len(description_labels)
    file_descriptions = open(path_info+'list_attr_celeba.txt').readlines()[2:]
    description_data = []
    n_data = len(file_descriptions) 
    for i in range(n_data):
        description_data.append([])
        if (str(i+1).zfill(6)+'.jpg')==file_descriptions[i].split()[0]:
            for j in file_descriptions[i].split()[1:]:
                description_data[i].append(j=='1')
        else:
            print('Error inidice:',i)
            
    return n_labels,  description_labels, n_data, np.array(description_data)

def Show_Images(current_figure,first_image,current_images,current_image_names,index_token_change,prediction_index,prediccion_probs, image_current_predictions,description_data, num_cols,num_rows):
    
    fig = current_figure[0]
    axs = current_figure[1]

    current_img=first_image
    current_index=0
    for i in range(num_rows):
        for j in range(num_cols):
            
            current_text=current_image_names[current_index]
            
            if np.sum(prediccion_probs)!=0:
                current_text+=' - Ref: '+str(np.round(prediccion_probs[current_index,0],2))
                
                if index_token_change<len(prediccion_probs[current_index,:]):
                    current_text+='\nT:'
                    for k in range(1,index_token_change):
                        current_text+=str(np.round(prediccion_probs[current_index,k],2))+' '
                        
                    current_text+='\nF:'                        
                    for k in range(index_token_change,len(prediccion_probs[current_index,:])):
                        current_text+=str(np.round(prediccion_probs[current_index,k],2))+' '
                elif index_token_change==99:
                    current_text+='\nTrue:'                        
                    for k in range(1,len(prediccion_probs[current_index,:])):
                        current_text+=str(np.round(prediccion_probs[current_index,k],2))+' '

                elif index_token_change==100:
                    current_text+='\nFalse:'                        
                    for k in range(1,len(prediccion_probs[current_index,:])):
                        current_text+=str(np.round(prediccion_probs[current_index,k],2))+' '
                        
                else:
                    current_text+='\nQuerys:'                        
                    for k in range(1,len(prediccion_probs[current_index,:])):
                        current_text+=str(np.round(prediccion_probs[current_index,k],2))+' '
                
                # if index_token_change<len(prediccion_probs[current_index,:]):
                    # current_text+='\nT: '+str(np.round(prediccion_probs[current_index,1:index_token_change],2))
                    # current_text+='\nF: '+str(np.round(prediccion_probs[current_index,index_token_change:],2))
                # else:
                    # current_text+='\nT: '+str(np.round(prediccion_probs[current_index,1:]))
                    
                if prediction_index<len(description_data[current_img-1,:]):
                    current_text+='\nCeleba info: '+str(description_data[current_img-1,prediction_index])
                    if image_current_predictions[current_index]==1 and description_data[current_img-1,prediction_index]:
                        current_color='green'
                        axs[i,j].axes.axes.set_xlabel(current_text, fontsize=10, color='black')
                    elif image_current_predictions[current_index]==0 and (not description_data[current_img-1,prediction_index]):
                        current_color='blue'
                        axs[i,j].axes.axes.set_xlabel(current_text, fontsize=10, color='black')
                    elif image_current_predictions[current_index]==1 and (not description_data[current_img-1,prediction_index]):
                        current_color='orange'    
                        axs[i,j].axes.axes.set_xlabel(current_text, fontsize=10, color='red')            
                    elif image_current_predictions[current_index]==0 and description_data[current_img-1,prediction_index]:
                        current_color='purple'
                        axs[i,j].axes.axes.set_xlabel(current_text, fontsize=10, color='red')
                    else:
                        current_color='black'
                        axs[i,j].axes.axes.set_xlabel(current_text, fontsize=10, color='black')
                else:
                    if image_current_predictions[current_index]==1:
                        current_color='green'
                    elif image_current_predictions[current_index]==0:
                        current_color='red'
                    axs[i,j].axes.axes.set_xlabel(current_text, fontsize=10, color='black')

                current_line_width=5
            
            else:
                axs[i,j].axes.axes.set_xlabel(current_text, fontsize=15, color='black')
                current_color='black'
                current_line_width=3
                
            axs[i,j].axes.xaxis.set_ticks([])
            axs[i,j].axes.xaxis.set_ticklabels([])
            axs[i,j].axes.yaxis.set_visible(False)
            
            axs[i,j].spines['bottom'].set_color(current_color)
            axs[i,j].spines['top'].set_color(current_color)
            axs[i,j].spines['left'].set_color(current_color)
            axs[i,j].spines['right'].set_color(current_color)
            
            axs[i,j].spines['bottom'].set_linewidth(current_line_width)
            axs[i,j].spines['top'].set_linewidth(current_line_width)
            axs[i,j].spines['left'].set_linewidth(current_line_width)
            axs[i,j].spines['right'].set_linewidth(current_line_width)
            
            current_img+=1
            current_index+=1
            
    st.write(fig)

def Load_Images(first_image,num_rows,num_cols):
    image_files=[]
    image_names=[]
    for i in range(num_rows*num_cols):
        image_current_path='Celeba/img_celeba/'+str(i+first_image).zfill(6)+'.jpg'
        image_files.append(np.array(Image.open(image_current_path)))
        image_names.append(str(i+first_image).zfill(6))
        
    fig, axs = plt.subplots(num_rows,num_cols,figsize=(3*num_cols,3*num_rows))
    plt.subplots_adjust(top = 1.2, bottom=0.0, hspace=0.25, wspace=0.1)
    current_index=0
    for i in range(num_rows):
        for j in range(num_cols):
            axs[i,j].imshow(image_files[current_index])
            axs[i,j].axes.axes.set_xlabel(image_names[current_index], fontsize=np.int(15))
            axs[i,j].axes.xaxis.set_ticks([])
            axs[i,j].axes.xaxis.set_ticklabels([])
            axs[i,j].axes.yaxis.set_visible(False)
            axs[i,j].spines['bottom'].set_color('black')
            axs[i,j].spines['top'].set_color('black')
            axs[i,j].spines['left'].set_color('black')
            axs[i,j].spines['right'].set_color('black')
            axs[i,j].spines['bottom'].set_linewidth(3)
            axs[i,j].spines['top'].set_linewidth(3)
            axs[i,j].spines['left'].set_linewidth(3)
            axs[i,j].spines['right'].set_linewidth(3)
            current_index+=1        

    return np.array(image_files), np.array(image_names), [fig, axs]
  
## Tokenization process
def Token_process(clip_tokens_file):

    ## OBTENER TOKENS
    clip_tokens = []
    with open('./'+clip_tokens_file) as f:
        clip_tokens = f.read().splitlines()
        
    n_tokens=len(clip_tokens)
    
    ## TOKENIZACION
    def argmax(iterable):
        return max(enumerate(iterable), key=lambda x: x[1])[0]

    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_transform = clip.load("ViT-B/32", device=clip_device, jit=False)
    clip_text = clip.tokenize(clip_tokens).to(clip_device)
    
    return n_tokens,clip_tokens,clip_device,clip_model, clip_transform, clip_text

## Tokenization process
def Token_process_query(clip_tokens):
        
    n_tokens=len(clip_tokens)
    
    ## TOKENIZACION
    def argmax(iterable):
        return max(enumerate(iterable), key=lambda x: x[1])[0]

    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_transform = clip.load("ViT-B/32", device=clip_device, jit=False)
    clip_text = clip.tokenize(clip_tokens).to(clip_device)
    
    return n_tokens,clip_tokens,clip_device,clip_model, clip_transform, clip_text

def search_feature(list_features, current_feature):
    for i in range(0,len(list_features)):
        if list_features[i]==current_feature:
            break
    return i    
    
def Show_Info():
    st.write('Token number:',Data_Init['init_data'][0]['n_tokens'])
    st.write('Tokens querys',Data_Init['init_data'][0]['clip_tokens'])
    st.markdown('#### List of querys')
    st.write(Feature_Options)
    # st.write('Prediction index:',Data_Init['init_data'][0]['prediction_index'])
    # st.write('Index change result:',Data_Init['init_data'][0]['index_token_change'])
    # st.write('Selected feature:',Data_Init['init_data'][0]['selected_feature'])
    # st.write(type(Data_Init['init_data'][0]['clip_tokens']))

def Reload_data():
    path_info='D:/Datasets/Celeba/'
    first_image=1
    num_cols=5
    num_rows=4
    new_query=['A picture of a person','A picture of a man','A picture of a woman']
    n_tokens,clip_tokens,clip_device,clip_model, clip_transform, clip_text = Token_process_query(new_query)
    n_labels,  description_labels, n_data, description_data=Get_features(path_info)
    current_images, current_image_names, current_figure =Load_Images(first_image,num_rows,num_cols)
    
    Saved_data={'path_info':path_info,
        'path_imgs':'D:/Datasets/Celeba/img_celeba/',
        'n_tokens':n_tokens,
        'clip_tokens':clip_tokens,
        'clip_device':clip_device,
        'clip_model':clip_model, 
        'clip_transform':clip_transform, 
        'clip_text':clip_text,
        'n_labels':n_labels,  
        'description_labels':description_labels, 
        'n_data':n_data, 
        'description_data':description_data,
        'prediction_index':20,
        'index_token_change':2,
        'mult_coef':1,
        'first_image':first_image,
        'function_predict':Predict_1_vs_2,
        'num_cols':num_cols,
        'num_rows':num_rows,
        'n_images':num_cols*num_rows,
        'current_images':current_images,
        'current_image_names':current_image_names,
        'current_figure':current_figure,
        'image_current_probs':np.zeros((num_cols*num_rows,n_tokens)),
        'results_vs_one':np.zeros((num_cols*num_rows,n_tokens-1)),
        'image_current_predictions':np.zeros((num_cols*num_rows))+2,
        'images_loaded':True,
        'model_loaded':True,
        'images_checked':False,
        'model_changing':False,
        'selected_feature':'Man / Woman',
        'user_input':'A picture of a man',
        'user_input_querys1':'A picture of a man',
        'user_input_querys2':'A picture of a woman',
        'step':0
    }
    return {"init_data": [Saved_data]}   
  
# ---------------   CACHE   ---------------

@st.cache(allow_output_mutation=True) 
def load_data():
    path_info='D:/Datasets/Celeba/'
    first_image=1
    num_cols=5
    num_rows=4
    new_query=['A picture of a person','A picture of a man','A picture of a woman']
    n_tokens,clip_tokens,clip_device,clip_model, clip_transform, clip_text = Token_process_query(new_query)
    n_labels,  description_labels, n_data, description_data=Get_features(path_info)
    current_images, current_image_names, current_figure =Load_Images(first_image,num_rows,num_cols)
    
    Saved_data={'path_info':path_info,
        'path_imgs':'D:/Datasets/Celeba/img_celeba/',
        'n_tokens':n_tokens,
        'clip_tokens':clip_tokens,
        'clip_device':clip_device,
        'clip_model':clip_model, 
        'clip_transform':clip_transform, 
        'clip_text':clip_text,
        'n_labels':n_labels,  
        'description_labels':description_labels, 
        'n_data':n_data, 
        'description_data':description_data,
        'prediction_index':20,
        'index_token_change':2,
        'mult_coef':1,
        'first_image':first_image,
        'function_predict':Predict_1_vs_2,
        'num_cols':num_cols,
        'num_rows':num_rows,
        'n_images':num_cols*num_rows,
        'current_images':current_images,
        'current_image_names':current_image_names,
        'current_figure':current_figure,
        'image_current_probs':np.zeros((num_cols*num_rows,n_tokens)),
        'results_vs_one':np.zeros((num_cols*num_rows,n_tokens-1)),
        'image_current_predictions':np.zeros((num_cols*num_rows))+2,
        'images_loaded':True,
        'model_loaded':True,
        'images_checked':False,
        'model_changing':False,
        'selected_feature':'Man / Woman',
        'user_input':'A picture of a man',
        'user_input_querys1':'A picture of a man',
        'user_input_querys2':'A picture of a woman',
        'step':0
    }
    return {"init_data": [Saved_data]}    

st.set_page_config(
    layout="wide",
    page_title='QuienEsQuien'
    # page_icon='gz_icon.jpeg'
)


## --------------- PROGRAMA ---------------
Data_Init=load_data()
Querys_Prediction_Index=[0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
Querys_List=["A picture of a person with five o'clock shadow", 'A picture of a person with arched eyebrows',
            'A picture of an attractive person', 'A picture of a person with bags under the eyes', 
            'A picture of a person who has bangs', 'A picture of a person with big lips', 
            'A picture of a person with big nose', 'A picture of a person with black hair', 'A picture of a person with blond hair', 
            'A blurry picture of a person', 'A picture of a person with brown hair', 'A picture of a person with bushy eyebrows', 
            'A picture of a chubby person ', 'A picture of a person with a double chin', 'A picture of a person wearing eyeglasses ', 
            'A picture of a person with goatee', 'A picture of a person with gray hair', 'A picture of a person wearing heavy makeup', 
            'A picture of a person with high cheekbones', 'A picture of a person a slightly open mouth', 
            'A picture of a person with mustache', 'A picture of a person with narrow eyes', 
            'A picture of a person who does not wear a beard', 'A picture of a person with an oval face', 
            'A picture of a person wiht pale skin', 'A picture of a person with pointy nose', 
            'A picture of a person who is receding hairline', 'A picture of a person with rosy cheeks', 
            'A picture of a person with sideburns', 'A picture of a person who is Smiling', 'A picture of a person with straight hair', 
            'A picture of a person with wavy hair', 'A picture of a person wearing earrings', 'A picture of a person wearing hat', 
            'A picture of a person wearing lipstick', 'A picture of a person with wearing necklace', 
            'A picture of a person with Wearing necktie', 'A picture of a person who is young'
            ]

## TITLE
if Data_Init['init_data'][0]['step']==0:
    st.title('Guess Who?')
    Data_Init['init_data'][0]["step"]=1
elif Data_Init['init_data'][0]["step"]==1:
    st.title('Guess Who??')
    Data_Init['init_data'][0]["step"]=2
elif Data_Init['init_data'][0]["step"]==2:
    st.title('Guess Who???')
    Data_Init['init_data'][0]["step"]=3
else:
    st.title('Guess Who????')
    Data_Init['init_data'][0]["step"]=0
# st.subheader('Averigua de quién se trata lo antes posible!!!')

## SIDEBAR
st.sidebar.markdown('# OPTIONS')

## Reset App
st.sidebar.markdown('## ')
Reset_App = st.sidebar.button('RESET', key='Reset_App')

## Select imagenes
st.sidebar.markdown('## Images selection (choose the number of the first image)')
Selected_Image=st.sidebar.number_input('A number between 1 and '+str(202999-Data_Init['init_data'][0]["n_images"]), min_value=1, max_value=202999-Data_Init['init_data'][0]["n_images"], value=1, step=Data_Init['init_data'][0]['n_images'], format='%i', key='Selected_Image', help=None)
  
if Selected_Image!=Data_Init['init_data'][0]["first_image"]:
    Data_Init['init_data'][0]["images_loaded"]=False

## Select tokens
st.sidebar.markdown('## Select a new Query')
Feature_Options=['Your own query', 'Your own 2 querys','Man / Woman','Bald / Haired',
            "A picture of a person with five o'clock shadow", 'A picture of a person with arched eyebrows',
            'A picture of an attractive person', 'A picture of a person with bags under the eyes', 
            'A picture of a person who has bangs', 'A picture of a person with big lips', 
            'A picture of a person with big nose', 'A picture of a person with black hair', 'A picture of a person with blond hair', 
            'A blurry picture of a person', 'A picture of a person with brown hair', 'A picture of a person with bushy eyebrows', 
            'A picture of a chubby person ', 'A picture of a person with a double chin', 'A picture of a person wearing eyeglasses ', 
            'A picture of a person with goatee', 'A picture of a person with gray hair', 'A picture of a person wearing heavy makeup', 
            'A picture of a person with high cheekbones', 'A picture of a person a slightly open mouth', 
            'A picture of a person with mustache', 'A picture of a person with narrow eyes', 
            'A picture of a person who does not wear a beard', 'A picture of a person with an oval face', 
            'A picture of a person wiht pale skin', 'A picture of a person with pointy nose', 
            'A picture of a person who is receding hairline', 'A picture of a person with rosy cheeks', 
            'A picture of a person with sideburns', 'A picture of a person who is Smiling', 'A picture of a person with straight hair', 
            'A picture of a person with wavy hair', 'A picture of a person wearing earrings', 'A picture of a person wearing hat', 
            'A picture of a person wearing lipstick', 'A picture of a person with wearing necklace', 
            'A picture of a person with Wearing necktie', 'A picture of a person who is young'
            ]
Selected_Feature=st.sidebar.selectbox('Suggested querys', Feature_Options, index=2, key='selected_feature', help=None)

## New token
if Selected_Feature=='Your own query':
    st.sidebar.markdown('## Test your own query versus "Picture of a person":')
    User_Input = st.sidebar.text_input('Write your own query', Data_Init['init_data'][0]['user_input'], key='User_Input', help=None)
    Check_Query = st.sidebar.button('Test your query', key='Check_Query')

## New token
if Selected_Feature=='Your own 2 querys':
    st.sidebar.markdown('## Test your own querys by introducing 2 descriptioons:')
    User_Input_Querys1 = st.sidebar.text_input('Write your "True" query', Data_Init['init_data'][0]['user_input_querys1'],key='User_Input_Querys1', help=None)
    User_Input_Querys2 = st.sidebar.text_input('Write your "False" query', Data_Init['init_data'][0]['user_input_querys2'],key='User_Input_Querys2', help=None)
    Check_Querys = st.sidebar.button('Test your own querys', key='Check_Querys')

## ACCIONES
if not Data_Init['init_data'][0]['model_loaded']:
    if Selected_Feature=='Bald / Haired':
        New_Query=['A picture of a person','A picture of a man','A picture of a woman','A picture of a yes bald man','A picture of a bald person']
        Data_Init['init_data'][0]['prediction_index']=4
        Data_Init['init_data'][0]['index_token_change']=999
        Data_Init['init_data'][0]['function_predict']=Predict_bald
        Data_Init['init_data'][0]['mult_coef']=1 
        Data_Init['init_data'][0]['n_tokens'],Data_Init['init_data'][0]['clip_tokens'],Data_Init['init_data'][0]['clip_device'],Data_Init['init_data'][0]["clip_model"],Data_Init['init_data'][0]['clip_transform'],Data_Init['init_data'][0]['clip_text']=Token_process_query(New_Query)
        Data_Init['init_data'][0]['model_loaded']=True

    elif Selected_Feature=='Man / Woman':
        New_Query=['A picture of a person','A picture of a man','A picture of a woman']
        Data_Init['init_data'][0]['prediction_index']=20
        Data_Init['init_data'][0]['index_token_change']=2
        Data_Init['init_data'][0]['function_predict']=Predict_1_vs_2
        Data_Init['init_data'][0]['mult_coef']=1
        Data_Init['init_data'][0]['n_tokens'],Data_Init['init_data'][0]['clip_tokens'],Data_Init['init_data'][0]['clip_device'],Data_Init['init_data'][0]["clip_model"],Data_Init['init_data'][0]['clip_transform'],Data_Init['init_data'][0]['clip_text']=Token_process_query(New_Query)
        Data_Init['init_data'][0]['model_loaded']=True

    elif (not Selected_Feature=='Your own query') and (not Selected_Feature=='Your own 2 querys'):
        New_Query=['A picture of a person',Selected_Feature]
        Current_Index = Querys_List.index(Selected_Feature)
        Data_Init['init_data'][0]['prediction_index']=Querys_Prediction_Index[Current_Index]
        Data_Init['init_data'][0]['index_token_change']=99
        Data_Init['init_data'][0]['function_predict']=Predict_1_vs_0
        Data_Init['init_data'][0]['mult_coef']=1
        Data_Init['init_data'][0]['n_tokens'],Data_Init['init_data'][0]['clip_tokens'],Data_Init['init_data'][0]['clip_device'],Data_Init['init_data'][0]["clip_model"],Data_Init['init_data'][0]['clip_transform'],Data_Init['init_data'][0]['clip_text']=Token_process_query(New_Query)
        Data_Init['init_data'][0]['model_loaded']=True

    st.sidebar.markdown('(new Query: '+Selected_Feature+')')

## Selected option changed
if Selected_Feature!=Data_Init['init_data'][0]['selected_feature']:
    Data_Init['init_data'][0]['model_loaded']=False
    Data_Init['init_data'][0]['model_changing']=True
    Data_Init['init_data'][0]['selected_feature']=Selected_Feature

# Option changing
if Selected_Feature=='Your own query':
    if Data_Init['init_data'][0]['user_input']!=User_Input:
        Data_Init['init_data'][0]['user_input']=User_Input
        Data_Init['init_data'][0]["model_changing"]=True

if Selected_Feature=='Your own 2 querys': 
    if Data_Init['init_data'][0]['user_input_querys1']!=User_Input_Querys1:
        Data_Init['init_data'][0]['user_input_querys1']=User_Input_Querys1
        Data_Init['init_data'][0]["model_changing"]=True
        
    if Data_Init['init_data'][0]['user_input_querys2']!=User_Input_Querys2:
        Data_Init['init_data'][0]['user_input_querys2']=User_Input_Querys2
        Data_Init['init_data'][0]["model_changing"]=True
        
## Check images / Load images (default querys)
if Data_Init['init_data'][0]['model_loaded'] and (not Selected_Feature=='Your own query') and (not Selected_Feature=='Your own 2 querys'):
    st.sidebar.markdown('## Current query')
    st.sidebar.markdown('#### '+Data_Init['init_data'][0]['selected_feature'])
    Check_Img = st.sidebar.button('Check', key='Check_Img')
    if Check_Img:
        Data_Init['init_data'][0]['image_current_probs'], Data_Init['init_data'][0]['results_vs_one'] = Token_img(Data_Init['init_data'][0]["n_images"],Data_Init['init_data'][0]['n_tokens'],Data_Init['init_data'][0]['index_token_change'],Data_Init['init_data'][0]['current_images'],Data_Init['init_data'][0]['clip_text'], Data_Init['init_data'][0]["clip_model"], Data_Init['init_data'][0]['clip_transform'], Data_Init['init_data'][0]['clip_device'])
        Data_Init['init_data'][0]["image_current_predictions"]=Data_Init['init_data'][0]['function_predict'](Data_Init['init_data'][0]['image_current_probs'],Data_Init['init_data'][0]['results_vs_one'],1)
        Data_Init['init_data'][0]['images_checked']=True

if (not Data_Init['init_data'][0]['model_loaded']) and (not Selected_Feature=='Your own query') and (not Selected_Feature=='Your own 2 querys'):
    st.sidebar.markdown('## Current model to load:')
    st.sidebar.markdown('#### '+Data_Init['init_data'][0]['selected_feature'])
    Load_Model = st.sidebar.button('Load Model', key='Load_Model')

# Extra inputs (user querys)
if  Data_Init['init_data'][0]["images_loaded"]:
    if Selected_Feature=='Your own query':    
        if Check_Query:
            st.sidebar.markdown('(new query introduced)')
            New_Query=['A Picture of a person',User_Input]
            Data_Init['init_data'][0]['selected_feature']='Your own query'
            Data_Init['init_data'][0]['prediction_index']=99
            Data_Init['init_data'][0]['index_token_change']=99
            Data_Init['init_data'][0]['function_predict']=Predict_1_vs_0
            Data_Init['init_data'][0]['mult_coef']=1
            Data_Init['init_data'][0]['n_tokens'],Data_Init['init_data'][0]['clip_tokens'],Data_Init['init_data'][0]['clip_device'],Data_Init['init_data'][0]["clip_model"],Data_Init['init_data'][0]['clip_transform'],Data_Init['init_data'][0]['clip_text']=Token_process_query(New_Query)
            Data_Init['init_data'][0]['image_current_probs'], Data_Init['init_data'][0]['results_vs_one'] = Token_img(Data_Init['init_data'][0]["n_images"],Data_Init['init_data'][0]['n_tokens'],Data_Init['init_data'][0]['index_token_change'],Data_Init['init_data'][0]['current_images'],Data_Init['init_data'][0]['clip_text'], Data_Init['init_data'][0]["clip_model"], Data_Init['init_data'][0]['clip_transform'], Data_Init['init_data'][0]['clip_device'])
            Data_Init['init_data'][0]["image_current_predictions"]=Data_Init['init_data'][0]['function_predict'](Data_Init['init_data'][0]['image_current_probs'],Data_Init['init_data'][0]['results_vs_one'],1)
            Data_Init['init_data'][0]['images_checked']=True
            Data_Init['init_data'][0]['model_loaded']=True
            Data_Init['init_data'][0]['model_changing']=False

    if Selected_Feature=='Your own 2 querys':
        if Check_Querys:
            st.sidebar.markdown('(new querys introduced)')
            New_Query=[User_Input_Querys1,User_Input_Querys2]        
            Data_Init['init_data'][0]['selected_feature']='Your own 2 querys'
            Data_Init['init_data'][0]['prediction_index']=99
            Data_Init['init_data'][0]['index_token_change']=100
            Data_Init['init_data'][0]['function_predict']=Predict_0_vs_1
            Data_Init['init_data'][0]['mult_coef']=1
            Data_Init['init_data'][0]['n_tokens'],Data_Init['init_data'][0]['clip_tokens'],Data_Init['init_data'][0]['clip_device'],Data_Init['init_data'][0]["clip_model"],Data_Init['init_data'][0]['clip_transform'],Data_Init['init_data'][0]['clip_text']=Token_process_query(New_Query)
            Data_Init['init_data'][0]['image_current_probs'], Data_Init['init_data'][0]['results_vs_one'] = Token_img(Data_Init['init_data'][0]["n_images"],Data_Init['init_data'][0]['n_tokens'],Data_Init['init_data'][0]['index_token_change'],Data_Init['init_data'][0]['current_images'],Data_Init['init_data'][0]['clip_text'], Data_Init['init_data'][0]["clip_model"], Data_Init['init_data'][0]['clip_transform'], Data_Init['init_data'][0]['clip_device'])
            Data_Init['init_data'][0]["image_current_predictions"]=Data_Init['init_data'][0]['function_predict'](Data_Init['init_data'][0]['image_current_probs'],Data_Init['init_data'][0]['results_vs_one'],1)
            Data_Init['init_data'][0]['images_checked']=True
            Data_Init['init_data'][0]['model_loaded']=True
            Data_Init['init_data'][0]['model_changing']=False


## Reload images
if not Data_Init['init_data'][0]["images_loaded"]:
    Data_Init['init_data'][0]["first_image"]=Selected_Image
    Data_Init['init_data'][0]['current_images'], Data_Init['init_data'][0]['current_image_names'], Data_Init['init_data'][0]['current_figure'] = Load_Images(Selected_Image, Data_Init['init_data'][0]["num_rows"], Data_Init['init_data'][0]["num_cols"])
    # Data_Init['init_data'][0]['image_current_probs']=np.zeros((Data_Init['init_data'][0]["n_images"],Data_Init['init_data'][0]['n_tokens']))
    # Data_Init['init_data'][0]['results_vs_one']=np.zeros((Data_Init['init_data'][0]["n_images"],Data_Init['init_data'][0]['n_tokens']-1))
    # Data_Init['init_data'][0]["image_current_predictions"]=np.zeros((Data_Init['init_data'][0]["n_images"]))+2
    # Data_Init['init_data'][0]['images_checked']=False
    Data_Init['init_data'][0]["images_loaded"]=True
    Data_Init['init_data'][0]["model_changing"]=True
    
## Model changing  
if Data_Init['init_data'][0]["model_changing"] or (not Data_Init['init_data'][0]['model_loaded']):
    Data_Init['init_data'][0]['image_current_probs']=np.zeros((Data_Init['init_data'][0]["n_images"],Data_Init['init_data'][0]['n_tokens']))
    Data_Init['init_data'][0]['results_vs_one']=np.zeros((Data_Init['init_data'][0]["n_images"],Data_Init['init_data'][0]['n_tokens']-1))
    Data_Init['init_data'][0]["image_current_predictions"]=np.zeros((Data_Init['init_data'][0]["n_images"]))+2
    Data_Init['init_data'][0]['images_checked']=False
    Data_Init['init_data'][0]['model_changing']=False

## Reset App
if Reset_App:
    Reload_data()

## Show images and info
if  Data_Init['init_data'][0]['images_checked'] and (not Selected_Feature=='Your own query') and (not Selected_Feature=='Your own 2 querys'):
    st.markdown('### According to celeba dataset info:')
    st.markdown('#### True (correct->green , wrong->orange) / False (correct->blue , wrong->purple)')
    st.markdown('#### ')
elif Data_Init['init_data'][0]['images_checked'] and (Selected_Feature=='Your own query' or Selected_Feature=='Your own 2 querys'):
    st.markdown('#### True -> green / False -> red')
    st.markdown('#### ')

Show_Images(Data_Init['init_data'][0]['current_figure'],Selected_Image,Data_Init['init_data'][0]['current_images'],Data_Init['init_data'][0]['current_image_names'],Data_Init['init_data'][0]['index_token_change'],Data_Init['init_data'][0]['prediction_index'],Data_Init['init_data'][0]['image_current_probs'], Data_Init['init_data'][0]["image_current_predictions"], Data_Init['init_data'][0]['description_data'], Data_Init['init_data'][0]["num_cols"],Data_Init['init_data'][0]["num_rows"])

Show_Info()

# used_widget_key = st.get_last_used_widget_key()

# lower_threshold = st.sidebar.text_input(label="Lower Threshold", value="0", key="na_lower")
# upper_threshold = st.sidebar.text_input(label="Upper Threshold", value="100", key="na_upper")
# st.sidebar.button(label="Submit", key="ta_submit")

# if used_widget_key == "ta_submit":
    # do_something(lower_threshold, upper_threshold)