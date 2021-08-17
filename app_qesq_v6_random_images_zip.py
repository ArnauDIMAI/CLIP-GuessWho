# -*- coding: utf-8 -*-

## Used Imports
import os
import io
import zipfile
# import pygame
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
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

# %matplotlib inline
# used_seed=seed(42)

## NOTES:

## --------------- FUNCTIONS ---------------
def Predict_1_vs_0(prediccion_probs):
    current_result=[]
    for i in range(len(prediccion_probs[:,0])):
        if prediccion_probs[i,1]>prediccion_probs[i,0]:
            current_result.append(1)
        else:
            current_result.append(0)

    return np.array(current_result)
    
def Predict_0_vs_1(prediccion_probs):
    current_result=[]
    for i in range(len(prediccion_probs[:,0])):
        if prediccion_probs[i,0]>prediccion_probs[i,1]:
            current_result.append(1)
        else:
            current_result.append(0)

    return np.array(current_result)
    
def Predict_1_vs_2(prediccion_probs):
    current_result=[]
    for i in range(len(prediccion_probs[:,0])):
        if prediccion_probs[i,1]>prediccion_probs[i,2]:
            current_result.append(1)
        else:
            current_result.append(0)

    return np.array(current_result)
    
    ['A picture of a person','A picture of a man','A picture of a woman',
    'A picture of a yes bald man','A picture of a bald person']
    
def Predict_bald(prediccion_probs):
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
    
def Predict_hair_color(prediccion_probs):
    current_result=[]
    for i in range(len(prediccion_probs[:,0])):
        if np.argmax(prediccion_probs[i,:])==0:
            current_result.append(1)
        else:
            current_result.append(0)

    return np.array(current_result)

## Test image wiht a clip model
def Token_img(n_images,n_tokens,current_images,current_images_discarted,clip_text, clip_model, clip_transform, clip_device):
    prediccion_probs=np.zeros((n_images,n_tokens))
    for i in range(n_images):
        prediccion_probs[i,:]=CLIP_get_probs_only(current_images[i], clip_text, clip_model, clip_transform, clip_device)

    return prediccion_probs
    
def CLIP_get_probs_only(img_file, img_txt, img_model, img_transf, img_device):
    img_proeprocessed = img_transf(Image.fromarray(img_file)).unsqueeze(0).to(img_device)
    img_features = img_model.encode_image(img_proeprocessed)
    txt_features = img_model.encode_text(img_txt)
    img_logits, img_logits_txt = img_model(img_proeprocessed, img_txt)
    # image_p=softmax(img_logits.detach().numpy()[0])
    image_p=img_logits.detach().numpy()[0]

    return np.round(image_p,2)

def Image_discarding(image_current_predictions,current_winner_index,current_images_discarted, n_images, num_cols, image_files,image_names):
    for i in range(len(current_images_discarted)):
        if current_images_discarted[i]==0 and image_current_predictions[i]!=image_current_predictions[current_winner_index]:
            current_images_discarted[i]=1

    n_images2=np.sum(current_images_discarted==0)            
    num_rows= int(round(n_images2/num_cols,0))
    if num_rows<n_images2/num_cols:
        num_rows+=1

    image_files2=[]
    image_names2=[]
    image_current_predictions2=[]
    current_index=0
    new_winner_index=0
    new_index=0
    for i in range(n_images):
        if current_images_discarted[current_index]==0:
            image_files2.append(image_files[current_index])
            image_names2.append(image_names[current_index])
            image_current_predictions2.append(image_current_predictions[current_index])
            if current_index==current_winner_index:
                new_winner_index=new_index
                
            new_index+=1
            
        current_index+=1
        
    fig, axs = plt.subplots(num_rows,num_cols,figsize=(1*num_cols,1*num_rows))
    plt.subplots_adjust(top = 0.9, bottom=0.0, hspace=0.2, wspace=0.05)

    current_index=0
    if num_rows>1:
        for i in range(num_rows):
            for j in range(num_cols):
                if current_index<n_images2:
                    axs[i,j].imshow(image_files2[current_index])
                    axs[i,j].axes.axes.set_xlabel(image_names2[current_index], fontsize=np.int(5))
                    axs[i,j].axes.xaxis.set_ticks([])
                    axs[i,j].axes.xaxis.set_ticklabels([])
                    axs[i,j].axes.yaxis.set_visible(False)
                    axs[i,j].spines['bottom'].set_color('black')
                    axs[i,j].spines['top'].set_color('black')
                    axs[i,j].spines['left'].set_color('black')
                    axs[i,j].spines['right'].set_color('black')
                    axs[i,j].spines['bottom'].set_linewidth(1)
                    axs[i,j].spines['top'].set_linewidth(1)
                    axs[i,j].spines['left'].set_linewidth(1)
                    axs[i,j].spines['right'].set_linewidth(1)
                else:
                    axs[i,j].imshow(np.ones((224,224,3)))
                    axs[i,j].axes.xaxis.set_ticks([])
                    axs[i,j].axes.xaxis.set_ticklabels([])
                    axs[i,j].axes.yaxis.set_visible(False)
                    axs[i,j].spines['bottom'].set_color('black')
                    axs[i,j].spines['top'].set_color('black')
                    axs[i,j].spines['left'].set_color('black')
                    axs[i,j].spines['right'].set_color('black')
                    axs[i,j].spines['bottom'].set_linewidth(1)
                    axs[i,j].spines['top'].set_linewidth(1)
                    axs[i,j].spines['left'].set_linewidth(1)
                    axs[i,j].spines['right'].set_linewidth(1)
                
                current_index+=1      
                
    else:
        for j in range(num_cols):
            if current_index<n_images2:
                axs[j].imshow(image_files2[current_index])
                axs[j].axes.axes.set_xlabel(image_names2[current_index], fontsize=np.int(5))
                axs[j].axes.xaxis.set_ticks([])
                axs[j].axes.xaxis.set_ticklabels([])
                axs[j].axes.yaxis.set_visible(False)
                axs[j].spines['bottom'].set_color('black')
                axs[j].spines['top'].set_color('black')
                axs[j].spines['left'].set_color('black')
                axs[j].spines['right'].set_color('black')
                axs[j].spines['bottom'].set_linewidth(1)
                axs[j].spines['top'].set_linewidth(1)
                axs[j].spines['left'].set_linewidth(1)
                axs[j].spines['right'].set_linewidth(1)
            else:
                axs[j].imshow(np.ones((224,224,3)))
                axs[j].axes.xaxis.set_ticks([])
                axs[j].axes.xaxis.set_ticklabels([])
                axs[j].axes.yaxis.set_visible(False)
                axs[j].spines['bottom'].set_color('black')
                axs[j].spines['top'].set_color('black')
                axs[j].spines['left'].set_color('black')
                axs[j].spines['right'].set_color('black')
                axs[j].spines['bottom'].set_linewidth(1)
                axs[j].spines['top'].set_linewidth(1)
                axs[j].spines['left'].set_linewidth(1)
                axs[j].spines['right'].set_linewidth(1)
            
            current_index+=1                           

    return image_current_predictions2, np.zeros(n_images2), image_files2, np.array(image_names2), [fig, axs], n_images2, new_winner_index, num_rows
    
def Show_images(show_results,current_figure,current_images_discarted, image_current_predictions,
                current_winner_index, num_cols, num_rows, n_images):
    fig = current_figure[0]
    axs = current_figure[1]
    current_index=0
    if num_rows>1:  
        for i in range(num_rows):
            for j in range(num_cols):
                if current_index>=n_images:
                    current_line_width=1
                    current_color='white'   
                elif show_results:
                    current_line_width=2
                    if image_current_predictions[current_index]==image_current_predictions[current_winner_index]:
                        current_color='green'
                    else:
                        current_color='red' 
                else:
                    current_line_width=1
                    current_color='black'  
                 
                axs[i,j].spines['bottom'].set_color(current_color)
                axs[i,j].spines['top'].set_color(current_color)
                axs[i,j].spines['left'].set_color(current_color)
                axs[i,j].spines['right'].set_color(current_color)
                
                axs[i,j].spines['bottom'].set_linewidth(current_line_width)
                axs[i,j].spines['top'].set_linewidth(current_line_width)
                axs[i,j].spines['left'].set_linewidth(current_line_width)
                axs[i,j].spines['right'].set_linewidth(current_line_width)       
                current_index+=1
    else:  
        for j in range(num_cols):
            if current_index>=n_images:
                current_line_width=1
                current_color='white'   
            elif show_results:
                current_line_width=2
                if image_current_predictions[current_index]==image_current_predictions[current_winner_index]:
                    current_color='green'
                else:
                    current_color='red' 
            else:
                current_line_width=1
                current_color='black'  
             
            axs[j].spines['bottom'].set_color(current_color)
            axs[j].spines['top'].set_color(current_color)
            axs[j].spines['left'].set_color(current_color)
            axs[j].spines['right'].set_color(current_color)
            
            axs[j].spines['bottom'].set_linewidth(current_line_width)
            axs[j].spines['top'].set_linewidth(current_line_width)
            axs[j].spines['left'].set_linewidth(current_line_width)
            axs[j].spines['right'].set_linewidth(current_line_width)       
            current_index+=1                
            
    return fig,axs

def Load_Images_randomly(num_rows,num_cols):
    n_images=num_rows*num_cols
    image_files=[]
    image_names=[]     
        
    archive = zipfile.ZipFile('guess_who_images.zip', 'r')
    listOfFileNames = archive.namelist()        
    image_index_all=list(range(len(listOfFileNames)))
    image_index=[]
    image_index.append(random.choice(image_index_all))
    image_index_all.remove(image_index[0])
    current_index=1
    while len(image_index)<n_images:
        image_index.append(random.choice(image_index_all))
        image_index_all.remove(image_index[current_index])
        current_index+=1
        
   # Iterate over the file names
    for current_index in image_index:
        image_current_path=listOfFileNames[current_index]
        image_files.append(np.array(Image.open(io.BytesIO(archive.read(image_current_path)))))
        image_names.append(image_current_path[-10:-4])
            
    fig, axs = plt.subplots(num_rows,num_cols,figsize=(1*num_cols,1*num_rows))
    plt.subplots_adjust(top = 0.9, bottom=0.0, hspace=0.2, wspace=0.05)
    current_index=0
    current_index=0
    for i in range(num_rows):
        for j in range(num_cols):
            axs[i,j].imshow(image_files[current_index])
            axs[i,j].axes.axes.set_xlabel(image_names[current_index], fontsize=np.int(5))
            axs[i,j].axes.xaxis.set_ticks([])
            axs[i,j].axes.xaxis.set_ticklabels([])
            axs[i,j].axes.yaxis.set_visible(False)
            axs[i,j].spines['bottom'].set_color('black')
            axs[i,j].spines['top'].set_color('black')
            axs[i,j].spines['left'].set_color('black')
            axs[i,j].spines['right'].set_color('black')
            axs[i,j].spines['bottom'].set_linewidth(1)
            axs[i,j].spines['top'].set_linewidth(1)
            axs[i,j].spines['left'].set_linewidth(1)
            axs[i,j].spines['right'].set_linewidth(1)
            current_index+=1        

    return image_files, np.array(image_names), [fig, axs]
  
# def Load_Images_randomly(num_rows,num_cols):
    # n_images=num_rows*num_cols
    # image_files=[]
    # image_names=[]
    
    # with ZipFile('.\guess_who_images.zip', 'r') as zipObj:
       ### Get a list of all archived file names from the zip
        # listOfFileNames = zipObj.namelist()
        
    # archive = zipfile.ZipFile('.\img_celeba.zip', 'r')
            
    # image_index_all=list(range(len(listOfFileNames)))
    # image_index=[]
    # image_index.append(random.choice(image_index_all))
    # image_index_all.remove(image_index[0])
    # current_index=1
    # while len(image_index)<n_images:
        # image_index.append(random.choice(image_index_all))
        # image_index_all.remove(image_index[current_index])
        # current_index+=1
        
   ### Iterate over the file names
    # for current_index in image_index:
        # image_current_path='img_celeba/'+str(current_index).zfill(6)+'.jpg'
        # image_files.append(np.array(Image.open(io.BytesIO(archive.read(image_current_path)))))
        # image_names.append(str(current_index).zfill(6))
            
    # fig, axs = plt.subplots(num_rows,num_cols,figsize=(1.5*num_cols,1.5*num_rows))
    # plt.subplots_adjust(top = 1.2, bottom=0.0, hspace=0.25, wspace=0.1)
    # current_index=0
    # for i in range(num_rows):
        # for j in range(num_cols):
            # axs[i,j].imshow(image_files[current_index])
            # axs[i,j].axes.axes.set_xlabel(image_names[current_index], fontsize=np.int(15))
            # axs[i,j].axes.xaxis.set_ticks([])
            # axs[i,j].axes.xaxis.set_ticklabels([])
            # axs[i,j].axes.yaxis.set_visible(False)
            # axs[i,j].spines['bottom'].set_color('black')
            # axs[i,j].spines['top'].set_color('black')
            # axs[i,j].spines['left'].set_color('black')
            # axs[i,j].spines['right'].set_color('black')
            # axs[i,j].spines['bottom'].set_linewidth(1)
            # axs[i,j].spines['top'].set_linewidth(1)
            # axs[i,j].spines['left'].set_linewidth(1)
            # axs[i,j].spines['right'].set_linewidth(1)
            # current_index+=1        

    # return image_files, np.array(image_names), [fig, axs]
  
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
    
def Show_Info(feature_options):
    st.sidebar.markdown('#### Questions List:')
    st.sidebar.write(feature_options)

def Reload_data():
    path_info='D:/Datasets/Celeba/'
    first_image=1
    num_cols=5
    num_rows=3
    n_images=num_cols*num_rows
    current_querys=['A picture of a man','A picture of a woman']
    n_tokens,clip_tokens,clip_device,clip_model, clip_transform, clip_text = Token_process_query(current_querys)
    current_images, current_image_names, current_figure =Load_Images_randomly(num_rows,num_cols)
    Data_Init['init_data'][0]['n_images']=n_images
    Data_Init['init_data'][0]['num_rows']=num_rows
    Data_Init['init_data'][0]['num_cols']=num_cols
    Data_Init['init_data'][0]['current_images']=current_images
    Data_Init['init_data'][0]['current_images_discarted']=np.zeros((n_images))
    Data_Init['init_data'][0]['current_image_names']=current_image_names
    Data_Init['init_data'][0]['current_figure']=current_figure
    Data_Init['init_data'][0]['model_changing']=False
    Data_Init['init_data'][0]['show_results']=False
    Data_Init['init_data'][0]['start_game']=False
    Data_Init['init_data'][0]['finished_game']=0
    Data_Init['init_data'][0]['questions_index']=0
    Data_Init['init_data'][0]['award']=100
    Data_Init['init_data'][0]['first_image']=first_image
    Data_Init['init_data'][0]['current_winner_index']=-1
    Data_Init['init_data'][0]['n_tokens']=n_tokens
    Data_Init['init_data'][0]['current_querys']=current_querys
    Data_Init['init_data'][0]['clip_tokens']=clip_tokens
    Data_Init['init_data'][0]['clip_device']=clip_device
    Data_Init['init_data'][0]['clip_model']=clip_model,
    Data_Init['init_data'][0]['clip_transform']=clip_transform,
    Data_Init['init_data'][0]['clip_text']=clip_text
    Data_Init['init_data'][0]['path_info']=path_info
    Data_Init['init_data'][0]['path_imgs']='D:/Datasets/Celeba/img_celeba/'
    Data_Init['init_data'][0]['function_predict']=Predict_0_vs_1
    Data_Init['init_data'][0]['token_type']=0
    Data_Init['init_data'][0]['image_current_probs']=np.zeros((num_cols*num_rows,n_tokens))
    Data_Init['init_data'][0]['image_current_predictions']=np.zeros((num_cols*num_rows))+2
    Data_Init['init_data'][0]['selected_feature']='Ask a Question'
    Data_Init['init_data'][0]['selected_question']='Ask a Question'
    Data_Init['init_data'][0]['user_input']='A picture of a person'
    Data_Init['init_data'][0]['user_input_querys1']='A picture of a man'
    Data_Init['init_data'][0]['user_input_querys2']='A picture of a woman'
    Data_Init['init_data'][0]['querys_list']=['A picture of a man', 'A picture of a woman', 'A picture of an attractive person', 'A picture of a young person', 
            'A picture of a person with receding hairline', 'A picture of a chubby person ', 'A picture of a person who is smiling', 'A picture of a bald person',
            'A picture of a person with black hair', 'A picture of a person with brown hair', 'A picture of a person with blond hair', 'A picture of a person with red hair', 
            'A picture of a person with gray hair', 'A picture of a person with straight hair', 'A picture of a person with wavy hair', 
            'A picture of a person who does not wear a beard', 'A picture of a person with mustache', 'A picture of a person with sideburns', 
            'A picture of a person with goatee', 'A picture of a person with heavy makeup', 'A picture of a person with eyeglasses ',             
            'A picture of a person with bushy eyebrows', 'A picture of a person with a double chin', 
            'A picture of a person with high cheekbones', 'A picture of a person with slightly open mouth', 
            'A picture of a person with narrow eyes', 'A picture of a person with an oval face', 
            'A picture of a person wiht pale skin', 'A picture of a person with pointy nose', 'A picture of a person with rosy cheeks', 
            "A picture of a person with five o'clock shadow", 'A picture of a person with arched eyebrows', 'A picture of a person with bags under the eyes', 
            'A picture of a person with bangs', 'A picture of a person with big lips', 'A picture of a person with big nose',            
            'A picture of a person with earrings', 'A picture of a person with hat', 
            'A picture of a person with lipstick', 'A picture of a person with necklace', 
            'A picture of a person with necktie', 'A blurry picture of a person'
            ]
    Data_Init['init_data'][0]['feature_questions']=['Are you a MAN?', 'Are you a WOMAN?', 'Are you an ATTRACTIVE person?', 'Are you YOUNG?',
                    'Are you a person with RECEDING HAIRLINES?', 'Are you a CHUBBY person?', 'Are you SMILING?','Are you BALD?', 
                    'Do you have BLACK HAIR?', 'Do you have BROWN HAIR?', 'Do you have BLOND HAIR?', 'Do you have RED HAIR?',
                    'Do you have GRAY HAIR?', 'Do you have STRAIGHT HAIR?', 'Do you have WAVY HAIR?',
                    'Do you have a BEARD?', 'Do you have a MUSTACHE?', 'Do you have SIDEBURNS?',
                    'Do you have a GOATEE?', 'Do you wear HEAVY MAKEUP?', 'Do you wear EYEGLASSES?',
                    'Do you have BUSHY EYEBROWS?', 'Do you have a DOUBLE CHIN?', 
                    'Do you have a high CHEECKBONES?', 'Do you have SLIGHTLY OPEN MOUTH?', 
                    'Do you have NARROWED EYES?', 'Do you have an OVAL FACE?', 
                    'Do you have PALE SKIN?', 'Do you have a POINTY NOSE?', 'Do you have ROSY CHEEKS?', 
                    "Do you have FIVE O'CLOCK SHADOW?", 'Do you have ARCHED EYEBROWS?', 'Do you have BUGS UNDER your EYES?', 
                    'Do you have BANGS?', 'Do you have a BIG LIPS?', 'Do you have a BIG NOSE?',
                    'Are you wearing EARRINGS?', 'Are you wearing a HAT?', 
                    'Are you wearing LIPSTICK?', 'Are you wearing NECKLACE?', 
                    'Are you wearing NECKTIE?', 'Is your image BLURRY?'
                    ]
  
# ---------------   CACHE   ---------------

@st.cache(allow_output_mutation=True,max_entries=10,ttl=3600) 
def load_data():
    path_info='D:/Datasets/Celeba/'
    first_image=1
    num_cols=5
    num_rows=3
    n_images=num_cols*num_rows
    current_querys=['A picture of a man','A picture of a woman']
    n_tokens,clip_tokens,clip_device,clip_model, clip_transform, clip_text = Token_process_query(current_querys)
    current_images, current_image_names, current_figure =Load_Images_randomly(num_rows,num_cols)
    
    Saved_data={
        'n_images':n_images,
        'num_rows':num_rows,
        'num_cols':num_cols,
        'current_images':current_images,
        'current_images_discarted':np.zeros((n_images)),
        'current_image_names':current_image_names,
        'current_figure':current_figure,
        'model_changing':False,
        'show_results':False,
        'start_game':False,
        'finished_game':0,
        'questions_index':0,
        'award':100,
        'first_image':first_image,
        'current_winner_index':-1,
        'n_tokens':n_tokens,
        'current_querys':current_querys,
        'clip_tokens':clip_tokens,
        'clip_device':clip_device,
        'clip_model':clip_model, 
        'clip_transform':clip_transform, 
        'clip_text':clip_text,
        'path_info':path_info,
        'path_imgs':'D:/Datasets/Celeba/img_celeba/',
        'function_predict':Predict_0_vs_1,
        'token_type':0,
        'image_current_probs':np.zeros((num_cols*num_rows,n_tokens)),
        'image_current_predictions':np.zeros((num_cols*num_rows))+2,
        'selected_feature':'Ask a Question',
        'selected_question':'Are you a MAN?',
        'user_input':'A picture of a person',
        'user_input_querys1':'A picture of a man',
        'user_input_querys2':'A picture of a woman',
        'querys_list':['A picture of a man', 'A picture of a woman', 'A picture of an attractive person', 'A picture of a young person', 
            'A picture of a person with receding hairline', 'A picture of a chubby person ', 'A picture of a person who is smiling', 'A picture of a bald person',
            'A picture of a person with black hair', 'A picture of a person with brown hair', 'A picture of a person with blond hair', 'A picture of a person with red hair', 
            'A picture of a person with gray hair', 'A picture of a person with straight hair', 'A picture of a person with wavy hair', 
            'A picture of a person who does not wear a beard', 'A picture of a person with mustache', 'A picture of a person with sideburns', 
            'A picture of a person with goatee', 'A picture of a person with heavy makeup', 'A picture of a person with eyeglasses ',             
            'A picture of a person with bushy eyebrows', 'A picture of a person with a double chin', 
            'A picture of a person with high cheekbones', 'A picture of a person with slightly open mouth', 
            'A picture of a person with narrow eyes', 'A picture of a person with an oval face', 
            'A picture of a person wiht pale skin', 'A picture of a person with pointy nose', 'A picture of a person with rosy cheeks', 
            "A picture of a person with five o'clock shadow", 'A picture of a person with arched eyebrows', 'A picture of a person with bags under the eyes', 
            'A picture of a person with bangs', 'A picture of a person with big lips', 'A picture of a person with big nose',            
            'A picture of a person with earrings', 'A picture of a person with hat', 
            'A picture of a person with lipstick', 'A picture of a person with necklace', 
            'A picture of a person with necktie', 'A blurry picture of a person'
            ],
        'feature_questions':['Are you a MAN?', 'Are you a WOMAN?', 'Are you an ATTRACTIVE person?', 'Are you YOUNG?',
                    'Are you a person with RECEDING HAIRLINES?', 'Are you a CHUBBY person?', 'Are you SMILING?','Are you BALD?', 
                    'Do you have BLACK HAIR?', 'Do you have BROWN HAIR?', 'Do you have BLOND HAIR?', 'Do you have RED HAIR?',
                    'Do you have GRAY HAIR?', 'Do you have STRAIGHT HAIR?', 'Do you have WAVY HAIR?',
                    'Do you have a BEARD?', 'Do you have a MUSTACHE?', 'Do you have SIDEBURNS?',
                    'Do you have a GOATEE?', 'Do you wear HEAVY MAKEUP?', 'Do you wear EYEGLASSES?',
                    'Do you have BUSHY EYEBROWS?', 'Do you have a DOUBLE CHIN?', 
                    'Do you have a high CHEECKBONES?', 'Do you have SLIGHTLY OPEN MOUTH?', 
                    'Do you have NARROWED EYES?', 'Do you have an OVAL FACE?', 
                    'Do you have PALE SKIN?', 'Do you have a POINTY NOSE?', 'Do you have ROSY CHEEKS?', 
                    "Do you have FIVE O'CLOCK SHADOW?", 'Do you have ARCHED EYEBROWS?', 'Do you have BUGS UNDER your EYES?', 
                    'Do you have BANGS?', 'Do you have a BIG LIPS?', 'Do you have a BIG NOSE?',
                    'Are you wearing EARRINGS?', 'Are you wearing a HAT?', 
                    'Are you wearing LIPSTICK?', 'Are you wearing NECKLACE?', 
                    'Are you wearing NECKTIE?', 'Is your image BLURRY?'
                    ]
    }
    return {"init_data": [Saved_data]}    

st.set_page_config(
    layout="wide",
    page_title='QuienEsQuien'
    # page_icon='gz_icon.jpeg'
)


## --------------- PROGRAMA ---------------
Data_Init=load_data() 
            
Feature_Options=['Ask a Question', 'Create your own query', 'Create your own 2 querys','Select a Winner']


## TITLE
# st.title('Guess Who?                         SCORE: '+ str(Data_Init['init_data'][0]['award']))
st.markdown("<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1><h2 style='text-align:right; float:right; color:gray; margin:0px;'>score: "+ str(Data_Init['init_data'][0]['award'])+"</h2>", unsafe_allow_html=True)

## SIDEBAR TITLE
st.sidebar.markdown('# OPTIONS PANEL')


## RESET APP
## RESET APP
Reset_App = st.sidebar.button('RESET GAME', key='Reset_App')

if Reset_App:
    Reload_data()
    Restart_App = st.button('GO TO IMAGES SELECTION TO START A NEW GAME', key='Restart_App')
    
else:                    
    ## FINISHED GAME BUTTON TO RELOAD GAME
    if Data_Init['init_data'][0]['finished_game']==99:
        Restart_App = st.button('GO TO IMAGES SELECTION TO START NEW GAME', key='Restart_App')

    Use_Images_Selected=False
    ## INITIALIZATION (SELECTING FIGURES)
    if (not Data_Init['init_data'][0]['start_game']) and (Data_Init['init_data'][0]['finished_game']==0):
        
        ## Select imagenes text
        st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>1. Choose the images you like.</h2>", unsafe_allow_html=True)
     
        ## celeba number - text
        st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to randomly modify the selected images.</h3>", unsafe_allow_html=True)
        
        ## celeba randomly - change button
        Random_Images = st.button('CHANGE IMAGES', key='Random_Images')
        if Random_Images:
            [ Data_Init['init_data'][0]['current_images'],
              Data_Init['init_data'][0]['current_image_names'], 
              Data_Init['init_data'][0]['current_figure'] ] = Load_Images_randomly(Data_Init['init_data'][0]['num_rows'],
                                                                                    Data_Init['init_data'][0]['num_cols'])
            Data_Init['init_data'][0]['model_changing']=True
                
        ## Start game button
        st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>2. Press the button to start playing.</h2>", unsafe_allow_html=True)
        Use_Images = st.button('START GAME', key='Use_Images')
        
        if Use_Images:
            ## Choose winner and start game
            Data_Init['init_data'][0]['current_winner_index']=random.choice(list(range(0,Data_Init['init_data'][0]['n_images'])))
            Data_Init['init_data'][0]['start_game']=True
            Data_Init['init_data'][0]['model_changing']=True
            Use_Images_Selected=True
            
    ## --- RUNNING GAME ---
    if (Data_Init['init_data'][0]['start_game']) and (Data_Init['init_data'][0]['finished_game']==0):
        ## SELECTING QUERY TYPE
        if Use_Images_Selected:
            st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>3. Select a type of Query to play.</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>1. Select a type of Query to play.</h2>", unsafe_allow_html=True)

        Selected_Feature=st.selectbox('Ask a question from a list, create your query or select a winner:', Feature_Options, 
                                               # index=Feature_Options.index(Data_Init['init_data'][0]['selected_feature']), 
                                               index=0, 
                                               key='selected_feature', help=None)     
        if Data_Init['init_data'][0]['selected_feature']!=Selected_Feature and not Data_Init['init_data'][0]['show_results']:
            Data_Init['init_data'][0]['selected_feature']=Selected_Feature
            Data_Init['init_data'][0]['model_changing']=True
            
        ## Select query - elements to show (questions)
        if Data_Init['init_data'][0]['selected_feature']=='Ask a Question':
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Select a Question from the list.</h3>", unsafe_allow_html=True)
            if Data_Init['init_data'][0]['questions_index']<len(Data_Init['init_data'][0]['feature_questions']):
                current_questions_index=Data_Init['init_data'][0]['questions_index']
            else:
                current_questions_index=0
            Selected_Question=st.selectbox('Suggested questions:', Data_Init['init_data'][0]['feature_questions'], 
                                               index=0,
                                               key='selected_question', help=None)
            
            if Selected_Question!=Data_Init['init_data'][0]['selected_question'] and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['selected_question']=Selected_Question
                Data_Init['init_data'][0]['questions_index']=Data_Init['init_data'][0]['feature_questions'].index(Selected_Question)
                Data_Init['init_data'][0]['model_changing']=True
                
            st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>Current Question: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+Data_Init['init_data'][0]['selected_question']+"</h3>", unsafe_allow_html=True)
            Use_Query = st.button('USE THIS QUESTION', key='Use_Query')

            if Data_Init['init_data'][0]['selected_question']=='Are you bald?' and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['current_querys']=['A picture of a person','A picture of a man','A picture of a woman',
                                                            'A picture of a yes bald man','A picture of a bald person']
                Data_Init['init_data'][0]['token_type']=0
                Data_Init['init_data'][0]['function_predict']=Predict_bald
                
            elif Data_Init['init_data'][0]['selected_question']=='Do you have BLACK HAIR?' and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['current_querys']=['A picture of a person who is black-haired',
                                                            'A picture of a person who is tawny-haired',
                                                            'A picture of a person who is blond-haired',
                                                            'A picture of a person who is gray-haired',
                                                            'A picture of a person who is red-haired',
                                                            'A picture of a person who is totally bald']
                Data_Init['init_data'][0]['token_type']=0
                Data_Init['init_data'][0]['function_predict']=Predict_hair_color

            elif Data_Init['init_data'][0]['selected_question']=='Do you have BROWN HAIR?' and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['current_querys']=['A picture of a person who is tawny-haired',
                                                            'A picture of a person who is black-haired',
                                                            'A picture of a person who is blond-haired',
                                                            'A picture of a person who is gray-haired',
                                                            'A picture of a person who is red-haired',
                                                            'A picture of a person who is totally bald']
                Data_Init['init_data'][0]['token_type']=0
                Data_Init['init_data'][0]['function_predict']=Predict_hair_color

            elif Data_Init['init_data'][0]['selected_question']=='Do you have BLOND HAIR?' and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['current_querys']=['A picture of a person who is blond-haired',
                                                            'A picture of a person who is tawny-haired',
                                                            'A picture of a person who is black-haired',
                                                            'A picture of a person who is gray-haired',
                                                            'A picture of a person who is red-haired',
                                                            'A picture of a person who is totally bald']
                Data_Init['init_data'][0]['token_type']=0
                Data_Init['init_data'][0]['function_predict']=Predict_hair_color
                
            elif Data_Init['init_data'][0]['selected_question']=='Do you have RED HAIR?' and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['current_querys']=['A picture of a person who is red-haired',
                                                            'A picture of a person who is tawny-haired',
                                                            'A picture of a person who is blond-haired',
                                                            'A picture of a person who is gray-haired',
                                                            'A picture of a person who is black-haired',
                                                            'A picture of a person who is totally bald']
                Data_Init['init_data'][0]['token_type']=0
                Data_Init['init_data'][0]['function_predict']=Predict_hair_color
                
            elif Data_Init['init_data'][0]['selected_question']=='Do you have GRAY HAIR?' and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['current_querys']=['A picture of a person who is gray-haired',
                                                            'A picture of a person who is tawny-haired',
                                                            'A picture of a person who is blond-haired',
                                                            'A picture of a person who is black-haired',
                                                            'A picture of a person who is red-haired',
                                                            'A picture of a person who is totally bald']
                Data_Init['init_data'][0]['token_type']=0
                Data_Init['init_data'][0]['function_predict']=Predict_hair_color
                
            elif Data_Init['init_data'][0]['selected_question']=='Are you a man?' and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['current_querys']=['A picture of a man','A picture of a woman']
                Data_Init['init_data'][0]['token_type']=0
                Data_Init['init_data'][0]['function_predict']=Predict_0_vs_1    
                
            elif Data_Init['init_data'][0]['selected_question']=='Are you a woman?' and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['current_querys']=['A picture of a woman','A picture of a man']
                Data_Init['init_data'][0]['token_type']=0
                Data_Init['init_data'][0]['function_predict']=Predict_0_vs_1         
                
            elif Data_Init['init_data'][0]['selected_question']=='Do you have a beard?' and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['current_querys']=['A picture of a person with beard','A picture of a person']
                Data_Init['init_data'][0]['token_type']=0
                Data_Init['init_data'][0]['function_predict']=Predict_0_vs_1
                
            elif Data_Init['init_data'][0]['selected_question']=='Are you YOUNG?' and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['current_querys']=['A picture of a young person','A picture of an aged person']
                Data_Init['init_data'][0]['token_type']=0
                Data_Init['init_data'][0]['function_predict']=Predict_0_vs_1

            elif  not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['current_querys']=[Data_Init['init_data'][0]['querys_list'][Data_Init['init_data'][0]['questions_index']],'A picture of a person']
                Data_Init['init_data'][0]['token_type']=0
                Data_Init['init_data'][0]['function_predict']=Predict_0_vs_1


        ## Select query - elements to show (1 user query)
        if Data_Init['init_data'][0]['selected_feature']=='Create your own query':
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Write your own query and press the button.</h3>", unsafe_allow_html=True)
            User_Input = st.text_input('It is recommended to use a text like: "A picture of a ... person" or "A picture of a person ..." (CLIP will check -> "Your query"  vs  "A picture of a person" )', Data_Init['init_data'][0]['user_input'], key='User_Input', help=None)
            
            if Data_Init['init_data'][0]['user_input']!=User_Input and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['user_input']=User_Input
                Data_Init['init_data'][0]['model_changing']=True
            
            Check_Query = st.button('USE MY QUERY:   '+Data_Init['init_data'][0]['user_input'], key='Check_Query')
            Data_Init['init_data'][0]['token_type']=-1
                

        ## Select query - elements to show (2 user querys)
        if Data_Init['init_data'][0]['selected_feature']=='Create your own 2 querys':
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Write your own querys by introducing 2 opposite descriptions.</h3>", unsafe_allow_html=True)
            User_Input_Querys1 = st.text_input('Write your "True" query:', Data_Init['init_data'][0]['user_input_querys1'],
                                                        key='User_Input_Querys1', help=None)
            User_Input_Querys2 = st.text_input('Write your "False" query:', Data_Init['init_data'][0]['user_input_querys2'],
                                                        key='User_Input_Querys2', help=None)
            Check_Querys = st.button('USE MY QUERYS:   '+User_Input_Querys1+' vs '+User_Input_Querys2, key='Check_Querys')
            Data_Init['init_data'][0]['token_type']=-2
            if Data_Init['init_data'][0]['user_input_querys1']!=User_Input_Querys1 and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['user_input_querys1']=User_Input_Querys1
                Data_Init['init_data'][0]['model_changing']=True
                
            if Data_Init['init_data'][0]['user_input_querys2']!=User_Input_Querys2 and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['user_input_querys2']=User_Input_Querys2
                Data_Init['init_data'][0]['model_changing']=True


        ## Select query - elements to show (winner selection)
        if Data_Init['init_data'][0]['selected_feature']=='Select a Winner': 
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Select a Winner picture name.</h3>", unsafe_allow_html=True)
            Winner_Options=['Winner not selected']
            Winner_Options.extend(Data_Init['init_data'][0]['current_image_names'])
            Selected_Winner=st.selectbox('If you are inspired, Select a Winner image directly:', Winner_Options, 
                                                    index=0, key='Selected_Winner', help=None)
            Check_Winner = st.button('CHECK THIS WINNER', key='Check_Winner')
            Data_Init['init_data'][0]['token_type']=-3
            if Selected_Winner!='Winner not selected' and not Data_Init['init_data'][0]['show_results']:
                Data_Init['init_data'][0]['model_changing']=True

            
        ## ACTIONS IF NOT SHOWING RESULTS

        if not Data_Init['init_data'][0]['show_results']:
            # Ask question
            if Data_Init['init_data'][0]['selected_feature']=='Ask a Question':
                if Use_Query:
                    [ Data_Init['init_data'][0]['n_tokens'],
                      Data_Init['init_data'][0]['clip_tokens'],
                      Data_Init['init_data'][0]['clip_device'],
                      Data_Init['init_data'][0]["clip_model"],                
                      Data_Init['init_data'][0]['clip_transform'],
                      Data_Init['init_data'][0]['clip_text'] ]=Token_process_query(Data_Init['init_data'][0]['current_querys'])
                    Data_Init['init_data'][0]['image_current_probs'] = Token_img(Data_Init['init_data'][0]['n_images'],
                                                                                Data_Init['init_data'][0]['n_tokens'],
                                                                                Data_Init['init_data'][0]['current_images'],
                                                                                Data_Init['init_data'][0]['current_images_discarted'],
                                                                                Data_Init['init_data'][0]['clip_text'], 
                                                                                Data_Init['init_data'][0]["clip_model"], 
                                                                                Data_Init['init_data'][0]['clip_transform'], 
                                                                                Data_Init['init_data'][0]['clip_device'])
                    Data_Init['init_data'][0]['image_current_predictions']=Data_Init['init_data'][0]['function_predict'](Data_Init['init_data'][0]['image_current_probs'])
                    Data_Init['init_data'][0]['model_changing']=False
                    Data_Init['init_data'][0]['show_results']=True
                    
                    ## delete used question
                    if Data_Init['init_data'][0]['selected_question']=='Are you a MAN?' or Data_Init['init_data'][0]['selected_question']=='Are you a WOMAN?':
                        del Data_Init['init_data'][0]['querys_list'][0:2]
                        del Data_Init['init_data'][0]['feature_questions'][0:2]
                    else:
                        del Data_Init['init_data'][0]['querys_list'][Data_Init['init_data'][0]['questions_index']]
                        del Data_Init['init_data'][0]['feature_questions'][Data_Init['init_data'][0]['questions_index']]
                    Data_Init['init_data'][0]['questions_index']=0
                    
            # User 1 query
            if Data_Init['init_data'][0]['selected_feature']=='Create your own query':
                if Check_Query:
                    if Data_Init['init_data'][0]['user_input']!='A picture of a person':
                        Data_Init['init_data'][0]['current_querys']=['A Picture of a person',Data_Init['init_data'][0]['user_input']]
                        Data_Init['init_data'][0]['function_predict']=Predict_1_vs_0
                        [ Data_Init['init_data'][0]['n_tokens'],
                          Data_Init['init_data'][0]['clip_tokens'],
                          Data_Init['init_data'][0]['clip_device'],
                          Data_Init['init_data'][0]["clip_model"],
                          Data_Init['init_data'][0]['clip_transform'],
                          Data_Init['init_data'][0]['clip_text'] ]=Token_process_query(Data_Init['init_data'][0]['current_querys'])
                        Data_Init['init_data'][0]['image_current_probs'] = Token_img(Data_Init['init_data'][0]['n_images'],
                                                                                    Data_Init['init_data'][0]['n_tokens'],
                                                                                    Data_Init['init_data'][0]['current_images'],
                                                                                    Data_Init['init_data'][0]['current_images_discarted'],
                                                                                    Data_Init['init_data'][0]['clip_text'], 
                                                                                    Data_Init['init_data'][0]["clip_model"], 
                                                                                    Data_Init['init_data'][0]['clip_transform'], 
                                                                                    Data_Init['init_data'][0]['clip_device'])
                        Data_Init['init_data'][0]['image_current_predictions']=Data_Init['init_data'][0]['function_predict'](Data_Init['init_data'][0]['image_current_probs'])
                        Data_Init['init_data'][0]['model_changing']=False
                        Data_Init['init_data'][0]['show_results']=True
                    else:
                        st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Your query must be different of 'A picture of a person'.</h3>", unsafe_allow_html=True)

            # User 2 querys
            if Data_Init['init_data'][0]['selected_feature']=='Create your own 2 querys':
                if Check_Querys:
                    # st.sidebar.markdown('(new 2 own querys:)')
                    Data_Init['init_data'][0]['current_querys']=[Data_Init['init_data'][0]['user_input_querys1'],Data_Init['init_data'][0]['user_input_querys2']]     
                    Data_Init['init_data'][0]['function_predict']=Predict_0_vs_1
                    [ Data_Init['init_data'][0]['n_tokens'],
                      Data_Init['init_data'][0]['clip_tokens'],
                      Data_Init['init_data'][0]['clip_device'],
                      Data_Init['init_data'][0]["clip_model"],
                      Data_Init['init_data'][0]['clip_transform'],
                      Data_Init['init_data'][0]['clip_text'] ]=Token_process_query(Data_Init['init_data'][0]['current_querys'])
                    Data_Init['init_data'][0]['image_current_probs'] = Token_img(Data_Init['init_data'][0]['n_images'],
                                                                                Data_Init['init_data'][0]['n_tokens'],
                                                                                Data_Init['init_data'][0]['current_images'],
                                                                                Data_Init['init_data'][0]['current_images_discarted'],
                                                                                Data_Init['init_data'][0]['clip_text'], 
                                                                                Data_Init['init_data'][0]["clip_model"], 
                                                                                Data_Init['init_data'][0]['clip_transform'], 
                                                                                Data_Init['init_data'][0]['clip_device'])
                    Data_Init['init_data'][0]['image_current_predictions']=Data_Init['init_data'][0]['function_predict'](Data_Init['init_data'][0]['image_current_probs'])
                    Data_Init['init_data'][0]['model_changing']=False
                    Data_Init['init_data'][0]['show_results']=True

            ## Select Winner
            if Data_Init['init_data'][0]['selected_feature']=='Select a Winner':
                if Check_Winner:
                    Selected_Index=np.where(Selected_Winner==Data_Init['init_data'][0]['current_image_names'])[0]
                    Data_Init['init_data'][0]['image_current_predictions']=np.zeros(Data_Init['init_data'][0]['n_images'])
                    Data_Init['init_data'][0]['image_current_predictions'][Selected_Index]=1    
                    Data_Init['init_data'][0]['model_changing']=False
                    Data_Init['init_data'][0]['show_results']=True
            

        ## ACTIONS IF SHOWING RESULTS
        if Data_Init['init_data'][0]['show_results']:
            if not np.sum(Data_Init['init_data'][0]['current_images_discarted']==0)==1:
                if Use_Images_Selected: 
                    st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>4. Press the button to continue.</h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>2. Press the button to continue.</h2>", unsafe_allow_html=True)
                Next_Query=st.button('NEXT QUERY', key='Next_Query')
            
            if Data_Init['init_data'][0]['token_type']==0:
                if Data_Init['init_data'][0]['image_current_predictions'][Data_Init['init_data'][0]['current_winner_index']]:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+Data_Init['init_data'][0]['selected_question']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>YES</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+Data_Init['init_data'][0]['selected_question']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>NO</h3>", unsafe_allow_html=True)
                    
            if Data_Init['init_data'][0]['token_type']==-1:
                if Data_Init['init_data'][0]['image_current_predictions'][Data_Init['init_data'][0]['current_winner_index']]:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+Data_Init['init_data'][0]['user_input']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>TRUE</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+Data_Init['init_data'][0]['user_input']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>FALSE</h3>", unsafe_allow_html=True)
                    
            if Data_Init['init_data'][0]['token_type']==-2:
                if Data_Init['init_data'][0]['image_current_predictions'][Data_Init['init_data'][0]['current_winner_index']]:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>The most accurate query is:</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>"+Data_Init['init_data'][0]['user_input_querys1']+"</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>The most accurate query is:</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>"+Data_Init['init_data'][0]['user_input_querys2']+"</h3>", unsafe_allow_html=True)
              
            if Data_Init['init_data'][0]['token_type']==-3:
                if not Selected_Winner==Data_Init['init_data'][0]['current_image_names'][Data_Init['init_data'][0]['current_winner_index']]:
                    st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>The winner picture is not:</h3><h3 style='text-align:left; float:center; color:red; margin:0px;'>"+Selected_Winner+"</h3>", unsafe_allow_html=True)

                                  
    ## MODEL CHANGIG - RESET VALUES OF PREDICTIONS  
    if Data_Init['init_data'][0]['model_changing']:
        Data_Init['init_data'][0]['image_current_probs']=np.zeros((Data_Init['init_data'][0]['n_images'],Data_Init['init_data'][0]['n_tokens']))
        Data_Init['init_data'][0]['image_current_predictions']=np.zeros((Data_Init['init_data'][0]['n_images']))+2
        Data_Init['init_data'][0]['model_changing']=False
     
     
    ## CREATE IMAGES TO SHOW
    Figure_To_Show=Show_images(Data_Init['init_data'][0]['show_results'],
                                                Data_Init['init_data'][0]['current_figure'],
                                                Data_Init['init_data'][0]['current_images_discarted'],
                                                Data_Init['init_data'][0]['image_current_predictions'],
                                                Data_Init['init_data'][0]['current_winner_index'], 
                                                Data_Init['init_data'][0]['num_cols'],Data_Init['init_data'][0]['num_rows'],
                                                Data_Init['init_data'][0]['n_images'])

   ## APPLY DISCARDING
    if Data_Init['init_data'][0]['show_results']:        
        Data_Init['init_data'][0]['show_results']=False
        previous_images_discarted=len(Data_Init['init_data'][0]['current_images_discarted'])
        [ Data_Init['init_data'][0]['image_current_predictions'],
          Data_Init['init_data'][0]['current_images_discarted'],
          Data_Init['init_data'][0]['current_images'],
          Data_Init['init_data'][0]['current_image_names'],
          Data_Init['init_data'][0]['current_figure'],
          Data_Init['init_data'][0]['n_images'],
          Data_Init['init_data'][0]['current_winner_index'],
          Data_Init['init_data'][0]['num_rows'] ] = Image_discarding(Data_Init['init_data'][0]['image_current_predictions'],
                                                                                Data_Init['init_data'][0]['current_winner_index'],
                                                                                Data_Init['init_data'][0]['current_images_discarted'],
                                                                                Data_Init['init_data'][0]['n_images'],
                                                                                Data_Init['init_data'][0]['num_cols'],
                                                                                Data_Init['init_data'][0]['current_images'],
                                                                                Data_Init['init_data'][0]['current_image_names'])
                                                                                
        if len(Data_Init['init_data'][0]['current_images_discarted'])>1:
            Data_Init['init_data'][0]['award']=Data_Init['init_data'][0]['award']-len(Data_Init['init_data'][0]['current_images_discarted'])
        
        ## penalty when "select winner" option and few images remain
        if Data_Init['init_data'][0]['token_type']==-3:   
            Data_Init['init_data'][0]['award']=Data_Init['init_data'][0]['award']-1
            if previous_images_discarted<np.round(Data_Init['init_data'][0]['n_images']*0.8):
                Data_Init['init_data'][0]['award']=Data_Init['init_data'][0]['award']+previous_images_discarted-np.round(Data_Init['init_data'][0]['n_images']*0.8)

        ## penalty when no image is discarted
        if previous_images_discarted==len(Data_Init['init_data'][0]['current_images_discarted']):   
            Data_Init['init_data'][0]['award']=Data_Init['init_data'][0]['award']-2


    ## SHOW FINAL RESULTS
    if Data_Init['init_data'][0]['finished_game']==99:
        if Data_Init['init_data'][0]['award']==1 or Data_Init['init_data'][0]['award']==-1:
            st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>¡¡¡ FINISHED WITH</h1><h1 style='text-align:left; float:left; color:green; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>"+str(Data_Init['init_data'][0]['award'])+"</h1><h1 style='text-align:left; float:left; color:black; margin:0px;'>POINT !!!</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>¡¡¡ FINISHED WITH</h1><h1 style='text-align:left; float:left; color:green; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>"+str(Data_Init['init_data'][0]['award'])+"</h1><h1 style='text-align:left; float:left; color:black; margin:0px;'>POINTS !!!</h1>", unsafe_allow_html=True)
        Reload_data()    
            
    ## Finish game button
    if np.sum(Data_Init['init_data'][0]['current_images_discarted']==0)==1:
        Data_Init['init_data'][0]['finished_game']=99
        st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>You found the Winner picture:</h1><h1 style='text-align:left; float:left; color:green; margin:0px;'>"+Data_Init['init_data'][0]['current_image_names'][Data_Init['init_data'][0]['current_winner_index']]+"</h1>", unsafe_allow_html=True)
        Finsih_Game = st.button('FINISH GAME', key='Finsih_Game')

    ##SHOW EXTRA INFO
    Show_Info(Data_Init['init_data'][0]['feature_questions'])
        
    ## SHOW CURRENT
    st.write(Figure_To_Show[0])
    
