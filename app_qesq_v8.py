# -*- coding: utf-8 -*-

## Used Imports
import os
import io
import zipfile
import random
import torch
import numpy as np
import streamlit as st
import clip
import gc

from io import BytesIO
from PIL import Image
from zipfile import ZipFile 
    

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
def Token_img(n_images,n_tokens,current_image_files,current_images_discarted,clip_text, clip_model, clip_transform, clip_device):
    prediccion_probs=np.zeros((n_images,n_tokens))
    for i in range(n_images):
        prediccion_probs[i,:]=CLIP_get_probs_only(current_image_files[i], clip_text, clip_model, clip_transform, clip_device)

    return prediccion_probs
    
def CLIP_get_probs_only(img_file, img_txt, img_model, img_transf, img_device):
    img_proeprocessed = img_transf(Image.fromarray(img_file)).unsqueeze(0).to(img_device)
    img_features = img_model.encode_image(img_proeprocessed)
    txt_features = img_model.encode_text(img_txt)
    img_logits, img_logits_txt = img_model(img_proeprocessed, img_txt)
    image_p=img_logits.detach().numpy()[0]

    return np.round(image_p,2)

def Image_discarding(image_current_predictions,current_winner_index,current_images_discarted, n_images, image_files,image_names):
    for i in range(len(current_images_discarted)):
        if current_images_discarted[i]==0 and image_current_predictions[i]!=image_current_predictions[current_winner_index]:
            current_images_discarted[i]=1

    n_images2=np.sum(current_images_discarted==0)        
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
                               
    return image_current_predictions2, np.zeros(n_images2), image_files2, np.array(image_names2), n_images2, new_winner_index
    
def Show_images(show_results,current_image_files,current_images_discarted, image_current_predictions,
                current_winner_index, n_images,current_image_names):
    remaining_images=[]     
    for current_index in range(n_images):
        if show_results:
            current_line_width=4
            if image_current_predictions[current_index]==image_current_predictions[current_winner_index]:
                current_color=np.array([0,255,0])
            else:
                current_color=np.array([255,0,0]) 
        else:
            current_line_width=2
            current_color=np.zeros(3)  
                 
        remaining_images.append(Highlight_Image(current_image_files[current_index],current_line_width,current_color))
           
    return np.array(remaining_images)/255


def Highlight_Image(image,thickness,color):
    image_size=240
    w,h,c = np.shape(image)
    images_separation=image_size-w-thickness*2
    image_highlighted=np.zeros([h+thickness*2,image_size,c])+255
    image_highlighted[thickness:w+thickness,thickness:w+thickness,:]=image
    image_highlighted[:thickness,:w+2*thickness,:]=color
    image_highlighted[w+thickness:,:w+2*thickness,:]=color
    image_highlighted[:,w+thickness:w+2*thickness,:]=color
    image_highlighted[:,:thickness,:]=color
    return image_highlighted

def Load_Images_randomly(n_images):
    image_files=[]
    image_names=[]
    image_index=[]
        
    archive = zipfile.ZipFile('guess_who_images.zip', 'r')
    listOfFileNames = archive.namelist()        
    image_index_all=list(range(len(listOfFileNames)))
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
        image_files.append(np.array(Image.open(BytesIO(archive.read(image_current_path)))))
        image_names.append(image_current_path[-10:-4])
                
    return image_files, np.array(image_names)
   
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

# ---------------   CACHE   ---------------

# @st.cache(allow_output_mutation=True,max_entries=2,ttl=3600) 
def load_data():
    path_info='D:/Datasets/Celeba/'
    n_images=20
    current_querys=['A picture of a man','A picture of a woman']
    n_tokens,clip_tokens,clip_device,clip_model, clip_transform, clip_text = Token_process_query(current_querys)
    current_image_files, current_image_names =Load_Images_randomly(n_images)
    
    Init_Data={
        'n_images':n_images,
        'current_image_files':current_image_files,
        'current_images_discarted':np.zeros((n_images)),
        'current_image_names':current_image_names,
        'model_changing':False,
        'show_results':False,
        'start_game':False,
        'finished_game':0,
        'questions_index':0,
        'award':100,
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
        'image_current_probs':np.zeros((n_images,n_tokens)),
        'image_current_predictions':np.zeros((n_images))+2,
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
    return Init_Data
    
st.set_page_config(
    layout="wide",
    # page_icon='gz_icon.jpeg',
    page_title='QuienEsQuien',
    initial_sidebar_state="collapsed"
)


## --------------- PROGRAMA ---------------


# INITIALIZATIONS

## User sesion
if 'init_data' not in st.session_state:
    st.session_state['init_data'] = load_data()
 
Feature_Options=['Ask a Question', 'Create your own query', 'Create your own 2 querys','Select a Winner']

## Title
st.markdown("<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1><h2 style='text-align:right; float:right; color:gray; margin:0px;'>score: "+ str(st.session_state['init_data']['award'])+"</h2>", unsafe_allow_html=True)

## sidebar title
st.sidebar.markdown('# OPTIONS PANEL')


## RESET APP
Reset_App = st.sidebar.button('RESET GAME', key='Reset_App')

if Reset_App:
    st.session_state['init_data'] = load_data()
    Restart_App = st.button('GO TO IMAGES SELECTION TO START A NEW GAME', key='Restart_App')
    
else:                    
    ## FINISHED GAME BUTTON TO RELOAD GAME
    if st.session_state['init_data']['finished_game']==99:
        Restart_App = st.button('GO TO IMAGES SELECTION TO START NEW GAME', key='Restart_App')

    Use_Images_Selected=False
    
    ## INITIALIZATION (SELECTING FIGURES)
    if (not st.session_state['init_data']['start_game']) and (st.session_state['init_data']['finished_game']==0):
        
        ## Select imagenes text
        st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>1. Choose the images you like.</h2>", unsafe_allow_html=True)
     
        ## celeba number - text
        st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to randomly modify the selected images.</h3>", unsafe_allow_html=True)
        
        ## celeba randomly - change button
        Random_Images = st.button('CHANGE IMAGES', key='Random_Images')
        if Random_Images:
            [ st.session_state['init_data']['current_image_files'],
              st.session_state['init_data']['current_image_names'] ] = Load_Images_randomly(st.session_state['init_data']['n_images'])
            st.session_state['init_data']['model_changing']=True
                
        ## Start game button
        st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>2. Press the button to start playing.</h2>", unsafe_allow_html=True)
        Use_Images = st.button('START GAME', key='Use_Images')
        
        if Use_Images:
            ## Choose winner and start game
            st.session_state['init_data']['current_winner_index']=random.choice(list(range(0,st.session_state['init_data']['n_images'])))
            st.session_state['init_data']['start_game']=True
            st.session_state['init_data']['model_changing']=True
            Use_Images_Selected=True
            
    ## --- RUNNING GAME ---
    if (st.session_state['init_data']['start_game']) and (st.session_state['init_data']['finished_game']==0):
        ## SELECTING QUERY TYPE
        if Use_Images_Selected:
            st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>3. Select a type of Query to play.</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>1. Select a type of Query to play.</h2>", unsafe_allow_html=True)

        Selected_Feature=st.selectbox('Ask a question from a list, create your query or select a winner:', Feature_Options, 
                                               index=0, 
                                               key='selected_feature', help=None)     
        if st.session_state['init_data']['selected_feature']!=Selected_Feature and not st.session_state['init_data']['show_results']:
            st.session_state['init_data']['selected_feature']=Selected_Feature
            st.session_state['init_data']['model_changing']=True
            
        ## Select query - elements to show (questions)
        if st.session_state['init_data']['selected_feature']=='Ask a Question':
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Select a Question from the list.</h3>", unsafe_allow_html=True)
            if st.session_state['init_data']['questions_index']<len(st.session_state['init_data']['feature_questions']):
                current_questions_index=st.session_state['init_data']['questions_index']
            else:
                current_questions_index=0
            Selected_Question=st.selectbox('Suggested questions:', st.session_state['init_data']['feature_questions'], 
                                               index=0,
                                               key='selected_question', help=None)
            
            if Selected_Question!=st.session_state['init_data']['selected_question'] and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['selected_question']=Selected_Question
                st.session_state['init_data']['questions_index']=st.session_state['init_data']['feature_questions'].index(Selected_Question)
                st.session_state['init_data']['model_changing']=True
                
            st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>Current Question: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+st.session_state['init_data']['selected_question']+"</h3>", unsafe_allow_html=True)
            Use_Query = st.button('USE THIS QUESTION', key='Use_Query')

            if st.session_state['init_data']['selected_question']=='Are you bald?' and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['current_querys']=['A picture of a person','A picture of a man','A picture of a woman',
                                                            'A picture of a yes bald man','A picture of a bald person']
                st.session_state['init_data']['token_type']=0
                st.session_state['init_data']['function_predict']=Predict_bald
                
            elif st.session_state['init_data']['selected_question']=='Do you have BLACK HAIR?' and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['current_querys']=['A picture of a person who is black-haired',
                                                            'A picture of a person who is tawny-haired',
                                                            'A picture of a person who is blond-haired',
                                                            'A picture of a person who is gray-haired',
                                                            'A picture of a person who is red-haired',
                                                            'A picture of a person who is totally bald']
                st.session_state['init_data']['token_type']=0
                st.session_state['init_data']['function_predict']=Predict_hair_color

            elif st.session_state['init_data']['selected_question']=='Do you have BROWN HAIR?' and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['current_querys']=['A picture of a person who is tawny-haired',
                                                            'A picture of a person who is black-haired',
                                                            'A picture of a person who is blond-haired',
                                                            'A picture of a person who is gray-haired',
                                                            'A picture of a person who is red-haired',
                                                            'A picture of a person who is totally bald']
                st.session_state['init_data']['token_type']=0
                st.session_state['init_data']['function_predict']=Predict_hair_color

            elif st.session_state['init_data']['selected_question']=='Do you have BLOND HAIR?' and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['current_querys']=['A picture of a person who is blond-haired',
                                                            'A picture of a person who is tawny-haired',
                                                            'A picture of a person who is black-haired',
                                                            'A picture of a person who is gray-haired',
                                                            'A picture of a person who is red-haired',
                                                            'A picture of a person who is totally bald']
                st.session_state['init_data']['token_type']=0
                st.session_state['init_data']['function_predict']=Predict_hair_color
                
            elif st.session_state['init_data']['selected_question']=='Do you have RED HAIR?' and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['current_querys']=['A picture of a person who is red-haired',
                                                            'A picture of a person who is tawny-haired',
                                                            'A picture of a person who is blond-haired',
                                                            'A picture of a person who is gray-haired',
                                                            'A picture of a person who is black-haired',
                                                            'A picture of a person who is totally bald']
                st.session_state['init_data']['token_type']=0
                st.session_state['init_data']['function_predict']=Predict_hair_color
                
            elif st.session_state['init_data']['selected_question']=='Do you have GRAY HAIR?' and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['current_querys']=['A picture of a person who is gray-haired',
                                                            'A picture of a person who is tawny-haired',
                                                            'A picture of a person who is blond-haired',
                                                            'A picture of a person who is black-haired',
                                                            'A picture of a person who is red-haired',
                                                            'A picture of a person who is totally bald']
                st.session_state['init_data']['token_type']=0
                st.session_state['init_data']['function_predict']=Predict_hair_color
                
            elif st.session_state['init_data']['selected_question']=='Are you a man?' and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['current_querys']=['A picture of a man','A picture of a woman']
                st.session_state['init_data']['token_type']=0
                st.session_state['init_data']['function_predict']=Predict_0_vs_1    
                
            elif st.session_state['init_data']['selected_question']=='Are you a woman?' and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['current_querys']=['A picture of a woman','A picture of a man']
                st.session_state['init_data']['token_type']=0
                st.session_state['init_data']['function_predict']=Predict_0_vs_1         
                
            elif st.session_state['init_data']['selected_question']=='Do you have a beard?' and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['current_querys']=['A picture of a person with beard','A picture of a person']
                st.session_state['init_data']['token_type']=0
                st.session_state['init_data']['function_predict']=Predict_0_vs_1
                
            elif st.session_state['init_data']['selected_question']=='Are you YOUNG?' and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['current_querys']=['A picture of a young person','A picture of an aged person']
                st.session_state['init_data']['token_type']=0
                st.session_state['init_data']['function_predict']=Predict_0_vs_1

            elif  not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['current_querys']=[st.session_state['init_data']['querys_list'][st.session_state['init_data']['questions_index']],'A picture of a person']
                st.session_state['init_data']['token_type']=0
                st.session_state['init_data']['function_predict']=Predict_0_vs_1


        ## Select query - elements to show (1 user query)
        if st.session_state['init_data']['selected_feature']=='Create your own query':
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Write your own query and press the button.</h3>", unsafe_allow_html=True)
            User_Input = st.text_input('It is recommended to use a text like: "A picture of a ... person" or "A picture of a person ..." (CLIP will check -> "Your query"  vs  "A picture of a person" )', st.session_state['init_data']['user_input'], key='User_Input', help=None)
            
            if st.session_state['init_data']['user_input']!=User_Input and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['user_input']=User_Input
                st.session_state['init_data']['model_changing']=True
            
            Check_Query = st.button('USE MY QUERY:   '+st.session_state['init_data']['user_input'], key='Check_Query')
            st.session_state['init_data']['token_type']=-1
                

        ## Select query - elements to show (2 user querys)
        if st.session_state['init_data']['selected_feature']=='Create your own 2 querys':
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Write your own querys by introducing 2 opposite descriptions.</h3>", unsafe_allow_html=True)
            User_Input_Querys1 = st.text_input('Write your "True" query:', st.session_state['init_data']['user_input_querys1'],
                                                        key='User_Input_Querys1', help=None)
            User_Input_Querys2 = st.text_input('Write your "False" query:', st.session_state['init_data']['user_input_querys2'],
                                                        key='User_Input_Querys2', help=None)
            Check_Querys = st.button('USE MY QUERYS:   '+User_Input_Querys1+' vs '+User_Input_Querys2, key='Check_Querys')
            st.session_state['init_data']['token_type']=-2
            if st.session_state['init_data']['user_input_querys1']!=User_Input_Querys1 and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['user_input_querys1']=User_Input_Querys1
                st.session_state['init_data']['model_changing']=True
                
            if st.session_state['init_data']['user_input_querys2']!=User_Input_Querys2 and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['user_input_querys2']=User_Input_Querys2
                st.session_state['init_data']['model_changing']=True


        ## Select query - elements to show (winner selection)
        if st.session_state['init_data']['selected_feature']=='Select a Winner': 
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Select a Winner picture name.</h3>", unsafe_allow_html=True)
            Winner_Options=['Winner not selected']
            Winner_Options.extend(st.session_state['init_data']['current_image_names'])
            Selected_Winner=st.selectbox('If you are inspired, Select a Winner image directly:', Winner_Options, 
                                                    index=0, key='Selected_Winner', help=None)
            Check_Winner = st.button('CHECK THIS WINNER', key='Check_Winner')
            st.session_state['init_data']['token_type']=-3
            if Selected_Winner!='Winner not selected' and not st.session_state['init_data']['show_results']:
                st.session_state['init_data']['model_changing']=True

            
        ## ACTIONS IF NOT SHOWING RESULTS

        if not st.session_state['init_data']['show_results']:
            # Ask question
            if st.session_state['init_data']['selected_feature']=='Ask a Question':
                if Use_Query:
                    [ st.session_state['init_data']['n_tokens'],
                      st.session_state['init_data']['clip_tokens'],
                      st.session_state['init_data']['clip_device'],
                      st.session_state['init_data']['clip_model'],                
                      st.session_state['init_data']['clip_transform'],
                      st.session_state['init_data']['clip_text'] ]=Token_process_query(st.session_state['init_data']['current_querys'])
                    st.session_state['init_data']['image_current_probs'] = Token_img(st.session_state['init_data']['n_images'],
                                                                                st.session_state['init_data']['n_tokens'],
                                                                                st.session_state['init_data']['current_image_files'],
                                                                                st.session_state['init_data']['current_images_discarted'],
                                                                                st.session_state['init_data']['clip_text'], 
                                                                                st.session_state['init_data']['clip_model'], 
                                                                                st.session_state['init_data']['clip_transform'], 
                                                                                st.session_state['init_data']['clip_device'])
                    st.session_state['init_data']['image_current_predictions']=st.session_state['init_data']['function_predict'](st.session_state['init_data']['image_current_probs'])
                    st.session_state['init_data']['model_changing']=False
                    st.session_state['init_data']['show_results']=True
                    
                    ## delete used question
                    if st.session_state['init_data']['selected_question']=='Are you a MAN?' or st.session_state['init_data']['selected_question']=='Are you a WOMAN?':
                        del st.session_state['init_data']['querys_list'][0:2]
                        del st.session_state['init_data']['feature_questions'][0:2]
                    else:
                        del st.session_state['init_data']['querys_list'][st.session_state['init_data']['questions_index']]
                        del st.session_state['init_data']['feature_questions'][st.session_state['init_data']['questions_index']]
                    st.session_state['init_data']['questions_index']=0
                    
            # User 1 query
            if st.session_state['init_data']['selected_feature']=='Create your own query':
                if Check_Query:
                    if st.session_state['init_data']['user_input']!='A picture of a person':
                        st.session_state['init_data']['current_querys']=['A Picture of a person',st.session_state['init_data']['user_input']]
                        st.session_state['init_data']['function_predict']=Predict_1_vs_0
                        [ st.session_state['init_data']['n_tokens'],
                          st.session_state['init_data']['clip_tokens'],
                          st.session_state['init_data']['clip_device'],
                          st.session_state['init_data']['clip_model'],
                          st.session_state['init_data']['clip_transform'],
                          st.session_state['init_data']['clip_text'] ]=Token_process_query(st.session_state['init_data']['current_querys'])
                        st.session_state['init_data']['image_current_probs'] = Token_img(st.session_state['init_data']['n_images'],
                                                                                    st.session_state['init_data']['n_tokens'],
                                                                                    st.session_state['init_data']['current_image_files'],
                                                                                    st.session_state['init_data']['current_images_discarted'],
                                                                                    st.session_state['init_data']['clip_text'], 
                                                                                    st.session_state['init_data']['clip_model'], 
                                                                                    st.session_state['init_data']['clip_transform'], 
                                                                                    st.session_state['init_data']['clip_device'])
                        st.session_state['init_data']['image_current_predictions']=st.session_state['init_data']['function_predict'](st.session_state['init_data']['image_current_probs'])
                        st.session_state['init_data']['model_changing']=False
                        st.session_state['init_data']['show_results']=True
                    else:
                        st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Your query must be different of 'A picture of a person'.</h3>", unsafe_allow_html=True)

            # User 2 querys
            if st.session_state['init_data']['selected_feature']=='Create your own 2 querys':
                if Check_Querys:
                    st.session_state['init_data']['current_querys']=[st.session_state['init_data']['user_input_querys1'],st.session_state['init_data']['user_input_querys2']]     
                    st.session_state['init_data']['function_predict']=Predict_0_vs_1
                    [ st.session_state['init_data']['n_tokens'],
                      st.session_state['init_data']['clip_tokens'],
                      st.session_state['init_data']['clip_device'],
                      st.session_state['init_data']['clip_model'],
                      st.session_state['init_data']['clip_transform'],
                      st.session_state['init_data']['clip_text'] ]=Token_process_query(st.session_state['init_data']['current_querys'])
                    st.session_state['init_data']['image_current_probs'] = Token_img(st.session_state['init_data']['n_images'],
                                                                                st.session_state['init_data']['n_tokens'],
                                                                                st.session_state['init_data']['current_image_files'],
                                                                                st.session_state['init_data']['current_images_discarted'],
                                                                                st.session_state['init_data']['clip_text'], 
                                                                                st.session_state['init_data']['clip_model'], 
                                                                                st.session_state['init_data']['clip_transform'], 
                                                                                st.session_state['init_data']['clip_device'])
                    st.session_state['init_data']['image_current_predictions']=st.session_state['init_data']['function_predict'](st.session_state['init_data']['image_current_probs'])
                    st.session_state['init_data']['model_changing']=False
                    st.session_state['init_data']['show_results']=True

            ## Select Winner
            if st.session_state['init_data']['selected_feature']=='Select a Winner':
                if Check_Winner:
                    Selected_Index=np.where(Selected_Winner==st.session_state['init_data']['current_image_names'])[0]
                    st.session_state['init_data']['image_current_predictions']=np.zeros(st.session_state['init_data']['n_images'])
                    st.session_state['init_data']['image_current_predictions'][Selected_Index]=1    
                    st.session_state['init_data']['model_changing']=False
                    st.session_state['init_data']['show_results']=True
            

        ## ACTIONS IF SHOWING RESULTS
        if st.session_state['init_data']['show_results']:
            if not np.sum(st.session_state['init_data']['current_images_discarted']==0)==1:
                if Use_Images_Selected: 
                    st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>4. Press the button to continue.</h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>2. Press the button to continue.</h2>", unsafe_allow_html=True)
                Next_Query=st.button('NEXT QUERY', key='Next_Query')
            
            if st.session_state['init_data']['token_type']==0:
                if st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index']]:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['selected_question']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>YES</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['selected_question']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>NO</h3>", unsafe_allow_html=True)
                    
            if st.session_state['init_data']['token_type']==-1:
                if st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index']]:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['user_input']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>TRUE</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['user_input']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>FALSE</h3>", unsafe_allow_html=True)
                    
            if st.session_state['init_data']['token_type']==-2:
                if st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index']]:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>The most accurate query is:</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>"+st.session_state['init_data']['user_input_querys1']+"</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>The most accurate query is:</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>"+st.session_state['init_data']['user_input_querys2']+"</h3>", unsafe_allow_html=True)
              
            if st.session_state['init_data']['token_type']==-3:
                if not Selected_Winner==st.session_state['init_data']['current_image_names'][st.session_state['init_data']['current_winner_index']]:
                    st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>The winner picture is not:</h3><h3 style='text-align:left; float:center; color:red; margin:0px;'>"+Selected_Winner+"</h3>", unsafe_allow_html=True)

                                  
    ## MODEL CHANGIG - RESET VALUES OF PREDICTIONS  
    if st.session_state['init_data']['model_changing']:
        st.session_state['init_data']['image_current_probs']=np.zeros((st.session_state['init_data']['n_images'],st.session_state['init_data']['n_tokens']))
        st.session_state['init_data']['image_current_predictions']=np.zeros((st.session_state['init_data']['n_images']))+2
        st.session_state['init_data']['model_changing']=False
     
     
    ## CREATE IMAGES TO SHOW
    Remaining_Images=Show_images(st.session_state['init_data']['show_results'],
                                                st.session_state['init_data']['current_image_files'],
                                                st.session_state['init_data']['current_images_discarted'],
                                                st.session_state['init_data']['image_current_predictions'],
                                                st.session_state['init_data']['current_winner_index'], 
                                                st.session_state['init_data']['n_images'],st.session_state['init_data']['current_image_names'])
    Image_Names=st.session_state['init_data']['current_image_names']

   ## APPLY DISCARDING
    if st.session_state['init_data']['show_results']:        
        st.session_state['init_data']['show_results']=False
        previous_images_discarted=len(st.session_state['init_data']['current_images_discarted'])
        [ st.session_state['init_data']['image_current_predictions'],
          st.session_state['init_data']['current_images_discarted'],
          st.session_state['init_data']['current_image_files'],
          st.session_state['init_data']['current_image_names'],
          st.session_state['init_data']['n_images'],
          st.session_state['init_data']['current_winner_index'] ] = Image_discarding(st.session_state['init_data']['image_current_predictions'],
                                                                                st.session_state['init_data']['current_winner_index'],
                                                                                st.session_state['init_data']['current_images_discarted'],
                                                                                st.session_state['init_data']['n_images'],
                                                                                st.session_state['init_data']['current_image_files'],
                                                                                st.session_state['init_data']['current_image_names'])
                                                                          
        if len(st.session_state['init_data']['current_images_discarted'])>1:
            st.session_state['init_data']['award']=st.session_state['init_data']['award']-len(st.session_state['init_data']['current_images_discarted'])
        
        ## penalty when "select winner" option and few images remain
        if st.session_state['init_data']['token_type']==-3:   
            st.session_state['init_data']['award']=st.session_state['init_data']['award']-1
            if previous_images_discarted<np.round(st.session_state['init_data']['n_images']*0.8):
                st.session_state['init_data']['award']=st.session_state['init_data']['award']+previous_images_discarted-np.round(st.session_state['init_data']['n_images']*0.8)

        ## penalty when no image is discarted
        if previous_images_discarted==len(st.session_state['init_data']['current_images_discarted']):   
            st.session_state['init_data']['award']=st.session_state['init_data']['award']-2


    ## SHOW FINAL RESULTS
    if st.session_state['init_data']['finished_game']==99:
        if st.session_state['init_data']['award']==1 or st.session_state['init_data']['award']==-1:
            st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>¡¡¡ FINISHED WITH</h1><h1 style='text-align:left; float:left; color:green; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>"+str(st.session_state['init_data']['award'])+"</h1><h1 style='text-align:left; float:left; color:black; margin:0px;'>POINT !!!</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>¡¡¡ FINISHED WITH</h1><h1 style='text-align:left; float:left; color:green; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>"+str(st.session_state['init_data']['award'])+"</h1><h1 style='text-align:left; float:left; color:black; margin:0px;'>POINTS !!!</h1>", unsafe_allow_html=True)
        st.session_state['init_data'] = load_data()            
        
    ## Finish game button
    if np.sum(st.session_state['init_data']['current_images_discarted']==0)==1:
        st.session_state['init_data']['finished_game']=99
        st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>You found the Winner picture:</h1><h1 style='text-align:left; float:left; color:green; margin:0px;'>"+st.session_state['init_data']['current_image_names'][st.session_state['init_data']['current_winner_index']]+"</h1>", unsafe_allow_html=True)
        Finsih_Game = st.button('FINISH GAME', key='Finsih_Game')

    ## SHOW EXTRA INFO
    Show_Info(st.session_state['init_data']['feature_questions'])
 
    ## SHOW CURRENT
    st.image(Remaining_Images, use_column_width=False, caption=Image_Names)
    
    ## CLEAR RESOURCES
    gc.collect()
