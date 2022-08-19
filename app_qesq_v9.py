# -*- coding: utf-8 -*-

## Used Imports
import os
import io
import zipfile
import random
import numpy as np
import streamlit as st
import clip
import gc

# import psutil  ## show info (cpu, memeory)

from io import BytesIO
from PIL import Image
from zipfile import ZipFile 
from pathlib import Path, PurePath, PureWindowsPath
# from streamlit import caching

## --------------- USED FUNCTIONS ---------------

def Predict_1_vs_0():
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
        if st.session_state['init_data']['image_current_probs'][i,1]>st.session_state['init_data']['image_current_probs'][i,0]:
            st.session_state['init_data']['image_current_predictions'].append(1)
        else:
            st.session_state['init_data']['image_current_predictions'].append(0)
    
    st.session_state['init_data']['image_current_predictions']=np.array(st.session_state['init_data']['image_current_predictions'])
    
def Predict_0_vs_1():
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
        if st.session_state['init_data']['image_current_probs'][i,0]>st.session_state['init_data']['image_current_probs'][i,1]:
            st.session_state['init_data']['image_current_predictions'].append(1)
        else:
            st.session_state['init_data']['image_current_predictions'].append(0)

    st.session_state['init_data']['image_current_predictions']=np.array(st.session_state['init_data']['image_current_predictions'])

def Predict_0_vs_all():
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
        if np.argmax(st.session_state['init_data']['image_current_probs'][i,:])==0:
            st.session_state['init_data']['image_current_predictions'].append(1)        
        else:
            st.session_state['init_data']['image_current_predictions'].append(0)

    st.session_state['init_data']['image_current_predictions']=np.array(st.session_state['init_data']['image_current_predictions'])
  
def Predict_all_vs_last():
    n_max=len(st.session_state['init_data']['image_current_probs'][:,0])-1
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
        if np.argmax(st.session_state['init_data']['image_current_probs'][i,:])==n_max:
            st.session_state['init_data']['image_current_predictions'].append(1)        
        else:
            st.session_state['init_data']['image_current_predictions'].append(0)

    st.session_state['init_data']['image_current_predictions']=np.array(st.session_state['init_data']['image_current_predictions'])   
def Predict_bald():
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
    
        if st.session_state['init_data']['image_current_probs'][i,0]>st.session_state['init_data']['image_current_probs'][i,1]:
            if st.session_state['init_data']['image_current_probs'][i,2]>st.session_state['init_data']['image_current_probs'][i,3]:
                st.session_state['init_data']['image_current_predictions'].append(1)
            else:
                st.session_state['init_data']['image_current_predictions'].append(0)
        else:
            if st.session_state['init_data']['image_current_probs'][i,4]>st.session_state['init_data']['image_current_probs'][i,5]:
                st.session_state['init_data']['image_current_predictions'].append(1)
            else:
                st.session_state['init_data']['image_current_predictions'].append(0)    

    st.session_state['init_data']['image_current_predictions']=np.array(st.session_state['init_data']['image_current_predictions'])

def Final_Results(N_img, Current_award, Player_indicator, Win_index, Current_images, Img_discarded):
    ## --------------- APPLY DISCARDING ---------------
    if st.session_state['init_data']['show_results']:        
        st.session_state['init_data']['previous_discarding_images_number']=N_img
        Image_discarding()
                   
        ## penalty - game not finished                                                       
        if N_img>1:
            Current_award=Current_award-N_img
        
        ## penalty - "select winner" option used
        if st.session_state['init_data']['token_type']==-3:   
            Current_award=Current_award-1-(st.session_state['init_data']['N_images']-st.session_state['init_data']['previous_discarding_images_number'])

        ## penalty - no image is discarted
        if st.session_state['init_data']['previous_discarding_images_number']==N_img:   
            Current_award=Current_award-5


    ## --------------- SHOW FINAL RESULTS ---------------
    if not st.session_state['init_data']['finished_game']:
        if np.sum(Img_discarded==0)==1:
            st.session_state['init_data']['finished_game']=True
            st.session_state['init_data']['change_player']=False
            st.markdown("<h1 style='text-align:left; color:gray; margin:0px;'>CONGRATULATIONS <span style='text-align:left; color:green; margin:0px;'>"+Player_indicator+"<span style='text-align:left; color:gray; margin:0px;'>! THE WINNER PICTURE IS: <span style='text-align:left; color:green; margin:0px;'>"+Current_images[Win_index]+"</h1>", unsafe_allow_html=True)

            Finsih_Game = st.button('FINISH GAME', key='Finsih_Game')
    return Current_award

def Ask_Question(Player_indicator, Win_index, Current_award):

        ## Finished Game:
    if st.session_state['init_data']['finished_game']:
        st.session_state['init_data']['reload_game']=True
        Restart_App = st.button('GO TO OPTIONS SELECTION TO START NEW GAME', key='Restart_App')
        if Current_award==1 or Current_award==-1:
            st.markdown("<h1 style='text-align:left; color:black; margin:0px;'>¡¡¡ <span style='text-align:left; color:green; margin:0px;'>"+Player_indicator+"<span style='text-align:left; color:black; margin:0px;'>YOU WIN WITH <span style='text-align:left; color:green; margin:0px;'>"+str(Current_award)+"<span style='text-align:left; color:black; margin:0px;'> POINT !!!</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align:left; color:black; margin:0px;'>¡¡¡ <span style='text-align:left; color:green; margin:0px;'>"+Player_indicator+"<span style='text-align:left; color:black; margin:0px;'>YOU WIN WITH <span style='text-align:left; color:green; margin:0px;'>"+str(Current_award)+"<span style='text-align:left; color:black; margin:0px;'> POINTS !!!</h1>", unsafe_allow_html=True)
    else:
    
        st.markdown("<h2 style='text-align:left; float:left; color:gray; margin:0px;'>"+Player_indicator+"Select a type of Query to play.</h2>", unsafe_allow_html=True)

        ## SelectBox - Select query type (game mode)
        Selected_Feature=st.selectbox('Ask a question from a list, create your query or select a winner:', ['Ask a Question', 'Create your own query', 'Create your own 2 querys','Select a Winner'], 
                                               index=0, key='selected_feature', help=None)
            
        ## --------------- SHOW ELEMENTS - QUESTIONS MODE ---------------
        if Selected_Feature=='Ask a Question':
            ## Game mode id
            st.session_state['init_data']['token_type']=0

            ## Text - Questions mode
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"+Player_indicator+"Select a Question from the list.</h3>", unsafe_allow_html=True)
            
            ## SelectBox - Select question
            Selected_Question=st.selectbox('Suggested questions:', st.session_state['init_data']['feature_questions'], 
                                               index=11, key='Selected_Question', help=None)
            st.session_state['init_data']['selected_question']=Selected_Question  # Save Info
            
            ## Current question index
            if Selected_Question not in st.session_state['init_data']['feature_questions']:
                Selected_Question=st.session_state['init_data']['feature_questions'][0]
            
            st.session_state['init_data']['questions_index']=st.session_state['init_data']['feature_questions'].index(Selected_Question)
               
            ## Text - Show current question
            st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+Player_indicator+"Current Question: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"
                        +Selected_Question+"</h3>", unsafe_allow_html=True)
            
            ## Button - Use current question
            Check_Question = st.button('USE THIS QUESTION', key='Check_Question')
            
            ## Check current question
            if st.session_state['init_data']['show_results']:
                st.session_state['init_data']['show_results']=False
                
            else:
                if Check_Question:
                    if Selected_Question=='Are you bald?':
                        if st.session_state['init_data']['Selected_Images_Source']=='Use Original "Guess Who" game images':
                            st.session_state['init_data']['current_querys']=["An illustration of a male person's face","An illustration of a female person's face",
                                                                        "An illustration of a bald man's face","An illustration of a haired person's face", 
                                                                        "An illustration of a bald person's face","An illustration of a person's face"]
                        
                        else:
                            st.session_state['init_data']['current_querys']=['A picture of a male person','A picture of a female person',
                                                                        'A picture of a bald man','A picture of a haired man', 
                                                                        'A picture of a bald person','A picture of a person']
                        st.session_state['init_data']['function_predict']=Predict_bald
                        
                    elif Selected_Question=='Do you have BLACK HAIR?':
                        if st.session_state['init_data']['Selected_Images_Source']=='Use Original "Guess Who" game images':
                            st.session_state['init_data']['current_querys']=["An illustration of a black haired person's face",
                                                                        "An illustration of a chocolate brown haired person's face",
                                                                        "An illustration of a neon tangerine haired person's face",
                                                                        "An illustration of a luminous golden blond haired person's face",
                                                                        "An illustration of a milky white haired person's face"]
                        else:
                            st.session_state['init_data']['current_querys']=["A picture of a black haired person",
                                                                        "A picture of a tawny haired person",
                                                                        "A picture of a blond haired person",
                                                                        "A picture of a gray haired person",
                                                                        "A picture of a red haired person",
                                                                        "A picture of a green haired person",
                                                                        "A picture of a blue haired person",
                                                                        "A picture of a bald-head person"]
                        st.session_state['init_data']['function_predict']=Predict_0_vs_all
                            
                    elif Selected_Question=='Do you have BROWN HAIR?':
                        if st.session_state['init_data']['Selected_Images_Source']=='Use Original "Guess Who" game images':
                            st.session_state['init_data']['current_querys']=["An illustration of a chocolate brown haired person's face",
                                                                        "An illustration of a black haired person's face",
                                                                        "An illustration of a neon tangerine haired person's face",
                                                                        "An illustration of a luminous golden blond haired person's face",
                                                                        "An illustration of a milky white haired person's face"]
                        else:
                            st.session_state['init_data']['current_querys']=["A picture of a tawny haired person",
                                                                        "A picture of a black haired person",
                                                                        "A picture of a blond haired person",
                                                                        "A picture of a gray haired person",
                                                                        "A picture of a red haired person",
                                                                        "A picture of a green haired person",
                                                                        "A picture of a blue haired person",
                                                                        "A picture of a bald-head person"]
                        st.session_state['init_data']['function_predict']=Predict_0_vs_all                            
                                                 
                    elif Selected_Question=='Do you have ORANGE HAIR?':
                        st.session_state['init_data']['current_querys']=["An illustration of a neon tangerine haired person's face",
                                                                        "An illustration of a chocolate brown haired person's face",
                                                                        "An illustration of a black haired person's face",
                                                                        "An illustration of a luminous golden blond haired person's face",
                                                                        "An illustration of a milky white haired person's face"]
                        st.session_state['init_data']['function_predict']=Predict_0_vs_all  
                                                 
                    elif Selected_Question=='Do you have YELLOW HAIR?':
                        st.session_state['init_data']['current_querys']=["An illustration of a luminous golden blond haired person's face",
                                                                        "An illustration of a chocolate brown haired person's face",
                                                                        "An illustration of a neon tangerine haired person's face",
                                                                        "An illustration of a black haired person's face",
                                                                        "An illustration of a milky white haired person's face"]
                        st.session_state['init_data']['function_predict']=Predict_0_vs_all 
                                                 
                    elif Selected_Question=='Do you have WHITE HAIR?':
                        st.session_state['init_data']['current_querys']=["An illustration of a milky white haired person's face",
                                                                        "An illustration of a chocolate brown haired person's face",
                                                                        "An illustration of a neon tangerine haired person's face",
                                                                        "An illustration of a luminous golden blond haired person's face",
                                                                        "An illustration of a black haired person's face"]
                        st.session_state['init_data']['function_predict']=Predict_0_vs_all                         

                    elif Selected_Question=='Do you have BLOND HAIR?':
                        st.session_state['init_data']['current_querys']=['A picture of a blond haired person',
                                                                    'A picture of a tawny haired person',
                                                                    'A picture of a black haired person',
                                                                    'A picture of a gray haired person',
                                                                    'A picture of a red haired person',
                                                                    'A picture of a green haired person',
                                                                    'A picture of a blue haired person',
                                                                    'A picture of a bald-head person']
                        st.session_state['init_data']['function_predict']=Predict_0_vs_all
                        
                    elif Selected_Question=='Do you have RED HAIR?':
                        st.session_state['init_data']['current_querys']=['A picture of a red haired person',
                                                                    'A picture of a tawny haired person',
                                                                    'A picture of a blond haired person',
                                                                    'A picture of a gray haired person',
                                                                    'A picture of a black haired person',
                                                                    'A picture of a green haired person',
                                                                    'A picture of a blue haired person',
                                                                    'A picture of a bald-head person']
                        st.session_state['init_data']['function_predict']=Predict_0_vs_all
                        
                    elif Selected_Question=='Do you have GRAY HAIR?':
                        st.session_state['init_data']['current_querys']=['A picture of a gray haired person',
                                                                    'A picture of a tawny haired person',
                                                                    'A picture of a blond haired person',
                                                                    'A picture of a black haired person',
                                                                    'A picture of a red haired person',
                                                                    'A picture of a green haired person',
                                                                    'A picture of a blue haired person',
                                                                    'A picture of a bald-head person']
                        st.session_state['init_data']['function_predict']=Predict_0_vs_all
                        
                    elif Selected_Question=='Do you wear EYEGLASSES?':
                        if st.session_state['init_data']['Selected_Images_Source']=='Use Original "Guess Who" game images':
                            st.session_state['init_data']['current_querys']=["An illustration of a person's face with eyeglasses",
                                                                        "An illustration of a person's face with glasses",
                                                                        "An illustration of a person's face with sunglasses",
                                                                        "An illustration of a person's face"]
                        else:
                            st.session_state['init_data']['current_querys']=['A picture of a person with eyeglasses',
                                                                        'A picture of a person with glasses',
                                                                        'A picture of a person with sunglasses',
                                                                        'A picture of a person']
                        st.session_state['init_data']['function_predict']=Predict_all_vs_last    
                   
                    else:
                        st.session_state['init_data']['current_querys']=[st.session_state['init_data']['querys_list_yes'][st.session_state['init_data']['questions_index']],
                                                                        st.session_state['init_data']['querys_list_no'][st.session_state['init_data']['questions_index']]]
                        st.session_state['init_data']['function_predict']=Predict_0_vs_1
                
                    CLIP_Process()
                    st.session_state['init_data']['function_predict']()
                    st.session_state['init_data']['show_results']=True
                    
                    
        ## --------------- SHOW ELEMENTS - 1 QUERY MOD ---------------
        if Selected_Feature=='Create your own query':              
            
            ## Game mode id
            st.session_state['init_data']['token_type']=-1

            ## Text - Query mode
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"+Player_indicator+"Write your own query and press the button.</h3>", unsafe_allow_html=True)
            
            ## TextInput - Select query
            User_Input = st.text_input('It is recommended to use a text like: "A picture of a ... person" or "A picture of a person ..." (CLIP will check -> "Your query"  vs  "A picture of a person" )', 'A picture of a person', key='User_Input', help=None)
            st.session_state['init_data']['user_input']=User_Input  # Save Info

            ## Text - Show current query
            st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+Player_indicator+"Current Query: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+User_Input+"</h3>", unsafe_allow_html=True)
            
            ## Button - Use current query
            Check_Query = st.button('USE MY OWN QUERY', key='Check_Query')
            
            ## Check current question            
            if st.session_state['init_data']['show_results']:
                st.session_state['init_data']['show_results']=False
            else:
                if Check_Query:
                    if User_Input!='A picture of a person':
                        st.session_state['init_data']['current_querys']=['A Picture of a person',User_Input]
                        st.session_state['init_data']['function_predict']=Predict_1_vs_0
                        CLIP_Process()
                        st.session_state['init_data']['function_predict']()
                        st.session_state['init_data']['show_results']=True
                        
                    else:
                        st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"+Player_indicator+"Your query must be different of 'A picture of a person'.</h3>", unsafe_allow_html=True)
               
               
        ## --------------- SHOW ELEMENTS - 2 QUERYS MODE ---------------
        if Selected_Feature=='Create your own 2 querys':
            
            ## Game mode id
            st.session_state['init_data']['token_type']=-2

            ## Text - Querys mode
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"+Player_indicator+"Write your own querys by introducing 2 opposite descriptions.</h3>", unsafe_allow_html=True)
            
            ## SelectBox - Select querys
            User_Input_Querys1 = st.text_input('Write your "True" query:', 'A picture of a person',
                                                        key='User_Input_Querys1', help=None)
            User_Input_Querys2 = st.text_input('Write your "False" query:', 'A picture of a person',
                                                        key='User_Input_Querys2', help=None)
            st.session_state['init_data']['user_input_querys1']=User_Input_Querys1  # Save Info
            st.session_state['init_data']['user_input_querys2']=User_Input_Querys2  # Save Info
                             
            ## Text - Show current querys
            st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+Player_indicator+"Current Querys: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+User_Input_Querys1+' vs '+User_Input_Querys2+"</h3>", unsafe_allow_html=True)
            
            ## Button - Use current querys
            Check_Querys = st.button('USE MY OWN QUERYS', key='Check_Querys')
            
            ## Check current querys
            if st.session_state['init_data']['show_results']:
                st.session_state['init_data']['show_results']=False
            else:
                if Check_Querys:
                    if User_Input_Querys1!=User_Input_Querys2:
                        st.session_state['init_data']['current_querys']=[User_Input_Querys1,User_Input_Querys2]     
                        st.session_state['init_data']['function_predict']=Predict_0_vs_1
                        CLIP_Process()
                        st.session_state['init_data']['function_predict']()
                        st.session_state['init_data']['show_results']=True
                        
                    else:
                        st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"+Player_indicator+"Your two own querys must be different.</h3>", unsafe_allow_html=True)


        ## --------------- SHOW ELEMENTS - WINNER MODE ---------------
        if Selected_Feature=='Select a Winner': 
            
            ## Game mode id
            st.session_state['init_data']['token_type']=-3

            ## Text - Winner mode
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"+Player_indicator+"Select a Winner picture name.</h3>", 
                        unsafe_allow_html=True)
            
            if st.session_state['init_data']['player2_turn']:
                st.session_state['init_data']['selected_winner2']=st.selectbox('If you are inspired, Select a Winner image directly:', st.session_state['init_data']['winner_options'],
                                            index=0, key='Selected_Winner', help=None)
                ## Text - Show current winner
                st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+Player_indicator+"Current Winner: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+st.session_state['init_data']['selected_winner2']+"</h3>", unsafe_allow_html=True)
            else:
                st.session_state['init_data']['selected_winner']=st.selectbox('If you are inspired, Select a Winner image directly:', st.session_state['init_data']['winner_options'],
                                            index=0, key='Selected_Winner', help=None)
                ## Text - Show current winner
                st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+Player_indicator+"Current Winner: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+st.session_state['init_data']['selected_winner']+"</h3>", unsafe_allow_html=True)
            
            ## Button - Use current winner
            Check_Winner = st.button('CHECK THIS WINNER', key='Check_Winner')
            st.session_state['init_data']['button_winner']=Check_Winner  # Save Info
                                                
            ## Check current winner
            if st.session_state['init_data']['show_results']:
                st.session_state['init_data']['show_results']=False
            else:
                if Check_Winner:
                    if st.session_state['init_data']['player2_turn']:
                        if st.session_state['init_data']['selected_winner2'] in st.session_state['init_data']['current_image_names2']:
                            st.session_state['init_data']['selected_winner_index2']=np.where(st.session_state['init_data']['selected_winner2']==st.session_state['init_data']['current_image_names2'])[0]
                            st.session_state['init_data']['image_current_predictions']=np.zeros(st.session_state['init_data']['n_images2'])
                            st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['selected_winner_index2']]=1    
                            st.session_state['init_data']['show_results']=True
                        else:
                            st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"+Player_indicator+"Your must select a not discarded picture.</h3>", unsafe_allow_html=True)
                    else:
                        if st.session_state['init_data']['selected_winner'] in st.session_state['init_data']['current_image_names']:
                            st.session_state['init_data']['selected_winner_index']=np.where(st.session_state['init_data']['selected_winner']==st.session_state['init_data']['current_image_names'])[0]
                            st.session_state['init_data']['image_current_predictions']=np.zeros(st.session_state['init_data']['n_images'])
                            st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['selected_winner_index']]=1    
                            st.session_state['init_data']['show_results']=True
                        else:
                            st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"+Player_indicator+"Your must select a not discarded picture.</h3>", unsafe_allow_html=True)


        ## --------------- ACTIONS SHOWING RESULTS ---------------
        if st.session_state['init_data']['show_results']:

            ## Show current results
            if st.session_state['init_data']['token_type']==0:
                if st.session_state['init_data']['image_current_predictions'][Win_index]:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['selected_question']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>YES</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['selected_question']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>NO</h3>", unsafe_allow_html=True)
                    
            if st.session_state['init_data']['token_type']==-1:
                if st.session_state['init_data']['image_current_predictions'][Win_index]:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['user_input']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>TRUE</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['user_input']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>FALSE</h3>", unsafe_allow_html=True)
                    
            if st.session_state['init_data']['token_type']==-2:
                if st.session_state['init_data']['image_current_predictions'][Win_index]:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>The most accurate query is:</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>"+st.session_state['init_data']['user_input_querys1']+"</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>The most accurate query is:</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>"+st.session_state['init_data']['user_input_querys2']+"</h3>", unsafe_allow_html=True)
              
            if st.session_state['init_data']['player2_turn']:
                if st.session_state['init_data']['token_type']==-3:
                    if not st.session_state['init_data']['selected_winner2']==st.session_state['init_data']['current_image_names'][Win_index]:
                        st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>The winner picture is not:</h3><h3 style='text-align:left; float:center; color:red; margin:0px;'>"+st.session_state['init_data']['selected_winner2']+"</h3>", unsafe_allow_html=True)
            else:
                if st.session_state['init_data']['token_type']==-3:
                    if not st.session_state['init_data']['selected_winner']==st.session_state['init_data']['current_image_names'][Win_index]:
                        st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>The winner picture is not:</h3><h3 style='text-align:left; float:center; color:red; margin:0px;'>"+st.session_state['init_data']['selected_winner']+"</h3>", unsafe_allow_html=True)

def CLIP_Process():
    ## Tokenization process
    clip_model, clip_transform=CLIP_Loading()
    clip_text = clip.tokenize(st.session_state['init_data']['current_querys']).to("cpu")
    n_tokens=len(st.session_state['init_data']['current_querys'])
    
    ## Image Process
    if st.session_state['init_data']['player2_turn']:
        st.session_state['init_data']['image_current_probs']=np.zeros((st.session_state['init_data']['n_images2'],n_tokens))
        for i in range(st.session_state['init_data']['n_images2']):
            current_image_file = Load_Image(i,st.session_state['init_data']['image_current_paths2'])
            img_preprocessed = clip_transform(Image.fromarray(current_image_file)).unsqueeze(0).to("cpu")
            img_logits, img_logits_txt = clip_model(img_preprocessed, clip_text)
            st.session_state['init_data']['image_current_probs'][i,:]=np.round(img_logits.detach().numpy()[0],2)
    else:
        st.session_state['init_data']['image_current_probs']=np.zeros((st.session_state['init_data']['n_images'],n_tokens))
        for i in range(st.session_state['init_data']['n_images']):
            current_image_file = Load_Image(i,st.session_state['init_data']['image_current_paths'])
            img_preprocessed = clip_transform(Image.fromarray(current_image_file)).unsqueeze(0).to("cpu")
            img_logits, img_logits_txt = clip_model(img_preprocessed, clip_text)
            st.session_state['init_data']['image_current_probs'][i,:]=np.round(img_logits.detach().numpy()[0],2)
        
    del i,n_tokens,clip_model,clip_transform,clip_text,current_image_file,img_preprocessed,img_logits,img_logits_txt
    gc.collect()
       
def Image_discarding():
    if st.session_state['init_data']['player2_turn']:
        for i in range(len(st.session_state['init_data']['current_images_discarted2'])):
            if st.session_state['init_data']['current_images_discarted2'][i]==0 and st.session_state['init_data']['image_current_predictions'][i]!=st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index2']]:
                st.session_state['init_data']['current_images_discarted2'][i]=1

        previous_names=st.session_state['init_data']['current_image_names2']
        st.session_state['init_data']['current_image_names2']=[]
        previous_files=st.session_state['init_data']['image_current_paths2']     
        st.session_state['init_data']['image_current_paths2']=[] 
        previous_predictions=st.session_state['init_data']['image_current_predictions'] 
        st.session_state['init_data']['image_current_predictions']=[]
        current_index=0
        new_index=0
        for i in range(st.session_state['init_data']['n_images2']):
            if st.session_state['init_data']['current_images_discarted2'][current_index]==0:
                st.session_state['init_data']['image_current_paths2'].append(previous_files[current_index])
                st.session_state['init_data']['current_image_names2'].append(previous_names[current_index])
                st.session_state['init_data']['image_current_predictions'].append(previous_predictions[current_index])
                if current_index==st.session_state['init_data']['current_winner_index2']:
                    st.session_state['init_data']['current_winner_index2']=new_index
                    
                new_index+=1
                
            current_index+=1
                
        st.session_state['init_data']['n_images2']=np.sum(st.session_state['init_data']['current_images_discarted2']==0)                     
        st.session_state['init_data']['current_image_names2']=np.array(st.session_state['init_data']['current_image_names2'])                   
        st.session_state['init_data']['image_current_paths2']=np.array(st.session_state['init_data']['image_current_paths2']) 
        st.session_state['init_data']['current_images_discarted2']=np.zeros(st.session_state['init_data']['n_images2'])
    else:
        for i in range(len(st.session_state['init_data']['current_images_discarted'])):
            if st.session_state['init_data']['current_images_discarted'][i]==0 and st.session_state['init_data']['image_current_predictions'][i]!=st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index']]:
                st.session_state['init_data']['current_images_discarted'][i]=1

        previous_names=st.session_state['init_data']['current_image_names']
        st.session_state['init_data']['current_image_names']=[]
        previous_files=st.session_state['init_data']['image_current_paths']     
        st.session_state['init_data']['image_current_paths']=[] 
        previous_predictions=st.session_state['init_data']['image_current_predictions'] 
        st.session_state['init_data']['image_current_predictions']=[]
        current_index=0
        new_index=0
        for i in range(st.session_state['init_data']['n_images']):
            if st.session_state['init_data']['current_images_discarted'][current_index]==0:
                st.session_state['init_data']['image_current_paths'].append(previous_files[current_index])
                st.session_state['init_data']['current_image_names'].append(previous_names[current_index])
                st.session_state['init_data']['image_current_predictions'].append(previous_predictions[current_index])
                if current_index==st.session_state['init_data']['current_winner_index']:
                    st.session_state['init_data']['current_winner_index']=new_index
                    
                new_index+=1
                
            current_index+=1
                
        st.session_state['init_data']['n_images']=np.sum(st.session_state['init_data']['current_images_discarted']==0)                     
        st.session_state['init_data']['current_image_names']=np.array(st.session_state['init_data']['current_image_names'])                   
        st.session_state['init_data']['image_current_paths']=np.array(st.session_state['init_data']['image_current_paths']) 
        st.session_state['init_data']['current_images_discarted']=np.zeros(st.session_state['init_data']['n_images'])
        
    del previous_names,previous_files,previous_predictions,current_index,new_index,i
      
def Show_images():
    showed_images=[] 
    if st.session_state['init_data']['player2_turn']:
        n_img=st.session_state['init_data']['n_images2']
        winner_index=st.session_state['init_data']['current_winner_index2']
        Current_path=st.session_state['init_data']['image_current_paths2']

    else:
        n_img=st.session_state['init_data']['n_images']
        winner_index=st.session_state['init_data']['current_winner_index']
        Current_path=st.session_state['init_data']['image_current_paths']
                
    for current_index in range(n_img):
        if st.session_state['init_data']['show_results']:
            current_line_width=4
            if st.session_state['init_data']['image_current_predictions'][current_index]==st.session_state['init_data']['image_current_predictions'][winner_index]:
                current_color=np.array([0,255,0])
            else:
                current_color=np.array([255,0,0]) 
        else:
            current_line_width=2
            current_color=np.zeros(3)  
        image_size=240
        current_image_file=Load_Image(current_index,Current_path)        
        w,h,c = np.shape(current_image_file)
            
        image_highlighted=np.zeros([h+current_line_width*2,image_size,c])+255
        image_highlighted[current_line_width:w+current_line_width,current_line_width:w+current_line_width,:]=current_image_file
        image_highlighted[:current_line_width,:w+2*current_line_width,:]=current_color
        image_highlighted[w+current_line_width:,:w+2*current_line_width,:]=current_color
        image_highlighted[:,w+current_line_width:w+2*current_line_width,:]=current_color
        image_highlighted[:,:current_line_width,:]=current_color
        showed_images.append(image_highlighted)
        
    ## result to array      
    showed_images=np.array(showed_images)/255
    del image_highlighted,current_index,current_line_width,current_color,image_size,current_image_file,w,h,c
    
    if st.session_state['init_data']['player2_turn']:
        return [showed_images, st.session_state['init_data']['current_image_names2']]
    else:
        return [showed_images, st.session_state['init_data']['current_image_names']]
    
def find_same_name(index,names_list):
    if st.session_state['init_data']['special_images_names']:
        fixed_name=names_list[index][:names_list[index].find('-')]
        index_list=[]
        for i in range(0,len(names_list)):
            if fixed_name==names_list[i][:names_list[i].find('-')]:
                index_list.append(i)
    else:
        if '.' in names_list[index]:
            fixed_name=names_list[index][:names_list[index].find('.')]
            index_list=[]
            for i in range(0,len(names_list)):
                if fixed_name==names_list[i][:names_list[i].find('.')]:
                    index_list.append(i)
        else:
            fixed_name=names_list[index]
            index_list=[]
            for i in range(0,len(names_list)):
                name_find=names_list[i]
                if fixed_name==names_list[i]:
                    index_list.append(i)
    return index_list

def find_list_elements(x,x_list):
    if type(x)==list:
        for x_element in x:
            x_list=find_list_elements(x_element,x_list)
        
    elif type(x)==str:
        if x[-4:]=='.jpg' or x[-4:]== '.png':
            x_list.append(x)
    return x_list
    
def Select_Images_Randomly():
    st.session_state['init_data']['image_current_paths']=[]
    st.session_state['init_data']['current_image_names']=[]
    image_index=[]
    image_delete=[]
    
    if st.session_state['init_data']['zip_file']=='frifam.zip':
        st.session_state['init_data']['special_images_names']=True
    else:
        st.session_state['init_data']['special_images_names']=False
    
    archive = zipfile.ZipFile(st.session_state['init_data']['zip_file'], 'r')
    listOfFileNames = archive.namelist()     
    image_index_all=list(range(len(listOfFileNames)))
    
    image_index.append(random.choice(image_index_all))
    
    image_delete=find_same_name(image_index[0],listOfFileNames)
    
    for i in image_delete:
        image_index_all.remove(i)  
        
    current_index=1         
    while len(image_index)<st.session_state['init_data']['N_images']:  
        image_index.append(random.choice(image_index_all))  
        image_delete=find_same_name(image_index[current_index],listOfFileNames)  
        for i in image_delete:
            image_index_all.remove(i)

        current_index+=1
        
   # Iterate over the file names
    for current_index in image_index:
        image_current_path=listOfFileNames[current_index]
        st.session_state['init_data']['image_current_paths'].append(image_current_path)
        current_name = os.path.basename(image_current_path)
        st.session_state['init_data']['special_images_names']=False
        if '-' in current_name:
            current_name = current_name[:current_name.find('-')-1]
            st.session_state['init_data']['special_images_names']=True
        elif '.' in current_name:
            current_name = current_name[:current_name.find('.')]
            st.session_state['init_data']['special_images_names']=False
        
        st.session_state['init_data']['current_image_names'].append(current_name)
                
    st.session_state['init_data']['current_image_names']=np.array(st.session_state['init_data']['current_image_names'])
    st.session_state['init_data']['current_image_names2']=st.session_state['init_data']['current_image_names']
    st.session_state['init_data']['image_current_paths']=np.array(st.session_state['init_data']['image_current_paths'])
    st.session_state['init_data']['image_current_paths2']=st.session_state['init_data']['image_current_paths']
    st.session_state['init_data']['winner_options']=st.session_state['init_data']['current_image_names']
    st.session_state['init_data']['images_not_selected']=False
    del image_index,archive,listOfFileNames,image_index_all,current_index,image_current_path

def Load_Image(current_index, current_path):
    archive = zipfile.ZipFile(st.session_state['init_data']['zip_file'], 'r')
    image_current_path=current_path[current_index]
    image_file=Image.open(BytesIO(archive.read(image_current_path)))
    image_file = image_file.convert('RGB')  
    
    if not (image_file.size[0] == 224 and image_file.size[1] == 224): 
        image_file=image_file.resize((224, 224))
    del image_current_path,archive
    return np.array(image_file)


def Select_Dataset():   
    if st.session_state['init_data']['Selected_Images_Source']=='Use Original "Guess Who" game images':
        ## Select zip file
        st.session_state['init_data']['zip_file']='Original.zip'

        st.session_state['init_data']['feature_questions']=["Are you a MAN?", "Are you a WOMAN?", "Are you an ATTRACTIVE person?", "Are you an CHUBBY person?", "Are you YOUNG?",
                    "Are you a person with RECEDING HAIRLINES?", "Are you SMILING?","Are you BALD?", 
                    "Do you have BLACK HAIR?", "Do you have BROWN HAIR?", "Do you have ORANGE HAIR?", "Do you have YELLOW HAIR?",
                    "Do you have WHITE HAIR?", "Do you have STRAIGHT HAIR?", "Do you have WAVY HAIR?",
                    "Do you have a BEARD?", "Do you have a MUSTACHE?", "Do you have SIDEBURNS?",
                    "Do you have a GOATEE?", "Do you wear HEAVY MAKEUP?", "Do you wear EYEGLASSES?",
                    "Do you have BUSHY EYEBROWS?", "Do you have a DOUBLE CHIN?", 
                    "Do you have a high CHEECKBONES?", "Do you have SLIGHTLY OPEN MOUTH?", 
                    "Do you have NARROWED EYES?", "Do you have an OVAL FACE?", 
                    "Do you have PALE SKIN?", "Do you have a POINTY NOSE?", "Do you have ROSY CHEEKS?", 
                    "Do you have FIVE O'CLOCK SHADOW?", "Do you have ARCHED EYEBROWS?", "Do you have BUGS UNDER your EYES?", 
                    "Do you have BANGS?", "Do you have a BIG LIPS?", "Do you have a BIG NOSE?",
                    "Are you wearing EARRINGS?", "Are you wearing a HAT?", 
                    "Are you wearing LIPSTICK?", "Are you wearing NECKLACE?", 
                    "Are you wearing NECKTIE?"]
    
        st.session_state['init_data']['querys_list_yes']=["A picture of a male person", "A picture of a female person", "A picture of an attractive person", "A picture of a fat person", "A picture of a young person", 
            "A picture of a receding-hairline person  ", "A picture of a smily person", "A picture of a bald person",
            "An illustration of a person's face with black hair", "An illustration of a person's face with brown hair", "An illustration of a person's face with orange hair", "An illustration of a person's face with yellow hair", 
            "An illustration of a person's face with white hair", "An illustration of a person's face with straight hair", "An illustration of a person's face with wavy hair", 
            "A picture of a unshaved person", "A picture of a mustachioed person", "An illustration of a person's face with bushy sideburns", 
            "An illustration of a person's face with goatee", "An illustration of a person's face with heavy makeup", "An illustration of a person's face with eyeglasses",             
            "An illustration of a person's face with bushy eyebrows", "A picture of a double chin person", 
            "An illustration of a person's face with high cheekbones", "An illustration of a person's face with opened mouth", 
            "An illustration of a person's face with narrow eyes", "An illustration of a person's face with an oval-shaped face", 
            "An illustration of a person's face wiht pale skin", "A picture of a pointy-nosed person ", "An illustration of a person's face with colored cheeks", 
            "A picture of a five o'clock shadow person", "A picture of a rounded eyebrows person", "An illustration of a person's face with bags under the eyes", 
            "An illustration of a person's face with bangs", "A picture of a wide-liped person", "A picture of a big-nosed person",            
            "An illustration of a person's face with earrings", "An illustration of a person's face with hat", 
            "An illustration of a person's face with lipstick", "A picture of a necklaced person", 
            "A picture of a necktied person"]
    
        st.session_state['init_data']['querys_list_no']=["A picture of a female person", "A picture of a male person", "A picture of an ugly person", "A picture of a slender person", "A picture of an aged person", 
            "A picture of a hairy person", "An illustration of a person's face", "A picture of a hairy person",
            "An illustration of a person's face", "An illustration of a person's face", "An illustration of a person's face", "An illustration of a person's face", 
            "An illustration of a person's face", "An illustration of a person's face with wavy hair", "An illustration of a person's face with straight hair", 
            "A picture of a glabrous person", "An illustration of a person's face", "An illustration of a person's face with shaved sideburns", 
            "An illustration of a person's face", "An illustration of a person's face with light makeup", "An illustration of a person's face ",             
            "An illustration of a person's face with sparse eyebrows", "An illustration of a person's face with a double chin", 
            "An illustration of a person's face with low cheekbones", "An illustration of a person's face with closed mouth", 
            "An illustration of a person's face with wide eyes", "An illustration of a person's face with a normal-shaped face", 
            "An illustration of a person's face wiht tanned skin", "A picture of a flat-nosed person", "An illustration of a person's face with pale cheeks", 
            "A picture of a shaved or unshaved person", "An illustration of a person's face a straight eyebrows person", "An illustration of a person's face with with smooth skin under the eyes", 
            "An illustration of a person's face", "A picture of a narrow-liped person", "A picture of a small-nosed person",            
            "An illustration of a person's face", "An illustration of a person's face with hair", 
            "An illustration of a person's face with natural lips", "An illustration of a person's face", 
            "An illustration of a person's face"]    
    
    else:
        ## Select zip file
        if st.session_state['init_data']['Selected_Images_Source']=='Use Celeba dataset':
            st.session_state['init_data']['zip_file']='guess_who_images.zip'
        elif st.session_state['init_data']['Selected_Images_Source']=='Use friends dataset':
            st.session_state['init_data']['zip_file']='frifam.zip'
        else:
            st.session_state['init_data']['zip_file']='Use images from specific path'

    
        st.session_state['init_data']['feature_questions']=["Are you a MAN?", "Are you a WOMAN?", "Are you an ATTRACTIVE person?", "Are you an CHUBBY person?", "Are you YOUNG?",
                    "Are you a person with RECEDING HAIRLINES?", "Are you SMILING?","Are you BALD?", 
                    "Do you have BLACK HAIR?", "Do you have BROWN HAIR?", "Do you have BLOND HAIR?", "Do you have RED HAIR?",
                    "Do you have GRAY HAIR?", "Do you have STRAIGHT HAIR?", "Do you have WAVY HAIR?",
                    "Do you have a BEARD?", "Do you have a MUSTACHE?", "Do you have SIDEBURNS?",
                    "Do you have a GOATEE?", "Do you wear HEAVY MAKEUP?", "Do you wear EYEGLASSES?",
                    "Do you have BUSHY EYEBROWS?", "Do you have a DOUBLE CHIN?", 
                    "Do you have a high CHEECKBONES?", "Do you have SLIGHTLY OPEN MOUTH?", 
                    "Do you have NARROWED EYES?", "Do you have an OVAL FACE?", 
                    "Do you have PALE SKIN?", "Do you have a POINTY NOSE?", "Do you have ROSY CHEEKS?", 
                    "Do you have FIVE O'CLOCK SHADOW?", "Do you have ARCHED EYEBROWS?", "Do you have BUGS UNDER your EYES?", 
                    "Do you have BANGS?", "Do you have a BIG LIPS?", "Do you have a BIG NOSE?",
                    "Are you wearing EARRINGS?", "Are you wearing a HAT?", 
                    "Are you wearing LIPSTICK?", "Are you wearing NECKLACE?", 
                    "Are you wearing NECKTIE?"]
    
        st.session_state['init_data']['querys_list_yes']=["A picture of a male person", "A picture of a female person", "A picture of an attractive person", "A picture of a fat person", "A picture of a young person", 
            "A picture of a receding-hairline person  ", "A picture of a smily person", "A picture of a bald person",
            "A picture of a person with black hair", "A picture of a person with brown hair", "A picture of a person with blond hair", "A picture of a person with red hair", 
            "A picture of a person with gray hair", "A picture of a person with straight hair", "A picture of a person with wavy hair", 
            "A picture of a unshaved person", "A picture of a mustachioed person", "A picture of a person with bushy sideburns", 
            "A picture of a person with goatee", "A picture of a person with heavy makeup", "A picture of a person with eyeglasses",             
            "A picture of a person with bushy eyebrows", "A picture of a double chin person", 
            "A picture of a person with high cheekbones", "A picture of a person with opened mouth", 
            "A picture of a person with narrow eyes", "A picture of a person with an oval-shaped face", 
            "A picture of a person wiht pale skin", "A picture of a pointy-nosed person ", "A picture of a person with colored cheeks", 
            "A picture of a five o'clock shadow person", "A picture of a rounded eyebrows person", "A picture of a person with bags under the eyes", 
            "A picture of a person with bangs", "A picture of a wide-liped person", "A picture of a big-nosed person",            
            "A picture of a person with earrings", "A picture of a person with hat", 
            "A picture of a person with lipstick", "A picture of a necklaced person", 
            "A picture of a necktied person"]
    
        st.session_state['init_data']['querys_list_no']=["A picture of a female person", "A picture of a male person", "A picture of an ugly person", "A picture of a slender person", "A picture of an aged person", 
            "A picture of a hairy person", "A picture of a person", "A picture of a hairy person",
            "A picture of a person", "A picture of a person", "A picture of a person", "A picture of a person", 
            "A picture of a person", "A picture of a person with wavy hair", "A picture of a person with straight hair", 
            "A picture of a glabrous person", "A picture of a person", "A picture of a person with shaved sideburns", 
            "A picture of a person", "A picture of a person with light makeup", "A picture of a person ",             
            "A picture of a person with sparse eyebrows", "A picture of a person with a double chin", 
            "A picture of a person with low cheekbones", "A picture of a person with closed mouth", 
            "A picture of a person with wide eyes", "A picture of a person with a normal-shaped face", 
            "A picture of a person wiht tanned skin", "A picture of a flat-nosed person", "A picture of a person with pale cheeks", 
            "A picture of a shaved or unshaved person", "A picture of a person a straight eyebrows person", "A picture of a person with with smooth skin under the eyes", 
            "A picture of a person", "A picture of a narrow-liped person", "A picture of a small-nosed person",            
            "A picture of a person", "A picture of a person with hair", 
            "A picture of a person with natural lips", "A picture of a person", 
            "A picture of a person"] 
                    

def Show_Info():
    #st.sidebar.markdown("<p></p><hr><h2 style='text-align:left; float:left; color:gray; margin:0px;'>INFO</h2>", unsafe_allow_html=True)
    #st.sidebar.write(st.session_state['init_data'])
    st.sidebar.markdown("<p></p><hr><h2 style='text-align:left; float:left; color:gray; margin:0px;'>List of avalaible Questions</h2>", unsafe_allow_html=True)
    st.sidebar.write(st.session_state['init_data']['feature_questions'])

def Load_Data(N):
    st.session_state['init_data']={
        'status':0,
        'award1':100,
        'award2':100,
        'N_images':N,
        'n_images':N,
        'n_images2':N,
        'N_players':1,
        'Selected_Images_Source':'Use Celeba dataset',
        'zip_file':'guess_who_images.zip',
        'previous_zip_file':'guess_who_images.zip',
        'special_images_names':False,
        'images_not_selected':True,
        'token_type':0,
        'feature_questions':['Are you a MAN?', 'Are you a WOMAN?', 'Are you an ATTRACTIVE person?', 'Are you an CHUBBY person?', 'Are you YOUNG?',
                    'Are you a person with RECEDING HAIRLINES?', 'Are you SMILING?','Are you BALD?', 
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
                    'Are you wearing NECKTIE?'],
        'selected_question':'Are you a MAN?',
        'questions_index':0,
        'show_results':False,
        'current_querys':['A picture of a person','A picture of a person'],
        'function_predict':Predict_0_vs_1,
        'querys_list_yes':['A picture of a male person', 'A picture of a female person', 'A picture of an attractive person', 'A picture of a fat person', 'A picture of a young person', 
            'A picture of a receding-hairline person  ', 'A picture of a smily person', 'A picture of a bald person',
            'A picture of a person with black hair', 'A picture of a person with brown hair', 'A picture of a person with blond hair', 'A picture of a person with red hair', 
            'A picture of a person with gray hair', 'A picture of a person with straight hair', 'A picture of a person with wavy hair', 
            'A picture of a unshaved person', 'A picture of a mustachioed person', 'A picture of a person with bushy sideburns', 
            'A picture of a person with goatee', 'A picture of a person with heavy makeup', 'A picture of a person with eyeglasses',             
            'A picture of a person with bushy eyebrows', 'A picture of a double chin person', 
            'A picture of a person with high cheekbones', 'A picture of a person with opened mouth', 
            'A picture of a person with narrow eyes', 'A picture of a person with an oval-shaped face', 
            'A picture of a person wiht pale skin', 'A picture of a pointy-nosed person ', 'A picture of a person with colored cheeks', 
            "A picture of a five o'clock shadow person", 'A picture of a rounded eyebrows person', 'A picture of a person with bags under the eyes', 
            'A picture of a person with bangs', 'A picture of a wide-liped person', 'A picture of a big-nosed person',            
            'A picture of a person with earrings', 'A picture of a person with hat', 
            'A picture of a person with lipstick', 'A picture of a necklaced person', 
            'A picture of a necktied person'
            ],
        'querys_list_no':['A picture of a female person', 'A picture of a male person', 'A picture of an ugly person', 'A picture of a slender person', 'A picture of an aged person', 
            'A picture of a hairy person', 'A picture of a person', 'A picture of a hairy person',
            'A picture of a person', 'A picture of a person', 'A picture of a person', 'A picture of a person', 
            'A picture of a person', 'A picture of a person with wavy hair', 'A picture of a person with straight hair', 
            'A picture of a glabrous person', 'A picture of a person', 'A picture of a person with shaved sideburns', 
            'A picture of a person', 'A picture of a person with light makeup', 'A picture of a person ',             
            'A picture of a person with sparse eyebrows', 'A picture of a person with a double chin', 
            'A picture of a person with low cheekbones', 'A picture of a person with closed mouth', 
            'A picture of a person with wide eyes', 'A picture of a person with a normal-shaped face', 
            'A picture of a person wiht tanned skin', 'A picture of a flat-nosed person', 'A picture of a person with pale cheeks', 
            "A picture of a shaved or unshaved person", 'A picture of a person a straight eyebrows person', 'A picture of a person with with smooth skin under the eyes', 
            'A picture of a person', 'A picture of a narrow-liped person', 'A picture of a small-nosed person',            
            'A picture of a person', 'A picture of a person with hair', 
            'A picture of a person with natural lips', 'A picture of a person', 
            'A picture of a person'
            ],
        'token_type':0,
        'user_input':'A picture of a person',
        'user_input_querys1':'A picture of a person',
        'user_input_querys2':'A picture of a person',
        'image_current_probs':np.zeros((N,2)),
        'selected_winner':'Winner not selected',
        'selected_winner2':'Winner not selected',
        'reset_app':False,
        'change_player':False,
        'player2_turn':False,
        'finished_game':False,
        'reload_game':False,        
        'random_winner':False,
        'show_images':[],
        'previous_discarding_images_number':0,
        'selected_winner_index':0,
        'selected_winner_index2':0,
        'current_images_discarted':np.zeros((N)),
        'current_images_discarted2':np.zeros((N)),
        'current_winner_index':0,
        'current_winner_index2':0,
        'current_image_names':[],
        'current_image_names2':[],
        'image_current_paths':[],
        'image_current_paths2':[],
        'winner_options':[],
        'image_current_predictions':np.zeros((N))+2}


## --------------- MAIN FUCTION ---------------
def Main_Program():

    ## --------------- LOAD DATA ---------------
    if 'init_data' not in st.session_state:
        Load_Data(20)


    ## --------------- RESET APP ---------------
    st.sidebar.markdown("<h2 style='text-align:left; float:left; color:gray; margin:0px;'>Restart the Game</h2>", unsafe_allow_html=True)
    st.session_state['init_data']['reset_app'] = st.sidebar.button('RESET GAME', key='Reset_App')
    if st.session_state['init_data']['reset_app']:
        Load_Data(st.session_state['init_data']['N_images'])


    ## --------------- SHOW INFO --------------
    #Show_Info()     
    
    
    ## --------------- CHANGE PLAYER TURN --------------- 
    if st.session_state['init_data']['change_player']:
        if st.session_state['init_data']['player2_turn']:
            st.session_state['init_data']['status']=131
            st.session_state['init_data']['player2_turn']=False
        else:
            st.session_state['init_data']['status']=132
            st.session_state['init_data']['player2_turn']=True
        st.session_state['init_data']['change_player']=False
        
        
    ## --------------- TITLE --------------- 
    if st.session_state['init_data']['finished_game']:
        st.markdown("<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1>", unsafe_allow_html=True)
    else:
        if st.session_state['init_data']['status']==0:
            st.markdown("<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1>", unsafe_allow_html=True)
        elif st.session_state['init_data']['N_players']!=1:
            st.markdown("<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1><h2 style='text-align:right;float:right; color:gray; margin:0px;'>P1 score: "
                        + str(st.session_state['init_data']['award1'])+"<p></p>P2 score: "+ str(st.session_state['init_data']['award2'])+"</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1><h2 style='text-align:right; float:right; color:gray; margin:0px;'>score: "
                    + str(st.session_state['init_data']['award1'])+"</h2>", unsafe_allow_html=True)


    ## --------------- INITIALIZATIONS ---------------
    if st.session_state['init_data']['status']==0:
        st.markdown("<h2 style='text-align:left; float:left; color:gray; margin:0px;'>Select 1 or 2 players and the number of images to use</h2>", unsafe_allow_html=True)
         
        ## Number of players
        N_Players=st.number_input('Select the number of players', min_value=1, max_value=2, value=1, step=1, format='%d', key='N_Players', help=None)
        
        if N_Players==2:
            Winner_selection_random=st.checkbox('Select to choose the winner images randomly', value=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
        else:
            Winner_selection_random=False
            
        ## Number of images
        N_Images=st.number_input('Select the number of images', min_value=5, max_value=24, value=24, step=1, format='%d', key='N_images', help=None)

        ## Type of images
        st.markdown("<h2 style='text-align:left; float:left; color:gray; margin:0px;'>Select the set of images to play with:</h2>", unsafe_allow_html=True)
        Selected_Images_Source=st.selectbox('Choose between: Celebrities images, Original "Guess Who" game images, My friends images or Your own images (selecting a source path with your images zip file)', 
                                                    ['Use Celeba dataset','Use Original "Guess Who" game images', 'Use friends dataset', 'Use images from specific path'],
                                                    index=1, key='Selected_Images_Source', help=None)
                                                    
        ## Current options selection                                           
        st.markdown("<p></p><hr><h2 style='text-align:left; float:left; color:gray; margin:0px;'>Selected options:</h2>", unsafe_allow_html=True)
        if Winner_selection_random:
            st.markdown("<h3 style='text-align:left; float:left; color:green; margin:0px;'>Players: "+str(N_Players)+" (autoselect winners randomly)</h3>", unsafe_allow_html=True)

        else:
            st.markdown("<h3 style='text-align:left; float:left; color:green; margin:0px;'>Players: "+str(N_Players)+"</h3>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:left; float:left; color:green; margin:0px;'>Number of images: "+str(N_Images)+"</h3>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:left; float:left; color:green; margin:0px;'>Images to use: "+Selected_Images_Source+"</h3>", unsafe_allow_html=True)
           
        ## Start game button
        Use_Images = st.button('START GAME (press to start playing after select the game options)', key='Use_Images')

        if Use_Images:            
            st.session_state['init_data']['N_images']=N_Images
            st.session_state['init_data']['n_images']=N_Images
            st.session_state['init_data']['n_images2']=N_Images
            st.session_state['init_data']['N_players']=N_Players
            st.session_state['init_data']['random_winner']=Winner_selection_random
            st.session_state['init_data']['current_images_discarted']=np.zeros((N_Images))
            st.session_state['init_data']['current_images_discarted2']=np.zeros((N_Images))
            st.session_state['init_data']['image_current_probs']=np.zeros((N_Images,2))
            st.session_state['init_data']['image_current_predictions']=np.zeros((N_Images))+2
            st.session_state['init_data']['Selected_Images_Source']=Selected_Images_Source
            if st.session_state['init_data']['N_players']==1:
                st.session_state['init_data']['status']=1
            else:
                st.session_state['init_data']['status']=101
            st.session_state['init_data']['player2_turn']=False
            Select_Dataset()


    ## --------------- IMAGE SELECTION ---------------
    if st.session_state['init_data']['status']==1 or st.session_state['init_data']['status']==101:
    
        ## Button - randomly change images
        st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to randomly modify the selected images.</h3>",
                                unsafe_allow_html=True)                   
        Random_Images = st.button('CHANGE IMAGES', key='Random_Images')
        if st.session_state['init_data']['images_not_selected'] or Random_Images:
            Select_Images_Randomly()
                        
        ## Button - start game
        st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to play with these images.</h3>", unsafe_allow_html=True)
        Accept_Images = st.button('SELECT THESE IMAGES', key='Accept_Images')
        
        if Accept_Images:
            ## Choose winner and start game
            st.session_state['init_data']['status']=st.session_state['init_data']['status']+10


    ## --------------- SELECT WINNER IMAGE ---------------  
    
    ## 1 player case
    if st.session_state['init_data']['status']==11:
        st.session_state['init_data']['current_winner_index']=random.choice(list(range(0,st.session_state['init_data']['N_images'])))
        st.session_state['init_data']['status']=st.session_state['init_data']['status']+20

                
    ## 2 player case - Player 1   
    if st.session_state['init_data']['status']==111:
        if st.session_state['init_data']['random_winner']:
            provisional_list=list(range(0,st.session_state['init_data']['N_images']))
            st.session_state['init_data']['current_winner_index']=random.choice(provisional_list)
            provisional_list.remove(st.session_state['init_data']['current_winner_index'])
            st.session_state['init_data']['current_winner_index2']=random.choice(provisional_list)
            st.session_state['init_data']['status']=131
        
        else:
            ## Select winner image by players
            st.markdown("<h2 style='text-align:left; float:left; color:gray; margin:0px;'>PLAYER 1: Select the image to be discovered by the Player 2</h2>", unsafe_allow_html=True)
            Image_Names_List=['Not selected']
            Image_Names_List.extend(st.session_state['init_data']['current_image_names2'])
            Player_2_Image=st.selectbox('(PLAYER 1: choose an image from the list)', 
                                                        Image_Names_List,
                                                        index=0, key='Player_2_Image', help=None)    
                               
            ## Button - start game
            if Player_2_Image!='Not selected':
                st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to validate the selection: "+Player_2_Image+"</h3>", unsafe_allow_html=True)
                Next_Player_Selection = st.button('CONFIRM CHOICE', key='Next_Player_Selection')
                if Next_Player_Selection:
                    st.session_state['init_data']['status']=st.session_state['init_data']['status']+10
                    st.session_state['init_data']['current_winner_index2']=Image_Names_List.index(Player_2_Image)-1
    
    
    ## 2 player case - Player 1 OK
    if st.session_state['init_data']['status']==121:
        st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>PLAYER 1: Press the button to hide the selection.</h3>", unsafe_allow_html=True)
        Next_Player_Selection2 = st.button('HIDE SELECTION', key='Next_Player_Selection2')
        if Next_Player_Selection2:
            st.session_state['init_data']['status']=112 
            st.session_state['init_data']['player2_turn']=True


    ## 2 player case - Player 2
    if st.session_state['init_data']['status']==112:
        ## Select winner image by players
        st.markdown("<h2 style='text-align:left; float:left; color:gray; margin:0px;'>PLAYER 2: Select the image to be discovered by the Player 1</h2>", unsafe_allow_html=True)
        Image_Names_List=['Not selected']
        Image_Names_List.extend(st.session_state['init_data']['current_image_names'])
        Player_1_Image=st.selectbox('(PLAYER 2: choose an image from the list)', 
                                                    Image_Names_List,
                                                    index=0, key='Player_1_Image', help=None)    
                           
        ## Button - start game
        if Player_1_Image!='Not selected':
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to validate the selection: "+Player_1_Image+"</h3>", unsafe_allow_html=True)
            Next_Player_Selection = st.button('CONFIRM CHOICE', key='Next_Player_Selection')
            if Next_Player_Selection:
                st.session_state['init_data']['status']=st.session_state['init_data']['status']+10
                st.session_state['init_data']['current_winner_index']=Image_Names_List.index(Player_1_Image)-1

 
    ## 2 player case - Player 2 OK
    if st.session_state['init_data']['status']==122:
        st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>PLAYER 2: Press the button to hide the selection.</h3>", unsafe_allow_html=True)
        Next_Player_Selection2 = st.button('START GAME', key='Next_Player_Selection2')
        if Next_Player_Selection2:
            st.session_state['init_data']['status']=131
            st.session_state['init_data']['player2_turn']=False


    ## 1 PLAYER GAME *********************************************************************************************************************************************************
    if st.session_state['init_data']['status']==31: 
        Ask_Question("", st.session_state['init_data']['current_winner_index'], st.session_state['init_data']['award1'])
                
        
    ## 2 PLAYER GAME - PLAYER 1 *********************************************************************************************************************************************************
    if st.session_state['init_data']['status']==131:
        Ask_Question("PLAYER 1: ", st.session_state['init_data']['current_winner_index'], st.session_state['init_data']['award1'])
    
    
    ## 2 PLAYER GAME - PLAYER 2 *********************************************************************************************************************************************************
    if st.session_state['init_data']['status']==132:    
        Ask_Question("PLAYER 2: ", st.session_state['init_data']['current_winner_index2'], st.session_state['init_data']['award2'])


    ## --------------- CALCULATE RESULTS ---------------
    if not st.session_state['init_data']['finished_game']:
    
        ## CREATE IMAGES TO SHOW
        if st.session_state['init_data']['status']>0:
            [st.session_state['init_data']['show_images'], st.session_state['init_data']['Showed_image_names']]=Show_images()        


        ## DISCARDING IMAGES AND FINAL RESULTS
        if st.session_state['init_data']['player2_turn']:
            st.session_state['init_data']['award2']=Final_Results(st.session_state['init_data']['n_images2'], st.session_state['init_data']['award2'], "PLAYER 2", st.session_state['init_data']['current_winner_index2'],st.session_state['init_data']['current_image_names2'],st.session_state['init_data']['current_images_discarted2']) 

        else:
            if st.session_state['init_data']['N_players']>1:
               st.session_state['init_data']['award1']=Final_Results(st.session_state['init_data']['n_images'], st.session_state['init_data']['award1'], "PLAYER 2", st.session_state['init_data']['current_winner_index'],st.session_state['init_data']['current_image_names'],st.session_state['init_data']['current_images_discarted']) 
            else:
               st.session_state['init_data']['award1']=Final_Results(st.session_state['init_data']['n_images'], st.session_state['init_data']['award1'], "", st.session_state['init_data']['current_winner_index'],st.session_state['init_data']['current_image_names'],st.session_state['init_data']['current_images_discarted']) 
                
          
        ## BUTTON NEXT
        if st.session_state['init_data']['show_results'] and (not st.session_state['init_data']['finished_game']):
            if st.session_state['init_data']['N_players']>1:
                st.session_state['init_data']['change_player']=True
                Next_Screen = st.button('NEXT PLAYER', key='next_screen')
            else:
                Next_Screen = st.button('NEXT QUERY', key='next_screen')
            
            
        ## SHOW CURRENT IMAGES
        if st.session_state['init_data']['status']>0:
            st.image(st.session_state['init_data']['show_images'], use_column_width=False, caption=st.session_state['init_data']['Showed_image_names'])        


    ## --------------- RELOAD GAME ---------------
    if st.session_state['init_data']['reload_game']:
        Load_Data(st.session_state['init_data']['N_images'])   


## --------------- CACHE FUCTION ---------------
@st.cache(ttl=12*3600)
def CLIP_Loading():
	  return clip.load("ViT-B/32", device="cpu", jit=False)


## --------------- STREAMLIT APP ---------------
st.set_page_config(
    layout="wide",
    page_icon='Logo DIMAI.png',
    page_title='QuienEsQuien',
    initial_sidebar_state="collapsed"
)


## --------------- START PRGRAM ---------------
Main_Program()


## --------------- SHOW INFO --------------
Show_Info()


## --------------- CLEAR RESOURCES ---------------
gc.collect()
# caching.clear_cache()
# torch.cuda.empty_cache()

    
## --------------- SHOW MORE INFO (cpu, memeory) ---------------
# st.sidebar.write(psutil.cpu_percent()) ## show info (cpu, memeory)
# st.sidebar.write(psutil.virtual_memory()) ## show info (cpu, memeory)
