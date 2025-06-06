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


## --------------- STREAMLIT APP ---------------
st.set_page_config(
    layout="wide",
    page_icon='Logo DIMAI.png',
    page_title='QuienEsQuien',
    initial_sidebar_state="collapsed"
)
    
    
## --------------- CACHE FUCTION ---------------
## @st.cache_data(ttl=12*3600)
def CLIP_Loading():
	  return clip.load("ViT-B/32", device="cpu", jit=False)


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
    n_max=len(st.session_state['init_data']['image_current_probs'][0,:])-1
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
        if np.argmax(st.session_state['init_data']['image_current_probs'][i,:])==n_max:
            st.session_state['init_data']['image_current_predictions'].append(0)        
        else:
            st.session_state['init_data']['image_current_predictions'].append(1)

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

def Final_Results(N_img, Current_award, Player_indicator, Win_index, Current_images, Img_discarded,Text_show_final_results_1,Text_show_final_results_2,Text_show_final_results_3,Text_show_final_results_4):
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
            st.markdown(Text_show_final_results_1+Player_indicator+Text_show_final_results_2+Current_images[Win_index]+Text_show_final_results_3, unsafe_allow_html=True)

            Finsih_Game = st.button(Text_show_final_results_4, key='Finsih_Game')
    return Current_award

def Ask_Question(Player_indicator,Win_index,Current_award,List_query_type,List_images_source,Text_show_elements_1,Text_show_elements_2,Text_show_elements_3,Text_show_elements_4,Text_show_elements_5,Text_show_elements_6,Text_show_elements_8,Text_show_elements_9,Text_show_elements_11,Text_show_elements_13,Text_show_elements_14,Text_show_elements_15,Text_show_elements_17,Text_show_elements_18,Text_show_elements_19,Text_show_elements_21,Text_show_elements_22,Text_show_elements_24,Text_show_elements_26,Text_show_elements_28,Text_show_elements_29,Text_show_elements_31,Text_show_elements_36,Text_show_elements_38,Text_finished_game_1,Text_finished_game_2,Text_finished_game_3,Text_finished_game_4,Text_finished_game_7,Text_finished_game_8,Text_finished_game_9,Text_show_results_1,Text_show_results_2,Text_show_results_4,Text_show_results_6,Text_show_results_8,Text_show_results_9,Text_show_results_13):
        ## Finished Game:
    if st.session_state['init_data']['finished_game']:
        st.session_state['init_data']['reload_game']=True
        Restart_App = st.button('GO TO OPTIONS SELECTION TO START NEW GAME', key='Restart_App')
        if Current_award==1 or Current_award==-1:
            st.markdown(Text_finished_game_1+Player_indicator+Text_finished_game_2+str(Current_award)+Text_finished_game_3, unsafe_allow_html=True)
        else:
            st.markdown(Text_finished_game_1+Player_indicator+Text_finished_game_2+str(Current_award)+Text_finished_game_4, unsafe_allow_html=True)
    else:
    
        st.markdown(Text_finished_game_7+Player_indicator+Text_finished_game_8, unsafe_allow_html=True)

        ## SelectBox - Select query type (game mode)
        Selected_Feature=st.selectbox(Text_finished_game_9, List_query_type, index=0, key='selected_feature', help=None)

        
        ## --------------- SHOW ELEMENTS - QUESTIONS MODE ---------------
        if Selected_Feature==List_query_type[0]:
            ## Game mode id
            st.session_state['init_data']['token_type']=0

            ## Text - Questions mode
            st.markdown(Text_show_elements_1+Player_indicator+Text_show_elements_2, unsafe_allow_html=True)
            
            ## SelectBox - Select question
            Selected_Question=st.selectbox('Suggested questions:', st.session_state['init_data']['feature_questions'], index=0, key='Selected_Question', help=None)
            st.session_state['init_data']['selected_question']=Selected_Question  # Save Info
            
            ## Current question index
            if Selected_Question not in st.session_state['init_data']['feature_questions']:
                Selected_Question=st.session_state['init_data']['feature_questions'][0]
            
            st.session_state['init_data']['questions_index']=st.session_state['init_data']['feature_questions'].index(Selected_Question)
               
            ## Text - Show current question
            st.markdown(Text_show_elements_3+Player_indicator+Text_show_elements_4+Selected_Question+Text_show_elements_5, unsafe_allow_html=True)
            
            ## Button - Use current question
            Check_Question = st.button(Text_show_elements_6, key='Check_Question')
            
            ## Check current question
            if st.session_state['init_data']['show_results']:
                st.session_state['init_data']['show_results']=False
                
            else:
                if Check_Question:
                    if st.session_state['init_data']['Selected_Images_Source']==List_images_source[1]:
                        if st.session_state['init_data']['questions_index']==7:
                            st.session_state['init_data']['current_querys']=["An illustration of a male person's face","An illustration of a female person's face",
                                                                        "An illustration of a bald man's face","An illustration of a haired person's face", 
                                                                        "An illustration of a bald person's face","An illustration of a person's face"]
                            st.session_state['init_data']['function_predict']=Predict_bald
                            
                            st.session_state['init_data']['current_querys']=["An illustration of a black haired person's face",
                                                                        "An illustration of a chocolate brown haired person's face",
                                                                        "An illustration of a neon tangerine haired person's face",
                                                                        "An illustration of a luminous blonde haired person's face",
                                                                        "An illustration of a porcelain white haired person's face"]
                            st.session_state['init_data']['function_predict']=Predict_0_vs_all
                                
                        elif st.session_state['init_data']['questions_index']==9:
                            st.session_state['init_data']['current_querys']=["An illustration of a chocolate brown haired person's face",
                                                                        "An illustration of a black haired person's face",
                                                                        "An illustration of a neon tangerine haired person's face",
                                                                        "An illustration of a luminous blonde haired person's face",
                                                                        "An illustration of a porcelain white haired person's face"]
                            st.session_state['init_data']['function_predict']=Predict_0_vs_all      
                                                     
                        elif st.session_state['init_data']['questions_index']==10:
                            st.session_state['init_data']['current_querys']=["An illustration of a luminous blonde haired person's face",
                                                                            "An illustration of a chocolate brown haired person's face",
                                                                            "An illustration of a neon tangerine haired person's face",
                                                                            "An illustration of a black haired person's face",
                                                                            "An illustration of a porcelain white haired person's face"]
                            st.session_state['init_data']['function_predict']=Predict_0_vs_all                       
                                                     
                        elif st.session_state['init_data']['questions_index']==11:
                            st.session_state['init_data']['current_querys']=["An illustration of a neon tangerine haired person's face",
                                                                            "An illustration of a chocolate brown haired person's face",
                                                                            "An illustration of a black haired person's face",
                                                                            "An illustration of a luminous blonde haired person's face",
                                                                            "An illustration of a porcelain white haired person's face"]
                            st.session_state['init_data']['function_predict']=Predict_0_vs_all  
                                                     
                        elif st.session_state['init_data']['questions_index']==12:
                            st.session_state['init_data']['current_querys']=["An illustration of a porcelain white haired person's face",
                                                                            "An illustration of a chocolate brown haired person's face",
                                                                            "An illustration of a neon tangerine haired person's face",
                                                                            "An illustration of a luminous blonde haired person's face",
                                                                            "An illustration of a black haired person's face"]
                            st.session_state['init_data']['function_predict']=Predict_0_vs_all                         

                        elif st.session_state['init_data']['questions_index']==20:
                            st.session_state['init_data']['current_querys']=["An illustration of a person's face with eyeglasses",
                                                                        "An illustration of a person's face with glasses",
                                                                        "An illustration of a person's face with sunglasses",
                                                                        "An illustration of a person's face"]
                            st.session_state['init_data']['function_predict']=Predict_all_vs_last    
                       
                        else:
                            st.session_state['init_data']['current_querys']=[st.session_state['init_data']['querys_list_yes'][st.session_state['init_data']['questions_index']],
                                                                            st.session_state['init_data']['querys_list_no'][st.session_state['init_data']['questions_index']]]
                            st.session_state['init_data']['function_predict']=Predict_0_vs_1                    
                    
                    else:
                    
                        if st.session_state['init_data']['questions_index']==7:
                            st.session_state['init_data']['current_querys']=['A picture of a male person','A picture of a female person',
                                                                            'A picture of a bald man','A picture of a haired man', 
                                                                            'A picture of a bald person','A picture of a person']
                            st.session_state['init_data']['function_predict']=Predict_bald
                            
                        elif st.session_state['init_data']['questions_index']==8:
                            st.session_state['init_data']['current_querys']=["A picture of a black haired person",
                                                                            "A picture of a tawny haired person",
                                                                            "A picture of a blond haired person",
                                                                            "A picture of a gray haired person",
                                                                            "A picture of a red haired person",
                                                                            "A picture of a green haired person",
                                                                            "A picture of a blue haired person",
                                                                            "A picture of a bald-head person"]
                            st.session_state['init_data']['function_predict']=Predict_0_vs_all
                                
                        elif st.session_state['init_data']['questions_index']==9:
                            st.session_state['init_data']['current_querys']=["A picture of a tawny haired person",
                                                                            "A picture of a black haired person",
                                                                            "A picture of a blond haired person",
                                                                            "A picture of a gray haired person",
                                                                            "A picture of a red haired person",
                                                                            "A picture of a green haired person",
                                                                            "A picture of a blue haired person",
                                                                            "A picture of a bald-head person"]
                            st.session_state['init_data']['function_predict']=Predict_0_vs_all                            
                        

                        elif st.session_state['init_data']['questions_index']==10:
                            st.session_state['init_data']['current_querys']=['A picture of a blond haired person',
                                                                        'A picture of a tawny haired person',
                                                                        'A picture of a black haired person',
                                                                        'A picture of a gray haired person',
                                                                        'A picture of a red haired person',
                                                                        'A picture of a green haired person',
                                                                        'A picture of a blue haired person',
                                                                        'A picture of a bald-head person']
                            st.session_state['init_data']['function_predict']=Predict_0_vs_all
                            
                        elif st.session_state['init_data']['questions_index']==11:
                            st.session_state['init_data']['current_querys']=['A picture of a red haired person',
                                                                        'A picture of a tawny haired person',
                                                                        'A picture of a blond haired person',
                                                                        'A picture of a gray haired person',
                                                                        'A picture of a black haired person',
                                                                        'A picture of a green haired person',
                                                                        'A picture of a blue haired person',
                                                                        'A picture of a bald-head person']
                            st.session_state['init_data']['function_predict']=Predict_0_vs_all
                            
                        elif st.session_state['init_data']['questions_index']==12:
                            st.session_state['init_data']['current_querys']=['A picture of a gray haired person',
                                                                        'A picture of a tawny haired person',
                                                                        'A picture of a blond haired person',
                                                                        'A picture of a black haired person',
                                                                        'A picture of a red haired person',
                                                                        'A picture of a green haired person',
                                                                        'A picture of a blue haired person',
                                                                        'A picture of a bald-head person']
                            st.session_state['init_data']['function_predict']=Predict_0_vs_all
                            
                        elif st.session_state['init_data']['questions_index']==20:
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
        if Selected_Feature==List_query_type[1]:
            
            ## Game mode id
            st.session_state['init_data']['token_type']=-1

            ## Text - Query mode
            st.markdown(Text_show_elements_1+Player_indicator+Text_show_elements_8, unsafe_allow_html=True)
            
            ## TextInput - Select query
            User_Input = st.text_input(Text_show_elements_9, 'A picture of a person', key='User_Input', help=None)
            st.session_state['init_data']['user_input']=User_Input  # Save Info

            ## Text - Show current query
            st.markdown(Text_show_elements_3+Player_indicator+Text_show_elements_11+User_Input+Text_show_elements_5, unsafe_allow_html=True)
            
            ## Button - Use current query
            Check_Query = st.button(Text_show_elements_13, key='Check_Query')
            
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
                        st.markdown(Text_show_elements_14+Player_indicator+Text_show_elements_15, unsafe_allow_html=True)
               
               
        ## --------------- SHOW ELEMENTS - 2 QUERYS MODE ---------------
        if Selected_Feature==List_query_type[2]:
            
            ## Game mode id
            st.session_state['init_data']['token_type']=-2

            ## Text - Querys mode
            st.markdown(Text_show_elements_1+Player_indicator+Text_show_elements_17, unsafe_allow_html=True)
            
            ## SelectBox - Select querys
            User_Input_Querys1 = st.text_input(Text_show_elements_18, 'A picture of a person', key='User_Input_Querys1', help=None)
            User_Input_Querys2 = st.text_input(Text_show_elements_19, 'A picture of a person', key='User_Input_Querys2', help=None)
            st.session_state['init_data']['user_input_querys1']=User_Input_Querys1
            st.session_state['init_data']['user_input_querys2']=User_Input_Querys2
                             
            ## Text - Show current querys
            st.markdown(Text_show_elements_3+Player_indicator+Text_show_elements_21+User_Input_Querys1+Text_show_elements_22+User_Input_Querys2+Text_show_elements_5, unsafe_allow_html=True)
            
            ## Button - Use current querys
            Check_Querys = st.button(Text_show_elements_24, key='Check_Querys')
            
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
                        st.markdown(Text_show_elements_14+Player_indicator+Text_show_elements_26, unsafe_allow_html=True)


        ## --------------- SHOW ELEMENTS - WINNER MODE ---------------
        if Selected_Feature==List_query_type[3]:
            
            ## Game mode id
            st.session_state['init_data']['token_type']=-3

            ## Text - Winner mode
            st.markdown(Text_show_elements_1+Player_indicator+Text_show_elements_28, unsafe_allow_html=True)
            
            if st.session_state['init_data']['player2_turn']:
                st.session_state['init_data']['selected_winner2']=st.selectbox(Text_show_elements_29, st.session_state['init_data']['winner_options'], index=0, key='Selected_Winner', help=None)
				
                ## Text - Show current winner
                st.markdown(Text_show_elements_3+Player_indicator+Text_show_elements_31+st.session_state['init_data']['selected_winner2']+Text_show_elements_5, unsafe_allow_html=True)
            else:
                st.session_state['init_data']['selected_winner']=st.selectbox(Text_show_elements_29, st.session_state['init_data']['winner_options'], index=0, key='Selected_Winner', help=None)
                ## Text - Show current winner
                st.markdown(Text_show_elements_3+Player_indicator+Text_show_elements_31+st.session_state['init_data']['selected_winner']+Text_show_elements_5, unsafe_allow_html=True)
            
            ## Button - Use current winner
            Check_Winner = st.button(Text_show_elements_36, key='Check_Winner')
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
                            st.markdown(Text_show_elements_14+Player_indicator+Text_show_elements_38, unsafe_allow_html=True)
                    else:
                        if st.session_state['init_data']['selected_winner'] in st.session_state['init_data']['current_image_names']:
                            st.session_state['init_data']['selected_winner_index']=np.where(st.session_state['init_data']['selected_winner']==st.session_state['init_data']['current_image_names'])[0]
                            st.session_state['init_data']['image_current_predictions']=np.zeros(st.session_state['init_data']['n_images'])
                            st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['selected_winner_index']]=1    
                            st.session_state['init_data']['show_results']=True
                        else:
                            st.markdown(Text_show_elements_14+Player_indicator+Text_show_elements_38, unsafe_allow_html=True)


        ## --------------- ACTIONS SHOWING RESULTS ---------------
        if st.session_state['init_data']['show_results']:
            ## Verify win index error
            if Win_index>len(st.session_state['init_data']['image_current_predictions'])-1:
                Win_index=0
                st.markdown("WIN INDEX CHANGED", unsafe_allow_html=True)
                st.markdown(Win_index, unsafe_allow_html=True)
                st.markdown(st.session_state['init_data']['image_current_predictions'], unsafe_allow_html=True)

            ## Show current results
            if st.session_state['init_data']['token_type']==0:
                if st.session_state['init_data']['image_current_predictions'][Win_index]:
                    st.markdown(Text_show_results_1+st.session_state['init_data']['selected_question']+Text_show_results_2, unsafe_allow_html=True)
                else:
                    st.markdown(Text_show_results_1+st.session_state['init_data']['selected_question']+Text_show_results_4, unsafe_allow_html=True)
                    
            if st.session_state['init_data']['token_type']==-1:
                if st.session_state['init_data']['image_current_predictions'][Win_index]:
                    st.markdown(Text_show_results_1+st.session_state['init_data']['user_input']+Text_show_results_6, unsafe_allow_html=True)
                else:
                    st.markdown(Text_show_results_1+st.session_state['init_data']['user_input']+Text_show_results_8, unsafe_allow_html=True)
                    
            if st.session_state['init_data']['token_type']==-2:
                if st.session_state['init_data']['image_current_predictions'][Win_index]:
                    st.markdown(Text_show_results_9+st.session_state['init_data']['user_input_querys1']+Text_show_elements_5, unsafe_allow_html=True)
                else:
                    st.markdown(Text_show_results_9+st.session_state['init_data']['user_input_querys2']+Text_show_elements_5, unsafe_allow_html=True)
              
            if st.session_state['init_data']['player2_turn']:
                if st.session_state['init_data']['token_type']==-3:
                    if not st.session_state['init_data']['selected_winner2']==st.session_state['init_data']['current_image_names'][Win_index]:
                        st.markdown(Text_show_results_13+st.session_state['init_data']['selected_winner2']+Text_show_elements_5, unsafe_allow_html=True)
            else:
                if st.session_state['init_data']['token_type']==-3:
                    if not st.session_state['init_data']['selected_winner']==st.session_state['init_data']['current_image_names'][Win_index]:
                        st.markdown(Text_show_results_13+st.session_state['init_data']['selected_winner']+Text_show_elements_5, unsafe_allow_html=True)

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
    previous_predictions=st.session_state['init_data']['image_current_predictions'] 
    current_index=0
    new_index=0
    index_not_new=True
    if st.session_state['init_data']['player2_turn']:
        for i in range(len(st.session_state['init_data']['current_images_discarted2'])):
            if st.session_state['init_data']['current_images_discarted2'][i]==0 and st.session_state['init_data']['image_current_predictions'][i]!=st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index2']]:
                st.session_state['init_data']['current_images_discarted2'][i]=1
 
        st.session_state['init_data']['image_current_predictions']=[]
        previous_names=st.session_state['init_data']['current_image_names2']
        previous_files=st.session_state['init_data']['image_current_paths2'] 
        st.session_state['init_data']['current_image_names2']=[]
        st.session_state['init_data']['image_current_paths2']=[]    
        for i in range(st.session_state['init_data']['n_images2']):
            if st.session_state['init_data']['current_images_discarted2'][current_index]==0:
                st.session_state['init_data']['image_current_paths2'].append(previous_files[current_index])
                st.session_state['init_data']['current_image_names2'].append(previous_names[current_index])
                st.session_state['init_data']['image_current_predictions'].append(previous_predictions[current_index])
                if current_index==st.session_state['init_data']['current_winner_index2'] and index_not_new:
                    st.session_state['init_data']['current_winner_index2']=new_index
                    index_not_new=False
					
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

        st.session_state['init_data']['image_current_predictions']=[]
        previous_names=st.session_state['init_data']['current_image_names']
        previous_files=st.session_state['init_data']['image_current_paths']    
        st.session_state['init_data']['current_image_names']=[]
        st.session_state['init_data']['image_current_paths']=[]  
        for i in range(st.session_state['init_data']['n_images']):
            if st.session_state['init_data']['current_images_discarted'][current_index]==0:
                st.session_state['init_data']['image_current_paths'].append(previous_files[current_index])
                st.session_state['init_data']['current_image_names'].append(previous_names[current_index])
                st.session_state['init_data']['image_current_predictions'].append(previous_predictions[current_index])
                if current_index==st.session_state['init_data']['current_winner_index'] and index_not_new:
                    st.session_state['init_data']['current_winner_index']=new_index
                    index_not_new=False
                    
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
            #st.markdown(current_index, unsafe_allow_html=True)
            #st.markdown(winner_index, unsafe_allow_html=True)
            #st.markdown(st.session_state['init_data']['image_current_predictions'], unsafe_allow_html=True)
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

def Show_Info(Text_questions_list_1):
    #st.sidebar.markdown("<p></p><hr><h2 style='text-align:left; float:left; color:gray; margin:0px;'>INFO</h2>", unsafe_allow_html=True)
    st.sidebar.markdown(Text_questions_list_1, unsafe_allow_html=True)
    st.sidebar.write(st.session_state['init_data']['feature_questions'])
    # st.sidebar.write(st.session_state['init_data'])

def Load_Data(N):
    st.session_state['init_data']={
        'language':'English',
        'status':-1,
        'award1':100,
        'award2':100,
        'N_images':N,
        'N_images_init':N,
        'n_images':N,
        'n_images2':N,
        'N_players':1,
        'N_players_init':1,
        'Selected_Images_Source':'Use Celeba dataset',
        'Selected_Images_Source_init':0,
        'zip_file':'guess_who_images.zip',
        'previous_zip_file':'guess_who_images.zip',
        'special_images_names':False,
        'images_not_selected':True,
        'images_loaded':False,
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


def ReLoad_Data(list_images_source):
    st.session_state['init_data']['status']=-1
    st.session_state['init_data']['award1']=100
    st.session_state['init_data']['award2']=100
    st.session_state['init_data']['N_images_init']=st.session_state['init_data']['N_images']
    st.session_state['init_data']['n_images']=st.session_state['init_data']['N_images_init']
    st.session_state['init_data']['n_images2']=st.session_state['init_data']['N_images_init']
    st.session_state['init_data']['N_players_init']=st.session_state['init_data']['N_players']
    # st.session_state['init_data']['Selected_Images_Source']='Use Celeba dataset'
    st.session_state['init_data']['Selected_Images_Source_init']=list_images_source.index(st.session_state['init_data']['Selected_Images_Source'])
    # st.session_state['init_data']['zip_file']='guess_who_images.zip'
    # st.session_state['init_data']['previous_zip_file']='guess_who_images.zip'
    # st.session_state['init_data']['special_images_names']=False
    st.session_state['init_data']['images_not_selected']=True
    st.session_state['init_data']['images_loaded']=False
    # st.session_state['init_data']['token_type']=0
    # st.session_state['init_data']['feature_questions']=['Are you a MAN?', 'Are you a WOMAN?', 'Are you an ATTRACTIVE person?', 'Are you an CHUBBY person?', 'Are you YOUNG?',
    #                 'Are you a person with RECEDING HAIRLINES?', 'Are you SMILING?','Are you BALD?', 
    #                 'Do you have BLACK HAIR?', 'Do you have BROWN HAIR?', 'Do you have BLOND HAIR?', 'Do you have RED HAIR?',
    #                 'Do you have GRAY HAIR?', 'Do you have STRAIGHT HAIR?', 'Do you have WAVY HAIR?',
    #                 'Do you have a BEARD?', 'Do you have a MUSTACHE?', 'Do you have SIDEBURNS?',
    #                 'Do you have a GOATEE?', 'Do you wear HEAVY MAKEUP?', 'Do you wear EYEGLASSES?',
    #                 'Do you have BUSHY EYEBROWS?', 'Do you have a DOUBLE CHIN?', 
    #                 'Do you have a high CHEECKBONES?', 'Do you have SLIGHTLY OPEN MOUTH?', 
    #                 'Do you have NARROWED EYES?', 'Do you have an OVAL FACE?', 
    #                 'Do you have PALE SKIN?', 'Do you have a POINTY NOSE?', 'Do you have ROSY CHEEKS?', 
    #                 "Do you have FIVE O'CLOCK SHADOW?", 'Do you have ARCHED EYEBROWS?', 'Do you have BUGS UNDER your EYES?', 
    #                 'Do you have BANGS?', 'Do you have a BIG LIPS?', 'Do you have a BIG NOSE?',
    #                 'Are you wearing EARRINGS?', 'Are you wearing a HAT?', 
    #                 'Are you wearing LIPSTICK?', 'Are you wearing NECKLACE?', 
    #                 'Are you wearing NECKTIE?']
    # st.session_state['init_data']['selected_question']='Are you a MAN?'
    # st.session_state['init_data']['questions_index']=0
    st.session_state['init_data']['show_results']=False
    # st.session_state['init_data']['current_querys']=['A picture of a person','A picture of a person']
    # st.session_state['init_data']['function_predict']=Predict_0_vs_1
    # st.session_state['init_data']['querys_list_yes']=['A picture of a male person', 'A picture of a female person', 'A picture of an attractive person', 'A picture of a fat person', 'A picture of a young person', 
    #         'A picture of a receding-hairline person  ', 'A picture of a smily person', 'A picture of a bald person',
    #         'A picture of a person with black hair', 'A picture of a person with brown hair', 'A picture of a person with blond hair', 'A picture of a person with red hair', 
    #         'A picture of a person with gray hair', 'A picture of a person with straight hair', 'A picture of a person with wavy hair', 
    #         'A picture of a unshaved person', 'A picture of a mustachioed person', 'A picture of a person with bushy sideburns', 
    #         'A picture of a person with goatee', 'A picture of a person with heavy makeup', 'A picture of a person with eyeglasses',             
    #         'A picture of a person with bushy eyebrows', 'A picture of a double chin person', 
    #         'A picture of a person with high cheekbones', 'A picture of a person with opened mouth', 
    #         'A picture of a person with narrow eyes', 'A picture of a person with an oval-shaped face', 
    #         'A picture of a person wiht pale skin', 'A picture of a pointy-nosed person ', 'A picture of a person with colored cheeks', 
    #         "A picture of a five o'clock shadow person", 'A picture of a rounded eyebrows person', 'A picture of a person with bags under the eyes', 
    #         'A picture of a person with bangs', 'A picture of a wide-liped person', 'A picture of a big-nosed person',            
    #         'A picture of a person with earrings', 'A picture of a person with hat', 
    #         'A picture of a person with lipstick', 'A picture of a necklaced person', 
    #         'A picture of a necktied person'
    #         ]
    # st.session_state['init_data']['querys_list_no']=['A picture of a female person', 'A picture of a male person', 'A picture of an ugly person', 'A picture of a slender person', 'A picture of an aged person', 
    #         'A picture of a hairy person', 'A picture of a person', 'A picture of a hairy person',
    #         'A picture of a person', 'A picture of a person', 'A picture of a person', 'A picture of a person', 
    #         'A picture of a person', 'A picture of a person with wavy hair', 'A picture of a person with straight hair', 
    #         'A picture of a glabrous person', 'A picture of a person', 'A picture of a person with shaved sideburns', 
    #         'A picture of a person', 'A picture of a person with light makeup', 'A picture of a person ',             
    #         'A picture of a person with sparse eyebrows', 'A picture of a person with a double chin', 
    #         'A picture of a person with low cheekbones', 'A picture of a person with closed mouth', 
    #         'A picture of a person with wide eyes', 'A picture of a person with a normal-shaped face', 
    #         'A picture of a person wiht tanned skin', 'A picture of a flat-nosed person', 'A picture of a person with pale cheeks', 
    #         "A picture of a shaved or unshaved person", 'A picture of a person a straight eyebrows person', 'A picture of a person with with smooth skin under the eyes', 
    #         'A picture of a person', 'A picture of a narrow-liped person', 'A picture of a small-nosed person',            
    #         'A picture of a person', 'A picture of a person with hair', 
    #         'A picture of a person with natural lips', 'A picture of a person', 
    #         'A picture of a person'
    #         ]
    # st.session_state['init_data']['token_type']=0
    # st.session_state['init_data']['user_input']='A picture of a person'
    # st.session_state['init_data']['user_input_querys1']='A picture of a person'
    # st.session_state['init_data']['user_input_querys2']='A picture of a person'
    st.session_state['init_data']['image_current_probs']=np.zeros((st.session_state['init_data']['N_images'],2))
    st.session_state['init_data']['selected_winner']='Winner '
    st.session_state['init_data']['selected_winner2']='Winner '
    st.session_state['init_data']['reset_app']=False
    st.session_state['init_data']['change_player']=False
    st.session_state['init_data']['player2_turn']=False
    st.session_state['init_data']['finished_game']=False
    st.session_state['init_data']['reload_game']=False       
    # st.session_state['init_data']['random_winner']=False
    st.session_state['init_data']['show_images']=[]
    st.session_state['init_data']['previous_discarding_images_number']=0
    st.session_state['init_data']['selected_winner_index']=0
    st.session_state['init_data']['selected_winner_index2']=0
    st.session_state['init_data']['current_images_discarted']=np.zeros((st.session_state['init_data']['N_images']))
    st.session_state['init_data']['current_images_discarted2']=np.zeros((st.session_state['init_data']['N_images']))
    st.session_state['init_data']['current_winner_index']=0
    st.session_state['init_data']['current_winner_index2']=0
    st.session_state['init_data']['current_image_names']=[]
    st.session_state['init_data']['current_image_names2']=[]
    st.session_state['init_data']['image_current_paths']=[]
    st.session_state['init_data']['image_current_paths2']=[]
    st.session_state['init_data']['winner_options']=[]
    st.session_state['init_data']['image_current_predictions']=np.zeros((st.session_state['init_data']['N_images']))+2

## --------------- MAIN FUCTION ---------------
def Main_Program():
	
    ## --------------- LOAD DATA ---------------
    if 'init_data' not in st.session_state:
        Load_Data(20)
    current_status=st.session_state['init_data']['status']
    
    ## --------------- SET TEXTS ---------------
    if st.session_state['init_data']['language']=='English':
	
        List_Query_Type=['Ask a Question', 'Create your own query', 'Create your own 2 querys','Select a Winner']

        List_Images_Source=['Use Celeba dataset','Use Original "Guess Who" game images','Use images from specific path']
        ## List_Images_Source=['Use Celeba dataset','Use Original "Guess Who" game images', 'Use friends dataset', 'Use images from specific path']
        Text_Show_Final_Results_1="<h1 style='text-align:left; color:gray; margin:0px;'>CONGRATULATIONS <span style='text-align:left; color:green; margin:0px;'>"

        Text_Show_Final_Results_2="<span style='text-align:left; color:gray; margin:0px;'>! THE WINNER PICTURE IS: <span style='text-align:left; color:green; margin:0px;'>"

        Text_Show_Final_Results_3="</h1>"

        Text_Show_Final_Results_4="FINISH GAME"


        Text_Finished_Game_1="<h1 style='text-align:left; color:black; margin:0px;'>¡¡¡ <span style='text-align:left; color:green; margin:0px;'>"

        Text_Finished_Game_2="<span style='text-align:left; color:black; margin:0px;'>YOU WIN WITH <span style='text-align:left; color:green; margin:0px;'>"

        Text_Finished_Game_3="<span style='text-align:left; color:black; margin:0px;'> POINT !!!</h1>"
	
        Text_Finished_Game_4="<span style='text-align:left; color:black; margin:0px;'> POINTS !!!</h1>"

        Text_Finished_Game_7="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>"

        Text_Finished_Game_8="Select a type of Query to play.</h2>"

        Text_Finished_Game_9="Ask a question from a list, create your query or select a winner:"


        Tex_Images_Source_1='Choose between: Celebrities images, Original "Guess Who" game images or Your own images (selecting a source path with your images zip file)'
        ## Tex_Images_Source_1='Choose between: Celebrities images, Original "Guess Who" game images, My friends images or Your own images (selecting a source path with your images zip file)'


        Text_Show_Elements_1="<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"

        Text_Show_Elements_2="Select a Question from the list.</h3>"

        Text_Show_Elements_3="<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"

        Text_Show_Elements_4="Current Question: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"

        Text_Show_Elements_5="</h3>"

        Text_Show_Elements_6='USE THIS QUESTION'

        Text_Show_Elements_8="Write your own query and press the button.</h3>"

        Text_Show_Elements_9="It is recommended to use a text like: 'A picture of a ... person' or 'A picture of a person ...' (CLIP will check -> 'Your query'  vs  'A picture of a person' )"

        Text_Show_Elements_11="Current Query: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"

        Text_Show_Elements_13="USE MY OWN QUERY"

        Text_Show_Elements_14="<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"

        Text_Show_Elements_15="Your query must be different of 'A picture of a person'.</h3>"

        Text_Show_Elements_17="Write your own querys by introducing 2 opposite descriptions.</h3>"

        Text_Show_Elements_18="Write your 'True' query:"

        Text_Show_Elements_19="Write your 'False' query:"

        Text_Show_Elements_21="Current Querys: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"

        Text_Show_Elements_22=" vs "

        Text_Show_Elements_24='USE MY OWN QUERYS'

        Text_Show_Elements_26="Your two own querys must be different.</h3>"

        Text_Show_Elements_28="Select a Winner picture name.</h3>"

        Text_Show_Elements_29="If you are inspired, Select a Winner image directly:"

        Text_Show_Elements_31="Current Winner: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"

        Text_Show_Elements_36="CHECK THIS WINNER"

        Text_Show_Elements_38="Your must select a not discarded picture.</h3>"


        Text_Show_Results_1="<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"

        Text_Show_Results_2="</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>YES</h3>"

        Text_Show_Results_4="</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>NO</h3>"

        Text_Show_Results_6="</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>TRUE</h3>"

        Text_Show_Results_8="</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>FALSE</h3>"

        Text_Show_Results_9="<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>The most accurate query is:</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>"

        Text_Show_Results_13="<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>The winner picture is not:</h3><h3 style='text-align:left; float:center; color:red; margin:0px;'>"

        
        Text_Questions_List_1="<p></p><hr><h2 style='text-align:left; float:left; color:gray; margin:0px;'>List of avalaible Questions</h2>"


        Text_Reset_App_1="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>Restart the Game</h2>"

        Text_Reset_App_2='RESET GAME'


        Text_Title_1="<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1>"

        Text_Title_3="<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1><h2 style='text-align:right;float:right; color:gray; margin:0px;'>P1 score: "

        Text_Title_4="<p></p>P2 score: "

        Text_Title_5="</h2>"

        Text_Title_6="<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1><h2 style='text-align:right; float:right; color:gray; margin:0px;'>score: "


        Text_Inicializations_1="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>Select 1 or 2 players and the number of images to use</h2>"

        Text_Inicializations_2="Select the number of players"

        Text_Inicializations_3="Select to choose the winner images randomly"

        Text_Inicializations_4='Select the number of images'

        Text_Inicializations_5="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>Select the set of images to play with:</h2>"

        Text_Inicializations_6="<p></p><hr><h2 style='text-align:left; float:left; color:gray; margin:0px;'>Selected options:</h2>"

        Text_Inicializations_7="<h3 style='text-align:left; float:left; color:green; margin:0px;'>Players: "

        Text_Inicializations_8=" (autoselect winners randomly)</h3>"
        
        Text_Inicializations_9="<h4 style='text-align:left; float:left; color:black; margin:0px;'>This application uses the CLIP algorithm, which allows matching images and text in English, due it was trained with many images obtained from Internet and their natural language descriptions.<h4 style='text-align:left; float:left; color:black; margin:0px;'>The funny thing is that CLIP is who answer the questions asked by the players, and therefore it is responsible for discarding the images after each question.<h4 style='text-align:left; float:left; color:black; margin:0px;'>Internally CLIP responds to the true or false query about the winner image, and then discards all the images that do not coincide with the answer of the winner one.<h4 style='text-align:left; float:left; color:black; margin:0px;'>CLIP is not an infallible algorithm and therefore may make mistakes when answering the questions. What we do know for sure is that CLIP will always answer the same if the same query is made, since it is a deterministic algorithm.<h4 style='text-align:left; float:left; color:black; margin:0px;'>Here is DAL-E, an application made with CLIP that allows you to create images from text (https://openai.com/blog/dall-e/).</h4>"
	
        Text_Inicializations_10="<h3 style='text-align:left; float:left; color:green; margin:0px;'>Language selected: "
        
        Text_Inicializations_11="<h3 style='text-align:left; float:left; color:green; margin:0px;'>Number of images: "
        
        Text_Inicializations_12="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>Select language / Escull l'idioma</h2>"

        Text_Inicializations_13="<h3 style='text-align:left; float:left; color:green; margin:0px;'>Images to use: "
        
        Text_Inicializations_14="Languages list"
        
        Text_Inicializations_15="START GAME (press to start playing after select the game options)"
	
        Text_Inicializations_16="<p></p><hr><h3 style='text-align:left; float:left; color:blue; margin:0px;'>Description<h3>"
	
        Text_Inicializations_17="<p></p><hr><h3 style='text-align:left; float:left; color:blue; margin:0px;'>Let's go to play<h3>"
        
        
        Text_Language_1="<h2 style='text-align:left; float:left; color:black; margin:0px;'>Select a '.zip' file with the images to play in '.jpg' or '.png' format.</h2>"
        
        Text_Language_2="Select images to play"
        
        Text_Language_3="CHANGE IMAGES"
        

        Text_Image_Selection_1="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to randomly modify the selected images.</h3>"

        Text_Image_Selection_2="CHANGE IMAGES"

        Text_Image_Selection_3="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to play with these images.</h3>"

        Text_Image_Selection_4="SELECT THESE IMAGES"


        Text_Select_Winner_1="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>PLAYER 1: Select the image to be discovered by the Player 2</h2>"

        Text_Select_Winner_2="Not selected"

        Text_Select_Winner_3="(PLAYER 1: choose an image from the list)"

        Text_Select_Winner_4="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to validate the selection: "

        Text_Select_Winner_6='CONFIRM CHOICE'

        Text_Select_Winner_7="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>PLAYER 1: Press the button to change player turn.</h3>"

        Text_Select_Winner_8='GO TO PLAYER 2 TURN'

        Text_Select_Winner_9="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>PLAYER 2: Select the image to be discovered by the Player 1</h2>"

        Text_Select_Winner_10='(PLAYER 2: choose an image from the list)'

        Text_Select_Winner_11="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to validate the selection: "

        Text_Select_Winner_13="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>PLAYER 2: Press the button to change player turn.</h3>"

        Text_Select_Winner_14='START GAME'
            
            
        Text_Ask_Question_0=""

        Text_Ask_Question_1="PLAYER 1: "

        Text_Ask_Question_2="PLAYER 2: "
	    
        Text_Calculate_Results_1="PLAYER 1"            
            
        Text_Calculate_Results_2="PLAYER 2"

        Text_Calculate_Results_3='NEXT PLAYER'

        Text_Calculate_Results_4='NEXT QUERY'
        	
        if st.session_state['init_data']['Selected_Images_Source']==List_Images_Source[1]:
            st.session_state['init_data']['feature_questions']=["Are you a MAN?", "Are you a WOMAN?", "Are you an ATTRACTIVE person?", "Are you an CHUBBY person?", "Are you YOUNG?",
                        "Are you a person with RECEDING HAIRLINES?", "Are you SMILING?","Are you BALD?", 
                        "Do you have BLACK HAIR?", "Do you have BROWN HAIR?", "Do you have YELLOW HAIR?", "Do you have ORANGE HAIR?",
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
                "An illustration of a person's face with black hair", "An illustration of a person's face with brown hair", "An illustration of a person's face with yellow hair", "An illustration of a person's face with orange hair",
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
                "An illustration of a person's face with forehead or hat", "A picture of a narrow-liped person", "A picture of a small-nosed person",            
                "An illustration of a person's face", "An illustration of a person's face with hair", 
                "An illustration of a person's face with natural lips", "An illustration of a person's face", 
                "An illustration of a person's face"]    
        
        else:
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
                
    else:
        List_Query_Type=['Fer una pregunta', 'Fer una consulta en anglès', 'Fer una doble consulta en anglès','Seleccionar un guanyador']

        List_Images_Source=["Usar imatges de famosos (el dataset Celeba)","Usar imatges del joc original 'Quí és Quí?'", "Usar imatges d'un arxiu 'zip'"]
        ## List_Images_Source=["Usar imatges de famosos (el dataset Celeba)","Usar imatges del joc original 'Quí és Quí?'","Usar imatges d'amics", "Usar imatges d'un arxiu 'zip'"]

        Text_Show_Final_Results_1="<h1 style='text-align:left; color:gray; margin:0px;'>FELICITATS <span style='text-align:left; color:green; margin:0px;'>"

        Text_Show_Final_Results_2="<span style='text-align:left; color:gray; margin:0px;'>! LA IMATGE GUANYADORA ES: <span style='text-align:left; color:green; margin:0px;'>"

        Text_Show_Final_Results_3="</h1>"

        Text_Show_Final_Results_4="JOC ACABAT"


        Text_Finished_Game_1="<h1 style='text-align:left; color:black; margin:0px;'>¡¡¡ <span style='text-align:left; color:green; margin:0px;'>"

        Text_Finished_Game_2="<span style='text-align:left; color:black; margin:0px;'>HAS GUANYAT AMB <span style='text-align:left; color:green; margin:0px;'>"

        Text_Finished_Game_3="<span style='text-align:left; color:black; margin:0px;'> PUNTS !!!</h1>"
	
        Text_Finished_Game_4="<span style='text-align:left; color:black; margin:0px;'> PUNTS !!!</h1>"

        Text_Finished_Game_7="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>"

        Text_Finished_Game_8="Tria com fer la consulta.</h2>"

        Text_Finished_Game_9="Triar una pregunta d'una llista, crear la teva consulta en anglès o selecciona directament un guanyador:"


        Tex_Images_Source_1="Tria entre: imatges de famosos, imatges del joc original 'Quí és Quí?' o les imatges que vulguis (proporcionant un arxiu '.zip')"
        ## Tex_Images_Source_1="Tria entre: imatges de famosos, imatges del joc original 'Quí és Quí?', imatges d'amics o les imatges que vulguis (proporcionant un arxiu '.zip')"


        Text_Show_Elements_1="<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"

        Text_Show_Elements_2="Tria una pregunta de la llista.</h3>"

        Text_Show_Elements_3="<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"

        Text_Show_Elements_4="Pregunta seleccionada: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"

        Text_Show_Elements_5="</h3>"

        Text_Show_Elements_6='FER AQUESTA PREGUNTA'

        Text_Show_Elements_8="Escriu la teva consulta en anglès i prem el botó.</h3>"

        Text_Show_Elements_9="És recomanable usar una estructura com ara: 'A picture of a ... person' o 'A picture of a person ...' (CLIP comprova -> 'La teva consulta en anglès'  vs  'A picture of a person' )"

        Text_Show_Elements_11="Current Query: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"

        Text_Show_Elements_13="FER AQUESTA CONSULTA"

        Text_Show_Elements_14="<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>"

        Text_Show_Elements_15="La teva consulta ha de ser diferent de 'A picture of a person'.</h3>"

        Text_Show_Elements_17="Escriu les teves consultes usant dos frases oposades en anglès.</h3>"

        Text_Show_Elements_18="Escriu la teva consulta 'Certa':"

        Text_Show_Elements_19="Escriu la teva consulta 'Falsa':"

        Text_Show_Elements_21="Consultes a fer: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"

        Text_Show_Elements_22=" vs "

        Text_Show_Elements_24='FER AQUESTA DOBLE CONSULTA'

        Text_Show_Elements_26="Les teves dos consultes han de ser diferents.</h3>"

        Text_Show_Elements_28="Tria la imatge guanyadora.</h3>"

        Text_Show_Elements_29="Si estàs inpirat, selecciona la imatge guanyadora directament:"

        Text_Show_Elements_31="Guanyador seleccionat: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"

        Text_Show_Elements_36="COMPROVA AQUEST GUANYADOR"

        Text_Show_Elements_38="Has de triar una imatge no descartada.</h3>"

        Text_Show_Results_1="<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"

        Text_Show_Results_2="</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>SI</h3>"

        Text_Show_Results_4="</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>NO</h3>"

        Text_Show_Results_6="</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>CERT</h3>"

        Text_Show_Results_8="</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>FALS</h3>"

        Text_Show_Results_9="<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>La cosulta certa és:</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>"

        Text_Show_Results_13="<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>La imatge guanyadora no és:</h3><h3 style='text-align:left; float:center; color:red; margin:0px;'>"


        Text_Questions_List_1="<p></p><hr><h2 style='text-align:left; float:left; color:gray; margin:0px;'>Llista de preguntes disponibles</h2>"


        Text_Reset_App_1="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>Reiniciar el joc</h2>"

        Text_Reset_App_2='REINICIAR EL JOC'


        Text_Title_1="<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Quí és Quí?</h1>"

        Text_Title_3="<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Quí és Quí</h1><h2 style='text-align:right;float:right; color:gray; margin:0px;'>Punts del Jugador 1: "

        Text_Title_4="<p></p>Punts del Jugador 2: "

        Text_Title_5="</h2>"

        Text_Title_6="<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Quí és Quí?</h1><h2 style='text-align:right; float:right; color:gray; margin:0px;'>Punts: "


        Text_Inicializations_1="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>Tria entre 1 o 2 jugadors i el número d'imatges a fer servir</h2>"

        Text_Inicializations_2="Tria el número de jugadors"

        Text_Inicializations_3="Selecciona per escollir les imatges guanyadores aleatòriament"

        Text_Inicializations_4="Tria el número d'imatges"

        Text_Inicializations_5="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>Tria el conjunt d'imatges a fer servir:</h2>"

        Text_Inicializations_6="<p></p><hr><h3 style='text-align:left; float:left; color:blue; margin:0px;'>Opcions seleccionades:</h2>"

        Text_Inicializations_7="<h3 style='text-align:left; float:left; color:green; margin:0px;'>Jugadors: "

        Text_Inicializations_8=" (seleccionar les imatges guanyadores aleatòriament)</h3>"
        
        Text_Inicializations_9="<h4 style='text-align:left; float:left; color:black; margin:0px;'>Aquesta aplicació utilitza l'algoritme CLIP, que permet combinar imatges i text en anglès, ja que s'ha entrenat amb moltes imatges obtingudes d'Internet i les seves descripcions en llenguatge natural.<h4 style='text-align:left; float:left; color:black; margin:0px;'>El més curiós és que CLIP és qui respon a les preguntes que fan els jugadors, i per tant s'encarrega de descartar les imatges després de cada pregunta.<h4 style='text-align:left; float:left; color:black; margin:0px;'>Internament, CLIP respon a la consulta vertadera o falsa sobre la imatge guanyadora, i després descarta totes les imatges que no coincideixen amb la resposta de la guanyadora.<h4 style='text-align:left; float:left; color:black; margin:0px;'>CLIP no és un algorisme infal·lible i, per tant, pot cometre errors en respondre les preguntes. El que sí que sabem del cert és que CLIP sempre respondrà igual si es fa la mateixa consulta, ja que és un algorisme determinista.<h4 style='text-align:left; float:left; color:black; margin:0px;'>Aquí teniu DAL-E, una aplicació feta amb CLIP que permet crear imatges a partir de text (https://openai.com/blog/dall-e/).</h4>"
                
        Text_Inicializations_10="<h3 style='text-align:left; float:left; color:green; margin:0px;'>Idioma seleccionat: "
        
        Text_Inicializations_11="<h3 style='text-align:left; float:left; color:green; margin:0px;'>Número d'imatges: "
        
        Text_Inicializations_12="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>Select language / Escull l'idioma</h2>"

        Text_Inicializations_13="<h3 style='text-align:left; float:left; color:green; margin:0px;'>Imatges a usar: "
        
        Text_Inicializations_14="Llista d'idiomes"
        
        Text_Inicializations_15="COMENÇAR EL JOC (prem per iniciar el joc després de triar les opcions)"
	
        Text_Inicializations_16="<p></p><hr><h3 style='text-align:left; float:left; color:blue; margin:0px;'>Descripció<h3>"
	
        Text_Inicializations_17="<p></p><hr><h3 style='text-align:left; float:left; color:blue; margin:0px;'>Anem a jugar<h3>"
        
        
        Text_Language_1="<h2 style='text-align:left; float:left; color:black; margin:0px;'>Selecciona un arxiu '.zip' amb les imatges a jugar en format '.jpg' o '.png'.</h2>"
        
        Text_Language_2="Tria les imatges amb les que jugar"
        
        Text_Language_3="CANVIAR IMATGES"
                

        Text_Image_Selection_1="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Prem per canviar les imatges aleatòriament.</h3>"

        Text_Image_Selection_2="CANVIAR IMATGES"

        Text_Image_Selection_3="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Prem el botó per jugar amb aquestes imatges.</h3>"

        Text_Image_Selection_4="JUGAR AMB AQUESTES IMATGES"


        Text_Select_Winner_1="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>JUGADOR 1: tria la imatge guanyadora del Jugador 2</h2>"

        Text_Select_Winner_2="No seleccionada"

        Text_Select_Winner_3="(JUGADOR 1: tria una imatge de la llista)"

        Text_Select_Winner_4="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Prem el botó per validar la selecció: "

        Text_Select_Winner_6='CONFIRMAR IMATGE GUANYADORA'

        Text_Select_Winner_7="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>JUGADOR 1: prem el botó per canviar de torn.</h3>"

        Text_Select_Winner_8='TORN DEL JUGADOR 2'

        Text_Select_Winner_9="<h2 style='text-align:left; float:left; color:gray; margin:0px;'>JUGADOR 2: tria la imatge guanyadora del Jugador 1</h2>"

        Text_Select_Winner_10='(JUGADOR 2: tria una imatge de la llista)'

        Text_Select_Winner_11="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Prem el botó per validar la selecció: "

        Text_Select_Winner_13="<h3 style='text-align:left; float:left; color:gray; margin:0px;'>JUGADOR 2: prem el botó per canviar de torn.</h3>"

        Text_Select_Winner_14='INICIAR JOC'
            
            
        Text_Ask_Question_0=""

        Text_Ask_Question_1="JUGADOR 1: "

        Text_Ask_Question_2="JUGADOR 2: "
            
        Text_Calculate_Results_1="JUGADOR 1"

        Text_Calculate_Results_2="JUGADOR 2"

        Text_Calculate_Results_3='SEGÜENT JUGADOR'

        Text_Calculate_Results_4='SEGÜENT CONSULTA'
                     
                     
        if st.session_state['init_data']['Selected_Images_Source']==List_Images_Source[1]:
            st.session_state['init_data']['feature_questions']=['Ets un HOME?', 'Ets una DONA?', 'Ets una persona ATRACTIVA?', 'Ets una persona GRASSONETA ?', 'Ets JOVE?',
                        "Ets una persona que PERD EL CABELL?", "SOMRIUS?", "Ets CALB?",
                        'Tens els CABELLS NEGRE?', 'Tens els CABELLS MARRONS?', 'Tens els CABELLS ROSSOS?', 'Tens els CABELLS TARONJA?',
                        'Tens els els CABELLS BLANCS?', 'Tens els CABELLS LLISOS?', 'Tens els CABELLS ONDULATS?',
                        'Tens BARBA?', 'Tens BIGOTI?', 'Tens PATILLES?',
                        'Tens una PERILLA?', 'Portes MAQUILLATGE?', 'Portes ULLERES?',
                        'Tens CELLES PELUDES?', 'Tens DOBLE BARBETA?',
                        'Tens els PÒMULS ALTS?', 'Tens la BOCA LLEUGERAMENT OBERTA?',
                        'Tens ELS ULLS ENTRETANCATS?', 'Tens la CARA OVALADA?',
                        'Tens la PEL PÀL·LIDA?', 'Tens el NAS DE PUNTA?', 'Tens les GALTES ROSADES?',
                        "Tens OMBRA DE LES 5 EN PUNT?", "Tens CELLES ARQUEJADES?", "FAS ULLERES?",
                        "Ten SERRELL?", "Tens els LLAVIS GRANS?", "Tens el NAS GRAN?",
                        'Portes ARRACADES?', 'Portes BARRET?',
                        'Portes PINTALLAVIS?', 'Portes COLLARETS?',
                        'Portes CORBATA?']
        
            st.session_state['init_data']['querys_list_yes']=["A picture of a male person", "A picture of a female person", "A picture of an attractive person", "A picture of a fat person", "A picture of a young person", 
                "A picture of a receding-hairline person  ", "A picture of a smily person", "A picture of a bald person",
                "An illustration of a person's face with black hair", "An illustration of a person's face with brown hair", "An illustration of a person's face with yellow hair",  "An illustration of a person's face with orange hair",
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
            st.session_state['init_data']['feature_questions']=['Ets un HOME?', 'Ets una DONA?', 'Ets una persona ATRACTIVA?', 'Ets una persona GRASSONETA ?', 'Ets JOVE?',
                        "Ets una persona que PERD EL CABELL?", "SOMRIUS?", "Ets CALB?",
                        'Tens els CABELLS NEGRE?', 'Tens els CABELLS MARRONS?', 'Tens els CABELLS ROSSOS?', 'Tens els CABELLS ROJOS?',
                        'Tens els els CABELLS GRISOS?', 'Tens els CABELLS LLISOS?', 'Tens els CABELLS ONDULATS?',
                        'Tens BARBA?', 'Tens BIGOTI?', 'Tens PATILLES?',
                        'Tens una PERILLA?', 'Portes MAQUILLATGE?', 'Portes ULLERES?',
                        'Tens CELLES PELUDES?', 'Tens DOBLE BARBETA?',
                        'Tens els PÒMULS ALTS?', 'Tens la BOCA LLEUGERAMENT OBERTA?',
                        'Tens ELS ULLS ENTRETANCATS?', 'Tens la CARA OVALADA?',
                        'Tens la PEL PÀL·LIDA?', 'Tens el NAS DE PUNTA?', 'Tens les GALTES ROSADES?',
                        "Tens OMBRA DE LES 5 EN PUNT?", "Tens CELLES ARQUEJADES?", "FAS ULLERES?",
                        "Ten SERRELL?", "Tens els LLAVIS GRANS?", "Tens el NAS GRAN?",
                        'Portes ARRACADES?', 'Portes BARRET?',
                        'Portes PINTALLAVIS?', 'Portes COLLARETS?',
                        'Portes CORBATA?']
                        
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


    ## --------------- RESET APP ---------------
    st.sidebar.markdown(Text_Reset_App_1, unsafe_allow_html=True)
    st.session_state['init_data']['reset_app'] = st.sidebar.button(Text_Reset_App_2, key='Reset_App')
    if st.session_state['init_data']['reset_app']:
        ReLoad_Data(List_Images_Source)
         
     
    ## --------------- SHOW INFO --------------
    Show_Info(Text_Questions_List_1)

    
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
        st.markdown(Text_Title_1, unsafe_allow_html=True)
    else:
        if st.session_state['init_data']['status']==0:
            st.markdown(Text_Title_1, unsafe_allow_html=True)
        elif st.session_state['init_data']['N_players']!=1:
            st.markdown(Text_Title_3 + str(st.session_state['init_data']['award1'])+ Text_Title_4 + str(st.session_state['init_data']['award2'])+Text_Title_5, unsafe_allow_html=True)
        else:
            st.markdown(Text_Title_6 + str(st.session_state['init_data']['award1'])+ Text_Title_5, unsafe_allow_html=True)


    ## --------------- LANGUAGE ---------------
    if st.session_state['init_data']['status']==-1:
            st.markdown(Text_Inicializations_12, unsafe_allow_html=True)
            col1, col2 = st.columns([1,5])
            with col1:
                Set_English=st.button("ENGLISH", key='Set_english')

            with col2:
                Set_Catala=st.button("CATALÀ", key='Set_catala')

            if Set_English:
                st.session_state['init_data']['language']='English'
                st.session_state['init_data']['status']=0

            if Set_Catala:
                st.session_state['init_data']['language']='Català'
                st.session_state['init_data']['status']=0
	
	  
    ## --------------- INITIALIZATIONS ---------------
    if st.session_state['init_data']['status']==0:
        st.markdown(Text_Inicializations_16, unsafe_allow_html=True)
        st.markdown(Text_Inicializations_9, unsafe_allow_html=True)
        st.markdown(Text_Inicializations_17, unsafe_allow_html=True)
        st.markdown(Text_Inicializations_1, unsafe_allow_html=True)

        ## Number of players
        N_Players=st.number_input(Text_Inicializations_2, min_value=1, max_value=2, value=st.session_state['init_data']['N_players_init'], step=1, format='%d', key='N_Players', help=None)
        
        if N_Players==2:
            Winner_selection_random=st.checkbox(Text_Inicializations_3, value=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False)
        else:
            Winner_selection_random=False
            
        ## Number of images
        N_Images=st.number_input(Text_Inicializations_4, min_value=5, max_value=24, value=st.session_state['init_data']['N_images_init'], step=1, format='%d', key='N_images', help=None)

        ## Type of images
        st.markdown(Text_Inicializations_5, unsafe_allow_html=True)
        Selected_Images_Source=st.selectbox(Tex_Images_Source_1, List_Images_Source, index=st.session_state['init_data']['Selected_Images_Source_init'], key='Selected_Images_Source', help=None)
  						 
        ## Current options selection                                           
        st.markdown(Text_Inicializations_6, unsafe_allow_html=True)
        if Winner_selection_random:
            st.markdown(Text_Inicializations_7+str(N_Players)+Text_Inicializations_8, unsafe_allow_html=True)

        else:
            st.markdown(Text_Inicializations_7+str(N_Players)+Text_Show_Elements_5, unsafe_allow_html=True)
        st.markdown(Text_Inicializations_11+str(N_Images)+Text_Show_Elements_5, unsafe_allow_html=True)
        st.markdown(Text_Inicializations_13+Selected_Images_Source+Text_Show_Elements_5, unsafe_allow_html=True)
        st.markdown(Text_Inicializations_10+st.session_state['init_data']['language'], unsafe_allow_html=True)
           
        ## Start game button
        Use_Images = st.button(Text_Inicializations_15, key='Use_Images')

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
            st.session_state['init_data']['Selected_Images_Source_init']=List_Images_Source.index(Selected_Images_Source)
            if st.session_state['init_data']['N_players']==1:
                st.session_state['init_data']['status']=1
            else:
                st.session_state['init_data']['status']=101
            st.session_state['init_data']['player2_turn']=False

            ## Select zip file
            if st.session_state['init_data']['Selected_Images_Source']==List_Images_Source[0]:
                st.session_state['init_data']['zip_file']='guess_who_images.zip'
            elif st.session_state['init_data']['Selected_Images_Source']==List_Images_Source[1]:
                st.session_state['init_data']['zip_file']='Original.zip'
            ## elif st.session_state['init_data']['Selected_Images_Source']==List_Images_Source[2]:
                ## st.session_state['init_data']['zip_file']='frifam.zip'
            else:
                st.session_state['init_data']['zip_file']='Use images from specific path'
            

    ## --------------- IMAGE SELECTION ---------------
    if st.session_state['init_data']['status']==1 or st.session_state['init_data']['status']==101:
        if st.session_state['init_data']['Selected_Images_Source']==List_Images_Source[2]:
        ## if st.session_state['init_data']['Selected_Images_Source']==List_Images_Source[3]:
            ## Specific source text
            st.markdown(Text_Language_1, unsafe_allow_html=True)
                
            Uploaded_File = st.file_uploader(Text_Language_2, type=[".zip"],accept_multiple_files=False, key="Uploaded_file")                    

            if Uploaded_File is not None:
                if Uploaded_File!=st.session_state['init_data']['zip_file']:            
                    st.session_state['init_data']['zip_file']= Uploaded_File
                    st.session_state['init_data']['images_loaded']=True
                    Select_Images_Randomly()
                
                if Uploaded_File==st.session_state['init_data']['zip_file']:  
                    ## Button - randomly change Celeba images
                    Random_Images = st.button(Text_Language_3, key='Random_Images')
                    if Random_Images:
                        Select_Images_Randomly()
                            
                    ## Button - start game
                    st.markdown(Text_Image_Selection_3, unsafe_allow_html=True)
                    Accept_Images = st.button(Text_Image_Selection_4, key='Accept_Images')
                    
                    if Accept_Images:
                        ## Choose winner and start game
                        st.session_state['init_data']['status']=st.session_state['init_data']['status']+10
                        
        else:
            st.session_state['init_data']['images_loaded']=True
            ## Button - randomly change images
            st.markdown(Text_Image_Selection_1, unsafe_allow_html=True)
            Random_Images = st.button(Text_Image_Selection_2, key='Random_Images')
            if st.session_state['init_data']['images_not_selected'] or Random_Images:
                Select_Images_Randomly()
                            
            ## Button - start game
            st.markdown(Text_Image_Selection_3, unsafe_allow_html=True)
            Accept_Images = st.button(Text_Image_Selection_4, key='Accept_Images')
            
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
            st.markdown(Text_Select_Winner_1, unsafe_allow_html=True)
            Image_Names_List=[Text_Select_Winner_2]
            Image_Names_List.extend(st.session_state['init_data']['current_image_names2'])
            Player_2_Image=st.selectbox(Text_Select_Winner_3, Image_Names_List, index=0, key='Player_2_Image', help=None)    
                               
            ## Button - start game
            if Player_2_Image!=Text_Select_Winner_2:
                st.markdown(Text_Select_Winner_4+Player_2_Image+Text_Show_Elements_5, unsafe_allow_html=True)
                Next_Player_Selection = st.button(Text_Select_Winner_6, key='Next_Player_Selection')
                if Next_Player_Selection:
                    st.session_state['init_data']['status']=st.session_state['init_data']['status']+10
                    st.session_state['init_data']['current_winner_index2']=Image_Names_List.index(Player_2_Image)-1
    
	
    ## 2 player case - Player 1 OK
    if st.session_state['init_data']['status']==121:
        st.markdown(Text_Select_Winner_7, unsafe_allow_html=True)
        Next_Player_Selection2 = st.button(Text_Select_Winner_8, key='Next_Player_Selection2')
        if Next_Player_Selection2:
            st.session_state['init_data']['status']=112 
            st.session_state['init_data']['player2_turn']=True


    ## 2 player case - Player 2
    if st.session_state['init_data']['status']==112:
        ## Select winner image by players
        st.markdown(Text_Select_Winner_9, unsafe_allow_html=True)
        Image_Names_List=[Text_Select_Winner_2]
        Image_Names_List.extend(st.session_state['init_data']['current_image_names'])
        Player_1_Image=st.selectbox(Text_Select_Winner_10, Image_Names_List, index=0, key='Player_1_Image', help=None)    
                           
        ## Button - start game
        if Player_1_Image!=Text_Select_Winner_2:
            st.markdown(Text_Select_Winner_11+Player_1_Image+Text_Show_Elements_5, unsafe_allow_html=True)
            Next_Player_Selection = st.button(Text_Select_Winner_6, key='Next_Player_Selection')
            if Next_Player_Selection:
                st.session_state['init_data']['status']=st.session_state['init_data']['status']+10
                st.session_state['init_data']['current_winner_index']=Image_Names_List.index(Player_1_Image)-1

 
    ## 2 player case - Player 2 OK
    if st.session_state['init_data']['status']==122:
        st.markdown(Text_Select_Winner_13, unsafe_allow_html=True)
        Next_Player_Selection2 = st.button(Text_Select_Winner_14, key='Next_Player_Selection2')
        if Next_Player_Selection2:
            st.session_state['init_data']['status']=131
            st.session_state['init_data']['player2_turn']=False


    ## 1 PLAYER GAME *********************************************************************************************************************************************************
    if st.session_state['init_data']['status']==31: 
        Ask_Question(Text_Ask_Question_0, st.session_state['init_data']['current_winner_index'], st.session_state['init_data']['award1'],List_Query_Type,List_Images_Source,Text_Show_Elements_1, Text_Show_Elements_2,Text_Show_Elements_3,Text_Show_Elements_4,Text_Show_Elements_5,Text_Show_Elements_6,Text_Show_Elements_8,Text_Show_Elements_9,Text_Show_Elements_11,Text_Show_Elements_13,Text_Show_Elements_14,Text_Show_Elements_15,Text_Show_Elements_17,Text_Show_Elements_18,Text_Show_Elements_19,Text_Show_Elements_21,Text_Show_Elements_22,Text_Show_Elements_24,Text_Show_Elements_26,Text_Show_Elements_28,Text_Show_Elements_29,Text_Show_Elements_31,Text_Show_Elements_36,Text_Show_Elements_38,Text_Finished_Game_1,Text_Finished_Game_2,Text_Finished_Game_3,Text_Finished_Game_4,Text_Finished_Game_7,Text_Finished_Game_8,Text_Finished_Game_9,Text_Show_Results_1,Text_Show_Results_2,Text_Show_Results_4,Text_Show_Results_6,Text_Show_Results_8,Text_Show_Results_9,Text_Show_Results_13)
                
        
    ## 2 PLAYER GAME - PLAYER 1 *********************************************************************************************************************************************************
    if st.session_state['init_data']['status']==131:
        Ask_Question(Text_Ask_Question_1, st.session_state['init_data']['current_winner_index'], st.session_state['init_data']['award1'],List_Query_Type,List_Images_Source,Text_Show_Elements_1, Text_Show_Elements_2,Text_Show_Elements_3,Text_Show_Elements_4,Text_Show_Elements_5,Text_Show_Elements_6,Text_Show_Elements_8,Text_Show_Elements_9,Text_Show_Elements_11,Text_Show_Elements_13,Text_Show_Elements_14,Text_Show_Elements_15,Text_Show_Elements_17,Text_Show_Elements_18,Text_Show_Elements_19,Text_Show_Elements_21,Text_Show_Elements_22,Text_Show_Elements_24,Text_Show_Elements_26,Text_Show_Elements_28,Text_Show_Elements_29,Text_Show_Elements_31,Text_Show_Elements_36,Text_Show_Elements_38,Text_Finished_Game_1,Text_Finished_Game_2,Text_Finished_Game_3,Text_Finished_Game_4,Text_Finished_Game_7,Text_Finished_Game_8,Text_Finished_Game_9,Text_Show_Results_1,Text_Show_Results_2,Text_Show_Results_4,Text_Show_Results_6,Text_Show_Results_8,Text_Show_Results_9,Text_Show_Results_13)
    
    
    ## 2 PLAYER GAME - PLAYER 2 *********************************************************************************************************************************************************
    if st.session_state['init_data']['status']==132:    
        Ask_Question(Text_Ask_Question_2, st.session_state['init_data']['current_winner_index2'], st.session_state['init_data']['award2'],List_Query_Type,List_Images_Source,Text_Show_Elements_1, Text_Show_Elements_2,Text_Show_Elements_3,Text_Show_Elements_4,Text_Show_Elements_5,Text_Show_Elements_6,Text_Show_Elements_8,Text_Show_Elements_9,Text_Show_Elements_11,Text_Show_Elements_13,Text_Show_Elements_14,Text_Show_Elements_15,Text_Show_Elements_17,Text_Show_Elements_18,Text_Show_Elements_19,Text_Show_Elements_21,Text_Show_Elements_22,Text_Show_Elements_24,Text_Show_Elements_26,Text_Show_Elements_28,Text_Show_Elements_29,Text_Show_Elements_31,Text_Show_Elements_36,Text_Show_Elements_38,Text_Finished_Game_1,Text_Finished_Game_2,Text_Finished_Game_3,Text_Finished_Game_4,Text_Finished_Game_7,Text_Finished_Game_8,Text_Finished_Game_9,Text_Show_Results_1,Text_Show_Results_2,Text_Show_Results_4,Text_Show_Results_6,Text_Show_Results_8,Text_Show_Results_9,Text_Show_Results_13)

	
	    ## --------------- CALCULATE RESULTS ---------------
    if not st.session_state['init_data']['finished_game']:
    
        ## CREATE IMAGES TO SHOW
        if st.session_state['init_data']['status']>0 and st.session_state['init_data']['images_loaded']:
	## and ((st.session_state['init_data']['status']!=0 and st.session_state['init_data']['status']!=1 and st.session_state['init_data']['status']!=101) or st.session_state['init_data']['Selected_Images_Source']!=List_Images_Source[2]):
        ## if st.session_state['init_data']['status']>0:
            [st.session_state['init_data']['show_images'], st.session_state['init_data']['Showed_image_names']]=Show_images()        

        ## DISCARDING IMAGES AND FINAL RESULTS
        if st.session_state['init_data']['player2_turn']:
            st.session_state['init_data']['award2']=Final_Results(st.session_state['init_data']['n_images2'], st.session_state['init_data']['award2'], Text_Calculate_Results_2, st.session_state['init_data']['current_winner_index2'],st.session_state['init_data']['current_image_names2'],st.session_state['init_data']['current_images_discarted2'],Text_Show_Final_Results_1,Text_Show_Final_Results_2,Text_Show_Final_Results_3,Text_Show_Final_Results_4) 
        else:
            if st.session_state['init_data']['N_players']>1:      
                st.session_state['init_data']['award1']=Final_Results(st.session_state['init_data']['n_images'], st.session_state['init_data']['award1'], Text_Calculate_Results_1, st.session_state['init_data']['current_winner_index'],st.session_state['init_data']['current_image_names'],st.session_state['init_data']['current_images_discarted'],Text_Show_Final_Results_1,Text_Show_Final_Results_2,Text_Show_Final_Results_3,Text_Show_Final_Results_4)
            else:      
                st.session_state['init_data']['award1']=Final_Results(st.session_state['init_data']['n_images'], st.session_state['init_data']['award1'], Text_Calculate_Results_1, st.session_state['init_data']['current_winner_index'],st.session_state['init_data']['current_image_names'],st.session_state['init_data']['current_images_discarted'],Text_Show_Final_Results_1,Text_Show_Final_Results_2,Text_Show_Final_Results_3,Text_Show_Final_Results_4)

	    
        ## BUTTON NEXT
        if st.session_state['init_data']['show_results'] and (not st.session_state['init_data']['finished_game']):
            if st.session_state['init_data']['N_players']>1:
                st.session_state['init_data']['change_player']=True
                Next_Screen = st.button(Text_Calculate_Results_3, key='next_screen')
            else:
                Next_Screen = st.button(Text_Calculate_Results_4, key='next_screen')
            
            
        ## SHOW CURRENT IMAGES
        if st.session_state['init_data']['status']>0 and st.session_state['init_data']['images_loaded']:
	## and ((st.session_state['init_data']['status']!=0 and st.session_state['init_data']['status']!=1 and st.session_state['init_data']['status']!=) or st.session_state['init_data']['Selected_Images_Source']!=List_Images_Source[2]):
        ## if st.session_state['init_data']['status']>0:
            st.image(st.session_state['init_data']['show_images'], use_container_width=False, caption=st.session_state['init_data']['Showed_image_names'])        
		


    ## --------------- RELOAD GAME ---------------
    if st.session_state['init_data']['reload_game']:
        ReLoad_Data(List_Images_Source)   
        
        
    ## --------------- CHECK STATUS CHANGE ---------------

    if current_status!=st.session_state['init_data']['status'] and (not st.session_state['init_data']['finished_game']) and current_status!=st.session_state['init_data']['status']!=-1:
        st.rerun()

## --------------- START PRGRAM ---------------
Main_Program()


## --------------- CLEAR RESOURCES ---------------
gc.collect()
# caching.clear_cache()
# torch.cuda.empty_cache()

    
## --------------- SHOW MORE INFO (cpu, memeory) ---------------
# st.sidebar.write(psutil.cpu_percent()) ## show info (cpu, memeory)
# st.sidebar.write(psutil.virtual_memory()) ## show info (cpu, memeory)
