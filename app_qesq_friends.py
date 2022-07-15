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
    
def Predict_1_vs_2():
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
        if st.session_state['init_data']['image_current_probs'][i,1]>st.session_state['init_data']['image_current_probs'][i,2]:
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
   
def CLIP_Process():
    ## Tokenization process
    clip_model, clip_transform=Load_CLIP()
    clip_text = clip.tokenize(st.session_state['init_data']['current_querys']).to("cpu")
    n_tokens=len(st.session_state['init_data']['current_querys'])
    
    ## Image Process
    st.session_state['init_data']['image_current_probs']=np.zeros((st.session_state['init_data']['n_images'],n_tokens))
    for i in range(st.session_state['init_data']['n_images']):
        current_image_file = Load_Image(i)
        img_preprocessed = clip_transform(Image.fromarray(current_image_file)).unsqueeze(0).to("cpu")
        img_logits, img_logits_txt = clip_model(img_preprocessed, clip_text)
        st.session_state['init_data']['image_current_probs'][i,:]=np.round(img_logits.detach().numpy()[0],2)
        gc.collect()
        
    del i,n_tokens,clip_model,clip_transform,clip_text,current_image_file,img_preprocessed,img_logits,img_logits_txt
    gc.collect()
       
def Image_discarding():
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
    for current_index in range(st.session_state['init_data']['n_images']):
        if st.session_state['init_data']['show_results']:
            current_line_width=4
            if st.session_state['init_data']['image_current_predictions'][current_index]==st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index']]:
                current_color=np.array([0,255,0])
            else:
                current_color=np.array([255,0,0]) 
        else:
            current_line_width=2
            current_color=np.zeros(3)  
            
        image_size=240
        current_image_file=Load_Image(current_index)        
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
    return showed_images
    
    
def find_same_name(index,names_list):
    name_find=names_list[index].find('-')
    fixed_name=names_list[index][:name_find]
    index_list=[]
    for i in range(0,len(names_list)):
        name_find=names_list[i].find('-')
        if fixed_name==names_list[i][:name_find]:
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
        
    archive = zipfile.ZipFile(st.session_state['init_data']['zip_file'], 'r')
    listOfFileNames = archive.namelist()     
    # listOfFileNames = find_list_elements(listOfFileElements,[])     
    image_index_all=list(range(len(listOfFileNames)))
    
    image_index.append(random.choice(image_index_all))
    
    if st.session_state['init_data']['images_with_name']:  
        image_delete=find_same_name(image_index[0],listOfFileNames)
        
        for i in image_delete:
            image_index_all.remove(i)  
            
        current_index=1 
        while len(image_index)<st.session_state['init_data']['n_images']:  

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
            current_name = current_name[:current_name.find('-')-1]
            st.session_state['init_data']['current_image_names'].append(current_name)
                    
        st.session_state['init_data']['current_image_names']=np.array(st.session_state['init_data']['current_image_names'])
        st.session_state['init_data']['image_current_paths']=np.array(st.session_state['init_data']['image_current_paths'])

    else:
        image_index_all.remove(image_index[0])
        current_index=1
        
        while len(image_index)<st.session_state['init_data']['n_images']:
            image_index.append(random.choice(image_index_all))
            image_index_all.remove(image_index[current_index])
            current_index+=1
            
       # Iterate over the file names
        for current_index in image_index:
            image_current_path=listOfFileNames[current_index]
            st.session_state['init_data']['image_current_paths'].append(image_current_path)
            st.session_state['init_data']['current_image_names'].append(image_current_path[-10:-4])
                    
        st.session_state['init_data']['current_image_names']=np.array(st.session_state['init_data']['current_image_names'])
        st.session_state['init_data']['image_current_paths']=np.array(st.session_state['init_data']['image_current_paths'])
       

    del image_index,archive,listOfFileNames,image_index_all,current_index,image_current_path
    
    
def Select_Images_Randomly_name_management():
    st.session_state['init_data']['image_current_paths']=[]
    st.session_state['init_data']['current_image_names']=[]
    image_index=[]
        
    archive = zipfile.ZipFile(st.session_state['init_data']['zip_file'], 'r')
    listOfFileNames = archive.namelist()        
    image_index_all=list(range(len(listOfFileNames)))
    image_index.append(random.choice(image_index_all))
    image_index_all.remove(image_index[0])
    current_index=1
    while len(image_index)<st.session_state['init_data']['n_images']:
        image_index.append(random.choice(image_index_all))
        image_index_all.remove(image_index[current_index])
        current_index+=1
        
   # Iterate over the file names
    for current_index in image_index:
        image_current_path=listOfFileNames[current_index]
        st.session_state['init_data']['image_current_paths'].append(image_current_path)
        st.session_state['init_data']['current_image_names'].append(image_current_path[-10:-4])
                
    st.session_state['init_data']['current_image_names']=np.array(st.session_state['init_data']['current_image_names'])
    st.session_state['init_data']['image_current_paths']=np.array(st.session_state['init_data']['image_current_paths'])
    del image_index,archive,listOfFileNames,image_index_all,current_index,image_current_path
  
def Load_Image(current_index):
    archive = zipfile.ZipFile(st.session_state['init_data']['zip_file'], 'r')
    image_current_path=st.session_state['init_data']['image_current_paths'][current_index]
    image_file=Image.open(BytesIO(archive.read(image_current_path)))
    image_file = image_file.convert('RGB')  
    
    if not (image_file.size[0] == 224 and image_file.size[1] == 224): 
        image_file=image_file.resize((224, 224))
    del image_current_path,archive
    return np.array(image_file)

def Show_Info():
    st.sidebar.markdown('## INFO')
    st.sidebar.write(st.session_state['init_data'])
    st.sidebar.markdown('#### Questions List:')
    st.sidebar.write(st.session_state['init_data']['feature_questions'])

def Load_Data(total_images_number):
    st.session_state['init_data']={
        'images_selected':False,
        'show_results':False,
        'start_game':False,
        'finished_game':False,
        'reload_game':False,
        'images_with_name':False,
        'award':100,
        'token_type':0,
        'questions_index':0,
        'selected_question':'Are you a MAN?',
        'first_question':'Are you a MAN?',
        'user_input':'A picture of a person',
        'user_input_querys1':'A picture of a person',
        'user_input_querys2':'A picture of a person',
        'current_querys':['A picture of a person','A picture of a person'],
        'selected_winner':'Winner not selected',
        'current_winner_index':-1,
        'N_images':total_images_number,
        'n_images':total_images_number,
        'zip_file':'guess_who_images.zip',
        'previous_zip_file':'guess_who_images.zip',
        'Showed_image_names':[],
        'current_images_discarted':np.zeros((total_images_number)),
        'winner_options':[],
        'current_image_names':[],
        'image_current_paths':[],
        'clip_tokens':['A picture of a person','A picture of a person'],
        'path_info':'D:/Datasets/Celeba/',
        'path_imgs':'D:/Datasets/Celeba/img_celeba/',
        'querys_list_yes':['A picture of a male person', 'A picture of a female person', 'A picture of an attractive person', 'A picture of a fat person', 'A picture of a young person', 
            'A picture of a receding-hairline person  ', 'A picture of a smily person', 'A picture of a bald person',
            'A picture of a person with black hair', 'A picture of a person with brown hair', 'A picture of a person with blond hair', 'A picture of a person with red hair', 
            'A picture of a person with gray hair', 'A picture of a person with straight hair', 'A picture of a person with wavy hair', 
            'A picture of a glabrous person', 'A picture of a mustachioed person', 'A picture of a person with bushy sideburns', 
            'A picture of a person with goatee', 'A picture of a person with heavy makeup', 'A picture of a person with eyeglasses ',             
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
        'querys_list_no':['A picture of a female person', 'A picture of a male person', 'A picture of an ugly person', 'A picture of a slender person', 'A picture of a aged person', 
            'A picture of a hairy person', 'A picture of a person', 'A picture of a hairy person',
            'A picture of a person', 'A picture of a person', 'A picture of a person', 'A picture of a person', 
            'A picture of a person', 'A picture of a person with wavy hair', 'A picture of a person with straight hair', 
            'A picture of a unshaved person', 'A picture of a person', 'A picture of a person with shaved sideburns', 
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
        'previous_discarding_images_number':0,
        'function_predict':Predict_0_vs_1,
        'image_current_probs':np.zeros((total_images_number,2)),
        'image_current_predictions':np.zeros((total_images_number))+2}
    
    Select_Images_Randomly()
    del total_images_number


## --------------- MAIN FUCTION ---------------

def Main_Program():

    ## SIDEBAR
    st.sidebar.markdown('# OPTIONS PANEL')

    ## Reset App APP
    Reset_App = st.sidebar.button('RESET GAME', key='Reset_App')

    ## Images number
    st.sidebar.markdown('# Number of images')
    Total_Images_Number=st.sidebar.number_input('Select the number of images of the game and press "RESET GAME"', min_value=5, max_value=40, value=20, 
                                                                        step=1, format='%d', key='Total_Images_Number', help=None)

    ## INITIALIZATIONS
     
    Feature_Options=['Ask a Question', 'Create your own query', 'Create your own 2 querys','Select a Winner']

    ## Load data to play
    if 'init_data' not in st.session_state:
        Load_Data(Total_Images_Number)
     
    ## Title
    if st.session_state['init_data']['finished_game']:
        st.markdown("<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1><h2 style='text-align:right; float:right; color:gray; margin:0px;'>score: "+ str(st.session_state['init_data']['award'])+"</h2>", unsafe_allow_html=True)

    ## GAME
    if Reset_App:
        Load_Data(Total_Images_Number)
        Restart_App = st.button('GO TO IMAGES SELECTION TO START A NEW GAME', key='Restart_App')
    else:                    
        ## FINISHED GAME BUTTON TO RELOAD GAME
        if st.session_state['init_data']['finished_game']:
            Restart_App = st.button('GO TO IMAGES SELECTION TO START NEW GAME', key='Restart_App')
            if st.session_state['init_data']['award']==1 or st.session_state['init_data']['award']==-1:
                st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>¡¡¡ FINISHED WITH</h1><h1 style='text-align:left; float:left; color:green; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>"+str(st.session_state['init_data']['award'])+"</h1><h1 style='text-align:left; float:left; color:black; margin:0px;'>POINT !!!</h1>", unsafe_allow_html=True)
            else:
                st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>¡¡¡ FINISHED WITH</h1><h1 style='text-align:left; float:left; color:green; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>"+str(st.session_state['init_data']['award'])+"</h1><h1 style='text-align:left; float:left; color:black; margin:0px;'>POINTS !!!</h1>", unsafe_allow_html=True)
        
        else:
            st.session_state['init_data']['images_selected']=False
            
            ## INITIALIZATION (SELECT FIGURES)
            if not st.session_state['init_data']['start_game']:
                
                ## Select images source
                st.sidebar.markdown('## Image selection source:')
                Selected_Images_Source=st.sidebar.selectbox('(Choose between default random images or specific source path)', 
                                                            ['Use Celeba dataset random images',
                                                            'Use friends random images',
                                                            'Use images from specific path'],
                                                            index=0, key='Selected_Images_Source', help=None)
            
                ## Select images source - Celeba default
                if Selected_Images_Source=='Use Celeba dataset random images':
                
                    st.session_state['init_data']['images_with_name']=False
                    st.session_state['init_data']['zip_file']='guess_who_images.zip'
                    if st.session_state['init_data']['zip_file']!=st.session_state['init_data']['previous_zip_file']:
                        st.session_state['init_data']['previous_zip_file']=st.session_state['init_data']['zip_file']
                        Select_Images_Randomly()
                        st.session_state['init_data']['winner_options']=st.session_state['init_data']['current_image_names']

                    ## Default source text
                    st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>1. Choose the images you like.</h2>",
                                unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to randomly modify the selected images.</h3>",
                                unsafe_allow_html=True)
                    
                    ## Button - randomly change Celeba images
                    Random_Images = st.button('CHANGE IMAGES', key='Random_Images')
                    if Random_Images:
                        Select_Images_Randomly()
                        st.session_state['init_data']['winner_options']=st.session_state['init_data']['current_image_names']
                        
                    ## Button - start game
                    st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>2. Press the button to start the game.</h2>", unsafe_allow_html=True)
                    Use_Images = st.button('START GAME', key='Use_Images')
                    
                    if Use_Images:
                        ## Choose winner and start game
                        st.session_state['init_data']['current_winner_index']=random.choice(list(range(0,st.session_state['init_data']['N_images'])))
                        st.session_state['init_data']['start_game']=True
                        st.session_state['init_data']['images_selected']=True
                        
                ## Select images source - Friends default
                if Selected_Images_Source=='Use friends random images':
                
                    st.session_state['init_data']['images_with_name']=True
                    st.session_state['init_data']['zip_file']='frifam.zip'
                    if st.session_state['init_data']['zip_file']!=st.session_state['init_data']['previous_zip_file']:
                        st.session_state['init_data']['previous_zip_file']=st.session_state['init_data']['zip_file']
                        Select_Images_Randomly()
                        st.session_state['init_data']['winner_options']=st.session_state['init_data']['current_image_names']                    
                    
                    ## Default source text
                    st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>1. Choose the images you like.</h2>",
                                unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to randomly modify the selected images.</h3>",
                                unsafe_allow_html=True)
                    
                    ## Button - randomly change Celeba images
                    Random_Images = st.button('CHANGE IMAGES', key='Random_Images')
                    if Random_Images:
                        Select_Images_Randomly()
                        st.session_state['init_data']['winner_options']=st.session_state['init_data']['current_image_names']
                        
                    ## Button - start game
                    st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>2. Press the button to start the game.</h2>", unsafe_allow_html=True)
                    Use_Images = st.button('START GAME', key='Use_Images')
                    
                    if Use_Images:
                        ## Choose winner and start game
                        st.session_state['init_data']['current_winner_index']=random.choice(list(range(0,st.session_state['init_data']['N_images'])))
                        st.session_state['init_data']['start_game']=True
                        st.session_state['init_data']['images_selected']=True              
                    
                ## Select images source - Celeba specific path
                if Selected_Images_Source=='Use images from specific path':
                    
                    st.session_state['init_data']['images_with_name']=False
                    ## Specific source text
                    st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>1. Choose the images you like.</h2>",
                                unsafe_allow_html=True)
                                
                    st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>To use images from specific path, press 'Use Path'. Press it again to randomly modify the selected images.</h3>",
                                unsafe_allow_html=True) 
                        
                    Uploaded_File = st.file_uploader("Select images to play", type=[".zip"],accept_multiple_files=False, key="Uploaded_file")                    

                    if Uploaded_File is not None:
                        st.session_state['init_data']['zip_file']= Uploaded_File
                        Select_Images_Randomly()
                        st.session_state['init_data']['winner_options']=st.session_state['init_data']['current_image_names']
                        
                    ## Button - randomly change Celeba images
                    Random_Images = st.button('CHANGE IMAGES', key='Random_Images')
                    if Random_Images:
                        Select_Images_Randomly()
                        st.session_state['init_data']['winner_options']=st.session_state['init_data']['current_image_names']
                    
                    if not (st.session_state['init_data']['zip_file']=='guess_who_images.zip' or st.session_state['init_data']['zip_file']=='frifam.zip'):
                    
                        ## Button - start game
                        st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>2. Press the button to start the game.</h2>", unsafe_allow_html=True)
                        Use_Images = st.button('START GAME', key='Use_Images')
                        
                        if Use_Images:
                            ## Choose winner and start game
                            st.session_state['init_data']['current_winner_index']=random.choice(list(range(0,st.session_state['init_data']['N_images'])))
                            st.session_state['init_data']['start_game']=True
                            st.session_state['init_data']['images_selected']=True
                        
                    
            ## RUN GAME
            if st.session_state['init_data']['start_game']:
            
                ## Text - Select query type (game mode)
                if st.session_state['init_data']['images_selected']:
                    st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>3. Select a type of Query to play.</h2>", unsafe_allow_html=True)
                else:
                    st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>1. Select a type of Query to play.</h2>", unsafe_allow_html=True)
                
                ## SelectBox - Select query type (game mode)
                Selected_Feature=st.selectbox('Ask a question from a list, create your query or select a winner:', Feature_Options, 
                                                       index=0, 
                                                       key='selected_feature', help=None)
                    
                ## SHOW ELEMENTS - QUESTIONS MODE
                if Selected_Feature=='Ask a Question':
                    ## Game mode id
                    st.session_state['init_data']['token_type']=0

                    ## Text - Questions mode
                    st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Select a Question from the list.</h3>", unsafe_allow_html=True)
                    
                    ## SelectBox - Select question
                    Selected_Question=st.selectbox('Suggested questions:', st.session_state['init_data']['feature_questions'], 
                                                       index=0,
                                                       key='Selected_Question', help=None)
                    st.session_state['init_data']['selected_question']=Selected_Question  # Save Info
                    
                    ## Current question index
                    if Selected_Question not in st.session_state['init_data']['feature_questions']:
                        Selected_Question=st.session_state['init_data']['feature_questions'][0]
                    
                    st.session_state['init_data']['questions_index']=st.session_state['init_data']['feature_questions'].index(Selected_Question)
                       
                    ## Text - Show current question
                    st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>Current Question: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+Selected_Question+"</h3>", unsafe_allow_html=True)
                    
                    ## Button - Use current question
                    Check_Question = st.button('USE THIS QUESTION', key='Check_Question')
                    st.session_state['init_data']['button_question']=Check_Question  # Save Info
                    
                    ## Check current question
                    if st.session_state['init_data']['show_results']:
                        st.session_state['init_data']['show_results']=False
                        
                    else:
                        if Check_Question:
                            if Selected_Question=='Are you bald?':
                                st.session_state['init_data']['current_querys']=['A picture of a male person','A picture of a female person',
                                                                            'A picture of a bald man','A picture of a haired man', 
                                                                            'A picture of a bald person','A picture of a person']
                                st.session_state['init_data']['function_predict']=Predict_bald
                                
                            elif Selected_Question=='Do you have BLACK HAIR?':
                                st.session_state['init_data']['current_querys']=['A picture of a black-haired person',
                                                                            'A picture of a tawny-haired person',
                                                                            'A picture of a blond-haired person',
                                                                            'A picture of a gray-haired person',
                                                                            'A picture of a red-haired person',
                                                                            'A picture of a green-haired person',
                                                                            'A picture of a blue-haired person',
                                                                            'A picture of a bald-head person']
                                st.session_state['init_data']['function_predict']=Predict_0_vs_all

                            elif Selected_Question=='Do you have BROWN HAIR?':
                                st.session_state['init_data']['current_querys']=['A picture of a tawny-haired person',
                                                                            'A picture of a black-haired person',
                                                                            'A picture of a blond-haired person',
                                                                            'A picture of a gray-haired person',
                                                                            'A picture of a red-haired person',
                                                                            'A picture of a green-haired person',
                                                                            'A picture of a blue-haired person',
                                                                            'A picture of a bald-head person']
                                st.session_state['init_data']['function_predict']=Predict_0_vs_all

                            elif Selected_Question=='Do you have BLOND HAIR?':
                                st.session_state['init_data']['current_querys']=['A picture of a blond-haired person',
                                                                            'A picture of a tawny-haired person',
                                                                            'A picture of a black-haired person',
                                                                            'A picture of a gray-haired person',
                                                                            'A picture of a red-haired person',
                                                                            'A picture of a green-haired person',
                                                                            'A picture of a blue-haired person',
                                                                            'A picture of a bald-head person']
                                st.session_state['init_data']['function_predict']=Predict_0_vs_all
                                
                            elif Selected_Question=='Do you have RED HAIR?':
                                st.session_state['init_data']['current_querys']=['A picture of a red-haired person',
                                                                            'A picture of a tawny-haired person',
                                                                            'A picture of a blond-haired person',
                                                                            'A picture of a gray-haired person',
                                                                            'A picture of a black-haired person',
                                                                            'A picture of a green-haired person',
                                                                            'A picture of a blue-haired person',
                                                                            'A picture of a bald-head person']
                                st.session_state['init_data']['function_predict']=Predict_0_vs_all
                                
                            elif Selected_Question=='Do you have GRAY HAIR?':
                                st.session_state['init_data']['current_querys']=['A picture of a gray-haired person',
                                                                            'A picture of a tawny-haired person',
                                                                            'A picture of a blond-haired person',
                                                                            'A picture of a black-haired person',
                                                                            'A picture of a red-haired person',
                                                                            'A picture of a green-haired person',
                                                                            'A picture of a blue-haired person',
                                                                            'A picture of a bald-head person']
                                st.session_state['init_data']['function_predict']=Predict_0_vs_all
                                
                           
                            elif  not st.session_state['init_data']['show_results']:
                                st.session_state['init_data']['current_querys']=[st.session_state['init_data']['querys_list_yes'][st.session_state['init_data']['questions_index']],
                                                                                st.session_state['init_data']['querys_list_no'][st.session_state['init_data']['questions_index']]]
                                st.session_state['init_data']['function_predict']=Predict_0_vs_1
                        
                            CLIP_Process()
                            st.session_state['init_data']['function_predict']()
                            st.session_state['init_data']['show_results']=True
                            
                ## SHOW ELEMENTS - 1 QUERY MOD
                if Selected_Feature=='Create your own query':              
                    
                    ## Game mode id
                    st.session_state['init_data']['token_type']=-1

                    ## Text - Query mode
                    st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Write your own query and press the button.</h3>", unsafe_allow_html=True)
                    
                    ## TextInput - Select query
                    User_Input = st.text_input('It is recommended to use a text like: "A picture of a ... person" or "A picture of a person ..." (CLIP will check -> "Your query"  vs  "A picture of a person" )', 'A picture of a person', key='User_Input', help=None)
                    st.session_state['init_data']['user_input']=User_Input  # Save Info

                    ## Text - Show current query
                    st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>Current Query: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+User_Input+"</h3>", unsafe_allow_html=True)
                    
                    ## Button - Use current query
                    Check_Query = st.button('USE MY OWN QUERY', key='Check_Query')
                    st.session_state['init_data']['button_query1']=Check_Query  # Save Info
                    
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
                                st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Your query must be different of 'A picture of a person'.</h3>", unsafe_allow_html=True)
                        
                ## SHOW ELEMENTS - 2 QUERYS MODE
                if Selected_Feature=='Create your own 2 querys':
                    
                    ## Game mode id
                    st.session_state['init_data']['token_type']=-2

                    ## Text - Querys mode
                    st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Write your own querys by introducing 2 opposite descriptions.</h3>", unsafe_allow_html=True)
                    
                    ## SelectBox - Select querys
                    User_Input_Querys1 = st.text_input('Write your "True" query:', 'A picture of a person',
                                                                key='User_Input_Querys1', help=None)
                    User_Input_Querys2 = st.text_input('Write your "False" query:', 'A picture of a person',
                                                                key='User_Input_Querys2', help=None)
                    st.session_state['init_data']['user_input_querys1']=User_Input_Querys1  # Save Info
                    st.session_state['init_data']['user_input_querys2']=User_Input_Querys2  # Save Info
                                     
                    ## Text - Show current querys
                    st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>Current Querys: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+User_Input_Querys1+' vs '+User_Input_Querys2+"</h3>", unsafe_allow_html=True)
                    
                    ## Button - Use current querys
                    Check_Querys = st.button('USE MY OWN QUERYS', key='Check_Querys')
                    st.session_state['init_data']['button_query2']=Check_Querys  # Save Info
                    
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
                                st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Your two own querys must be different.</h3>", unsafe_allow_html=True)

                ## SHOW ELEMENTS - WINNER MODE
                if Selected_Feature=='Select a Winner': 
                    
                    ## Game mode id
                    st.session_state['init_data']['token_type']=-3

                    ## Text - Winner mode
                    st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Select a Winner picture name.</h3>", unsafe_allow_html=True)
                    
                    ## SelectBox - Select winner
                    # st.session_state['init_data']['winner_options']=['Winner not selected']
                    # st.session_state['init_data']['winner_options'].extend(st.session_state['init_data']['current_image_names'])
                    
                    # if st.session_state['init_data']['selected_winner'] not in st.session_state['init_data']['winner_options']:
                        # st.write(st.session_state['init_data']['selected_winner'])
                        # st.write(st.session_state['init_data']['winner_options'])
                        
                    Selected_Winner=st.selectbox('If you are inspired, Select a Winner image directly:', st.session_state['init_data']['winner_options'],
                                                    index=0, key='Selected_Winner', help=None)
                    st.session_state['init_data']['selected_winner']=Selected_Winner  # Save Info
                    
                    ## Text - Show current winner
                    st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>Current Winner: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+Selected_Winner+"</h3>", unsafe_allow_html=True)
                    
                    ## Button - Use current winner
                    Check_Winner = st.button('CHECK THIS WINNER', key='Check_Winner')
                    st.session_state['init_data']['button_winner']=Check_Winner  # Save Info
                                                        
                    ## Check current winner
                    if st.session_state['init_data']['show_results']:
                        st.session_state['init_data']['show_results']=False
                    else:
                        if Check_Winner:
                            if Selected_Winner in st.session_state['init_data']['current_image_names']:
                                st.session_state['init_data']['selected_winner_index']=np.where(Selected_Winner==st.session_state['init_data']['current_image_names'])[0]
                                st.session_state['init_data']['image_current_predictions']=np.zeros(st.session_state['init_data']['n_images'])
                                st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['selected_winner_index']]=1    
                                st.session_state['init_data']['show_results']=True
                                
                                # Delete Winner elements   
                                # del st.session_state['Selected_Winner']                                    
                            else:
                                st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Your must select a not discarded picture.</h3>", unsafe_allow_html=True)


                ## ACTIONS SHOWING RESULTS
                if st.session_state['init_data']['show_results']:
                
                    ## Continue game
                    if not np.sum(st.session_state['init_data']['current_images_discarted']==0)==1:
                        if st.session_state['init_data']['images_selected']: 
                            st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>4. Press the button to continue.</h2>", unsafe_allow_html=True)
                        else:
                            st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>2. Press the button to continue.</h2>", unsafe_allow_html=True)
                        
                        ## Button - Next query
                        Next_Query=st.button('NEXT QUERY', key='Next_Query')

                    ## Show current results
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
                        if not st.session_state['init_data']['selected_winner']==st.session_state['init_data']['current_image_names'][st.session_state['init_data']['current_winner_index']]:
                            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>The winner picture is not:</h3><h3 style='text-align:left; float:center; color:red; margin:0px;'>"+st.session_state['init_data']['selected_winner']+"</h3>", unsafe_allow_html=True)

         
        ## CREATE IMAGES TO SHOW
        Showed_Images=Show_images()
        st.session_state['init_data']['Showed_image_names']=st.session_state['init_data']['current_image_names']


       ## APPLY DISCARDING
        if st.session_state['init_data']['show_results']:        
            st.session_state['init_data']['previous_discarding_images_number']=st.session_state['init_data']['n_images']
            Image_discarding()
                       
            ## penalty - game not finished                                                       
            if st.session_state['init_data']['n_images']>1:
                st.session_state['init_data']['award']=st.session_state['init_data']['award']-st.session_state['init_data']['n_images']
            
            ## penalty - "select winner" option used
            if st.session_state['init_data']['token_type']==-3:   
                st.session_state['init_data']['award']=st.session_state['init_data']['award']-1-(st.session_state['init_data']['N_images']-st.session_state['init_data']['previous_discarding_images_number'])

            ## penalty - no image is discarted
            if st.session_state['init_data']['previous_discarding_images_number']==st.session_state['init_data']['n_images']:   
                st.session_state['init_data']['award']=st.session_state['init_data']['award']-5


        ## SHOW FINAL RESULTS
        if st.session_state['init_data']['finished_game']:
            st.session_state['init_data']['reload_game']=True

        else:
            ## CHECK FINISHED GAME 
            if np.sum(st.session_state['init_data']['current_images_discarted']==0)==1 and not st.session_state['init_data']['finished_game']:
                st.session_state['init_data']['finished_game']=True
                st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>You found the Winner picture:</h1><h1 style='text-align:left; float:left; color:green; margin:0px;'>"+st.session_state['init_data']['current_image_names'][st.session_state['init_data']['current_winner_index']]+"</h1>", unsafe_allow_html=True)
                Finsih_Game = st.button('FINISH GAME', key='Finsih_Game')


        ## SHOW CURRENT IMAGES
        st.image(Showed_Images, use_column_width=False, caption=st.session_state['init_data']['Showed_image_names'])
        
        del Showed_Images

        ## RELOAD GAME
        if st.session_state['init_data']['reload_game']:
            Load_Data(st.session_state['init_data']['N_images']) 
            
            
    ## SHOW EXTRA INFO
    # Show_Info() 
 
## --------------- CACHE FUCTION ---------------
@st.cache(ttl=12*3600)
def Load_CLIP():
	  return clip.load("ViT-B/32", device="cpu", jit=False)

## --------------- STREAMLIT APP ---------------

st.set_page_config(
    layout="wide",
    page_icon='Logo DIMAI.png',
    page_title='QuienEsQuien',
    initial_sidebar_state="collapsed"
)

## CLEAR RESOURCES
Main_Program()
gc.collect()
# caching.clear_cache()
# torch.cuda.empty_cache()

    
## SHOW INFO (cpu, memeory)
# st.sidebar.write(psutil.cpu_percent()) ## show info (cpu, memeory)
# st.sidebar.write(psutil.virtual_memory()) ## show info (cpu, memeory)
