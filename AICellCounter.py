# -*- coding: utf-8 -*-
"""
@author: AntonWang
"""

from func.counter import get_01_segment_of_raw_img_without_sample_counts,\
    count_with_01_segment,find_count_map_and_get_cell_instances, \
        learn_gmms, find_count_map, DBSCAN_mask_output
from func.DL_processing import use_DL_model_to_preprocess_img, resize_img, resize_img_pool

import numpy as np
import cv2 as cv
import tkinter
from tkinter import ttk
from tkinter import LEFT, TOP, BOTTOM, X, FLAT, RAISED, RIGHT
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import messagebox
import pickle
from PIL import Image
import copy
import os
import webbrowser
    
_NAME = 'AICellCounter'
_SHOW_NAME_RAW_IMG_WITH_COUNT = 'Raw image with count'
_SHOW_NAME_COUNT_RESULT = 'Count result'
_PADDING_PIX = 3
_MAXSIZE_FOR_ONE_PROCESS = 700
_MIN_MASK_SIZE=10 # min mask size of one target cell
_DL_MODEL_LAYER=5 # which DL model to use to pre-process the input img, or you can input the name of the target layer
_DL_MODEL_NAME_LIST = ["vgg19"]
_COLOR_OF_TARGET_ON_COUNT_IMG = [255, 255, 255]
_RESULT_ADJUST_SATA = ["No adjustment", "Delete noises", "Add missing cells"]
_RESULT_ADJUST_RADIUS=2
_RESULT_ADJUST_TARGET_COLOR_FOR_SHOW = [255, 255, 255]
_RESULT_ADJUST_TARGET_COLOR_NOISES = (255, 0, 255)
_RESULT_ADJUST_TARGET_COLOR_MISSING_CELLS = (0, 255, 255)

def save_obj(obj, name):
    filepath = os.path.split(name)[0]
    
    if filepath!="":
        if not os.path.exists(filepath):
            try:
                os.makedirs(filepath)
            except:
                pass
                
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    filepath = os.path.split(name)[0]
    
    if filepath!="":
        if not os.path.exists(filepath):
            try:
                os.makedirs(filepath)
            except:
                pass
                
    with open(name, 'rb') as f:
        return pickle.load(f)

class Data_Load_and_Process(object):
    def __init__(self):
        self.green_img_filename = None
        self.blue_img_filename = None
        self.red_img_filename = None
        self.count_img_filename = None
        
        self.green_raw_img = None
        self.blue_raw_img = None
        self.red_raw_img = None
        self.input_raw_img = None
        self.input_count_img = None
        
        self.mask_output = None # 01 mask, 1 means cell and 0 means background
        self.single_cells = None # mask showing cell instances, 0 means bg and ints (>0) means diff cells
        self.num_of_cells = 0
        
        self.foreground_gmm = None
        self.background_gmm = None
        
        self.show_count_on_rgb = None
        self.result_adjust_stat = _RESULT_ADJUST_SATA[0] #["No adjustment", "Delete noises", "Add missing cells"]
        self.result_adjust_mouse_begin = False
        self.fake_cell_value_record = []
        self.missing_cell_value = 0
        
        self.color_of_the_target = _COLOR_OF_TARGET_ON_COUNT_IMG # cell label color: white
        self.foreground_gmm_load_file_path = "GMM/foreground_gmm.pkl"
        self.background_gmm_load_file_path = "GMM/background_gmm.pkl"
        self.foreground_gmm_save_file_path = "GMM/foreground_gmm.pkl"
        self.background_gmm_save_file_path = "GMM/background_gmm.pkl"
        self.img_max_size_for_one_process = _MAXSIZE_FOR_ONE_PROCESS
        self.min_mask_size = _MIN_MASK_SIZE
        self.DL_model_name = _DL_MODEL_NAME_LIST[0]
        self.which_DL_layer = _DL_MODEL_LAYER
        self.need_train_based_on_existing_gmm = False
        self.result_adjust_target_color_for_show = _RESULT_ADJUST_TARGET_COLOR_FOR_SHOW # cell label color: white
        self.result_adjust_draw = {'color':(0, 0, 0), 'val':0} # control the current drawing color and drawing value
        self.result_adjust_color_of_noises = _RESULT_ADJUST_TARGET_COLOR_NOISES
        self.result_adjust_color_of_missing_cells = _RESULT_ADJUST_TARGET_COLOR_MISSING_CELLS
        self.result_adjust_radius = _RESULT_ADJUST_RADIUS
        
        self.toplevel_win_list = []
        self.msg = None
    
    def load_raw_img(self):    
        try:
            self.green_raw_img=Image.open(self.green_img_filename)
            self.green_raw_img=np.array(self.green_raw_img, dtype=np.float)
        except:
            self.green_raw_img=0
        
        try:
            self.blue_raw_img=Image.open(self.blue_img_filename)
            self.blue_raw_img=np.array(self.blue_raw_img, dtype=np.float)
        except:
            self.blue_raw_img = 0
        
        try:
            self.red_raw_img=Image.open(self.red_img_filename)
            self.red_raw_img=np.array(self.red_raw_img, dtype=np.float)
        except:
            self.red_raw_img = 0
        
        try:
            self.input_raw_img=self.green_raw_img+self.blue_raw_img+self.red_raw_img
            if np.sum(self.input_raw_img)<=0:
                messagebox.showinfo(title='Hello', message='No image detected.')
                return False
            return True
        except:
            messagebox.showinfo(title='Hello', message='Image loading fails.')
            return False
    
    
    def load_count_img(self):
        try:
            self.input_count_img=Image.open(self.count_img_filename)
            self.input_count_img=np.array(self.input_count_img, dtype=np.float)
            self.input_count_img=find_count_map_and_get_cell_instances(self.input_count_img,
                                                                       color_of_the_target = self.color_of_the_target)
            return True
        except:
            messagebox.showinfo(title='Hello', message='Count image loading fails.')
            return False
    
    
    def load_GMMs(self):
        try:
            self.foreground_gmm=load_obj(self.foreground_gmm_load_file_path)
            self.background_gmm=load_obj(self.background_gmm_load_file_path)
            return True
        except:
            messagebox.showinfo(title='Hello', message='GMM loading fails.')
            return False
    
    def save_GMMs(self):
        try:
            save_obj(self.foreground_gmm, self.foreground_gmm_save_file_path)
            save_obj(self.background_gmm, self.background_gmm_save_file_path)
            messagebox.showinfo(title='Hello', message='Models are saved!')
            return True
        except:
            messagebox.showinfo(title='Hello', message='GMM saving fails.')
            return False
    
    def show_count_on_raw_img(self):
        self.show_count_on_rgb=copy.deepcopy(self.input_raw_img)
        self.show_count_on_rgb[np.where(self.single_cells>0)[0],
                               np.where(self.single_cells>0)[1],:]=self.result_adjust_target_color_for_show
    
    def recount_cell_instances(self):
        if len(self.fake_cell_value_record)>=1:
            self.fake_cell_value_record=np.unique(
                np.array(self.fake_cell_value_record))
            self.fake_cell_value_record=\
                self.fake_cell_value_record[self.fake_cell_value_record>0]
            print("Fake cells are: "+str(self.fake_cell_value_record))
        
            # delete those fake cells in single_cells
            for value in self.fake_cell_value_record:
                self.single_cells[
                    np.where(self.single_cells==value)]=0
            
            self.fake_cell_value_record=[]
            
        #self.single_cells, _, _ = DBSCAN_mask_output(self.single_cells,
        #                                             threshold=self.min_mask_size,
        #                                             dbscan_eps=1, dbscan_min_samples=1)
        self.num_of_cells = np.max([0, len(np.unique(self.single_cells))-1])
    
    def count(self):
        if self.load_raw_img() and self.load_GMMs():
            feature_map_dict=use_DL_model_to_preprocess_img(raw_img_input=self.input_raw_img,
                                                            which_layers=[self.which_DL_layer],
                                                            pre_trained_model_name=self.DL_model_name)
            feature_map_i = feature_map_dict[self.which_DL_layer]
            
            self.mask_output=get_01_segment_of_raw_img_without_sample_counts(feature_map_i,
                                                                             self.foreground_gmm,
                                                                             self.background_gmm,
                                                                             img_max_size_for_one_process=self.img_max_size_for_one_process)
            self.mask_output=resize_img(self.mask_output,
                                        (self.input_raw_img.shape[1], self.input_raw_img.shape[0]))
            
            self.single_cells, _, _=count_with_01_segment(self.mask_output, min_mask_size=self.min_mask_size)
            
            self.num_of_cells = np.max([0, len(np.unique(self.single_cells))-1])
            
            self.show_count_on_raw_img()
            
            return True
        else:
            messagebox.showinfo(title='Hello', message='Something wrong with counting.')
            return False
    
    def train(self):
        if self.load_raw_img() and self.load_count_img():
            feature_map_dict=use_DL_model_to_preprocess_img(raw_img_input=self.input_raw_img,
                                                            which_layers=[self.which_DL_layer],
                                                            pre_trained_model_name=self.DL_model_name)
            feature_map_i = feature_map_dict[self.which_DL_layer]
            self.input_count_img=resize_img_pool(self.input_count_img,
                                                 (feature_map_i.shape[0], feature_map_i.shape[1]))
            
            self.mask_output, self.foreground_gmm, self.background_gmm=\
                learn_gmms(feature_map_i,
                           mask_input=self.input_count_img,
                           need_train_based_on_existing_gmm=self.need_train_based_on_existing_gmm,
                           foreground_gmm=self.foreground_gmm,
                           background_gmm=self.background_gmm)
            self.mask_output=resize_img(self.mask_output,
                                        (self.input_raw_img.shape[1], self.input_raw_img.shape[0]))
            
            self.single_cells, _, _=count_with_01_segment(self.mask_output, min_mask_size=self.min_mask_size)
            
            self.num_of_cells = np.max([0, len(np.unique(self.single_cells))-1])
            
            #self.save_GMMs()
            
            self.show_count_on_raw_img()
            
            return True
        else:
            messagebox.showinfo(title='Hello', message='Something wrong with training.')
            return False
    
    def retrain(self):
        if self.load_raw_img():
            feature_map_dict=use_DL_model_to_preprocess_img(raw_img_input=self.input_raw_img,
                                                            which_layers=[self.which_DL_layer],
                                                            pre_trained_model_name=self.DL_model_name)
            feature_map_i = feature_map_dict[self.which_DL_layer]
            
            current_count_img = np.array(self.single_cells>0, dtype=float)
            current_count_img = resize_img_pool(current_count_img,
                                                (feature_map_i.shape[0], feature_map_i.shape[1]))
            
            self.mask_output, self.foreground_gmm, self.background_gmm=\
                learn_gmms(feature_map_i,
                           mask_input=current_count_img,
                           need_train_based_on_existing_gmm=self.need_train_based_on_existing_gmm,
                           foreground_gmm=self.foreground_gmm,
                           background_gmm=self.background_gmm)
            self.mask_output=resize_img(self.mask_output,
                                        (self.input_raw_img.shape[1], self.input_raw_img.shape[0]))
            
            self.single_cells, _, _=count_with_01_segment(self.mask_output, min_mask_size=self.min_mask_size)
            
            self.num_of_cells = np.max([0, len(np.unique(self.single_cells))-1])
            
            #self.save_GMMs()
            
            self.show_count_on_raw_img()
            
            return True
        else:
            messagebox.showinfo(title='Hello', message='Something wrong with re-training.')
            return False
    
    def reset(self):
        self.green_img_filename = None
        self.blue_img_filename = None
        self.red_img_filename = None
        self.count_img_filename = None
        
        self.green_raw_img = None
        self.blue_raw_img = None
        self.red_raw_img = None
        self.input_raw_img = None
        self.input_count_img = None
        
        self.mask_output = None # 01 mask, 1 means cell and 0 means background
        self.single_cells = None # mask showing cell instances, 0 means bg and ints (>0) means diff cells
        self.num_of_cells = 0
        
        self.foreground_gmm = None
        self.background_gmm = None
        
        self.show_count_on_rgb = None
        self.result_adjust_stat = _RESULT_ADJUST_SATA[0] #["No adjustment", "Delete noises", "Add missing cells"]
        self.result_adjust_mouse_begin = False
        self.fake_cell_value_record = []
        self.missing_cell_value = 0
        
        self.color_of_the_target = _COLOR_OF_TARGET_ON_COUNT_IMG # cell label color: white
        self.foreground_gmm_load_file_path = "GMM/foreground_gmm.pkl"
        self.background_gmm_load_file_path = "GMM/background_gmm.pkl"
        self.foreground_gmm_save_file_path = "GMM/foreground_gmm.pkl"
        self.background_gmm_save_file_path = "GMM/background_gmm.pkl"
        self.img_max_size_for_one_process = _MAXSIZE_FOR_ONE_PROCESS
        self.min_mask_size = _MIN_MASK_SIZE
        self.DL_model_name = _DL_MODEL_NAME_LIST[0]
        self.which_DL_layer = _DL_MODEL_LAYER
        self.need_train_based_on_existing_gmm = False
        self.result_adjust_target_color_for_show = _RESULT_ADJUST_TARGET_COLOR_FOR_SHOW # cell label color: white
        self.result_adjust_draw = {'color':(0, 0, 0), 'val':0} # control the current drawing color and drawing value
        self.result_adjust_color_of_noises = _RESULT_ADJUST_TARGET_COLOR_NOISES
        self.result_adjust_color_of_missing_cells = _RESULT_ADJUST_TARGET_COLOR_MISSING_CELLS
        self.result_adjust_radius = _RESULT_ADJUST_RADIUS


def main():
    data = Data_Load_and_Process()
    
    root = tkinter.Tk()
    
    # Set the size of the window
    # win_len = 50
    # win_wid = 100
    # root.geometry(str(win_wid) + 'x' + str(win_len))
    
    root.title(_NAME)
    
    # msg
    label_text = "Welcome to "+_NAME
    data.msg = tkinter.StringVar()
    data.msg.set(label_text)
    
    # toolbar
    toolbar = tkinter.Frame(root)
    count_button = tkinter.Button(toolbar, text="Count pipeline", relief=RAISED,
                          command=lambda: COUNT_COMPLETE_FUNC(data, main_frame))
    count_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    train_button = tkinter.Button(toolbar, text="Train pipeline", relief=RAISED,
                          command=lambda: TRAIN_COMPLETE_FUNC(data, main_frame))
    train_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    reset_button = tkinter.Button(toolbar, text="Reset", relief=RAISED,
                          command=lambda: RESET(data))
    reset_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    toolbar.pack(side=TOP, fill=X)
    
    # print msg
    main_frame = ttk.Frame(root, padding=_PADDING_PIX)
    label = ttk.Label(main_frame, textvariable=data.msg)
    label.pack()
    main_frame.pack(side=TOP, fill=X)
    
    # copyright
    copyright_frame = ttk.Frame(root, padding=_PADDING_PIX)
    label = ttk.Label(copyright_frame,
                      text='Copyright © Andong Wang. All rights reserved.',
                      font=('Arial', 7))
    label.pack()
    copyright_frame.pack(side=BOTTOM, fill=X)
    
    # menu
    # The default is for menus to be "tear-off" -- they can be dragged
    # off the menubar.  Use whichever style best suits your GUI.
    root.option_add('*tearOff', False)

    # Make the menu bar
    menubar = tkinter.Menu(root)
    root['menu'] = menubar

    # Make the pull-down menu's on the menu bar.
    # data_menu: data loading 
    data_menu = tkinter.Menu(menubar)
    menubar.add_cascade(menu=data_menu, label='Data')
    
    # operation_menu: train and count
    operation_menu = tkinter.Menu(menubar)
    menubar.add_cascade(menu=operation_menu, label='Operation')
    
    # setting_menu: set parameters
    setting_menu = tkinter.Menu(menubar)
    menubar.add_cascade(menu=setting_menu, label='Setting')
    
    # miscellaneous
    miscellaneous_menu = tkinter.Menu(menubar)
    menubar.add_cascade(menu=miscellaneous_menu, label='Miscellaneous')
    
    # Make menu items for each menu on the menu bar.
    # Bind callbacks using lambda, as we have seen elsewhere,
    # but this time with a    command=...   optional argument supplied.
    
    # data
    data_menu.add_command(label='Load image (red channel)',
                          command=lambda: LOAD_RED_IMG(data))
    
    data_menu.add_command(label='Load image (green channel)',
                          command=lambda: LOAD_GREEN_IMG(data))

    data_menu.add_command(label='Load image (blue channel)',
                          command=lambda: LOAD_BLUE_IMG(data))
    
    data_menu.add_command(label='Load count image (as groundtruth)',
                          command=lambda: LOAD_COUNT_IMG(data))
    
    # operation
    operation_menu.add_command(label='Count',
                               command=lambda: COUNT(data, main_frame))
    operation_menu.add_command(label='Train',
                               command=lambda: TRAIN(data, main_frame))
    operation_menu.add_command(label='Re-train',
                               command=lambda: RETRAIN(data, main_frame))
    operation_menu.add_command(label='Save results',
                               command=lambda: SAVE_RESULTS(data))
    operation_menu.add_command(label='Reset',
                               command=lambda: RESET(data))
    
    # setting
    setting_menu.add_command(label='Settings (count)',
                             command=lambda: COUNT_SETTINGS(data, main_frame))
    setting_menu.add_command(label='Settings (train and retrain)',
                             command=lambda: TRAIN_SETTINGS(data, main_frame))
    
    # miscellaneous
    miscellaneous_menu.add_command(label='Help',
                                   command=lambda: HELP(data))
    miscellaneous_menu.add_command(label='About',
                                   command=lambda: ABOUT(data))
    
    root.mainloop()

def LOAD_RED_IMG(data_load_and_process):
    red_img_filename = askopenfilename()
    data_load_and_process.red_img_filename = red_img_filename
    data_load_and_process.msg.set('Load red-channel image ' + red_img_filename)

def LOAD_GREEN_IMG(data_load_and_process):
    green_img_filename = askopenfilename()
    data_load_and_process.green_img_filename = green_img_filename
    data_load_and_process.msg.set('Load green-channel image ' + green_img_filename)

def LOAD_BLUE_IMG(data_load_and_process):
    blue_img_filename = askopenfilename()
    data_load_and_process.blue_img_filename = blue_img_filename
    data_load_and_process.msg.set('Load blue-channel image ' + blue_img_filename)

def LOAD_COUNT_IMG(data_load_and_process):
    count_img_filename = askopenfilename()
    data_load_and_process.count_img_filename = count_img_filename
    data_load_and_process.msg.set('Load count image ' + count_img_filename)

def CLOSE_ALL_TOPLEVEL_WINS(data_load_and_process):
    for k in range(len(data_load_and_process.toplevel_win_list)):
        try:
            data_load_and_process.toplevel_win_list[k].destroy()
        except:
            pass
    data_load_and_process.top_level_win_list=[]

def SET_WINDOW_TO_SHOW_IMG(data_load_and_process):
    CLOSE_WIN_FOR_IMG_SHOW(data_load_and_process)
    cv.namedWindow(_SHOW_NAME_RAW_IMG_WITH_COUNT,  flags=cv.WINDOW_NORMAL)
    cv.namedWindow(_SHOW_NAME_COUNT_RESULT, flags=cv.WINDOW_NORMAL)
    
    def ONMOUSE_ADJUST_RESULT(event, x, y, flags, param):
        if data_load_and_process.result_adjust_stat==_RESULT_ADJUST_SATA[1]:
            if event == cv.EVENT_LBUTTONDOWN:
                data_load_and_process.result_adjust_mouse_begin=True
                print('drawing fake cells; loc: ('+str(y)+' ,'+str(x)+'); cell value: '+\
                      str(data_load_and_process.single_cells[y, x]))
                cv.circle(data_load_and_process.show_count_on_rgb, (x, y),
                          data_load_and_process.result_adjust_radius,
                          data_load_and_process.result_adjust_draw['color'], -1)
                data_load_and_process.fake_cell_value_record.append(
                    data_load_and_process.single_cells[y, x])
                SHOW_IMG(data_load_and_process)
        
            elif event == cv.EVENT_MOUSEMOVE:
                if data_load_and_process.result_adjust_mouse_begin==True:
                    print('drawing fake cells; loc: ('+str(y)+' ,'+str(x)+'); cell value: '+\
                          str(data_load_and_process.single_cells[y, x]))
                    cv.circle(data_load_and_process.show_count_on_rgb, (x, y),
                              data_load_and_process.result_adjust_radius,
                              data_load_and_process.result_adjust_draw['color'], -1)
                    data_load_and_process.fake_cell_value_record.append(
                        data_load_and_process.single_cells[y, x])
                    SHOW_IMG(data_load_and_process)
                    
            elif event == cv.EVENT_LBUTTONUP:
                if data_load_and_process.result_adjust_mouse_begin==True:
                    data_load_and_process.result_adjust_mouse_begin=False
                    print('drawing fake cells; loc: ('+str(y)+' ,'+str(x)+'); cell value: '+\
                          str(data_load_and_process.single_cells[y, x]))
                    cv.circle(data_load_and_process.show_count_on_rgb, (x, y),
                              data_load_and_process.result_adjust_radius,
                              data_load_and_process.result_adjust_draw['color'], -1)
                    data_load_and_process.fake_cell_value_record.append(
                        data_load_and_process.single_cells[y, x])
                    SHOW_IMG(data_load_and_process)
                    
        elif data_load_and_process.result_adjust_stat==_RESULT_ADJUST_SATA[2]:
            if event == cv.EVENT_LBUTTONDOWN:
                data_load_and_process.result_adjust_mouse_begin=True
                data_load_and_process.missing_cell_value = np.max(data_load_and_process.single_cells)+1
                print('drawing missing cells; loc: ('+str(y)+' ,'+str(x)+'); cell value: '+\
                      str(data_load_and_process.missing_cell_value))
                cv.circle(data_load_and_process.show_count_on_rgb, (x, y),
                          data_load_and_process.result_adjust_radius,
                          data_load_and_process.result_adjust_draw['color'], -1)
                cv.circle(data_load_and_process.single_cells, (x, y),
                          data_load_and_process.result_adjust_radius,
                          data_load_and_process.missing_cell_value, -1)
                SHOW_IMG(data_load_and_process)
            
            elif event == cv.EVENT_MOUSEMOVE:
                if data_load_and_process.result_adjust_mouse_begin==True:
                    print('drawing missing cells; loc: ('+str(y)+' ,'+str(x)+'); cell value: '+\
                          str(data_load_and_process.missing_cell_value))
                    cv.circle(data_load_and_process.show_count_on_rgb, (x, y),
                              data_load_and_process.result_adjust_radius,
                              data_load_and_process.result_adjust_draw['color'], -1)
                    cv.circle(data_load_and_process.single_cells, (x, y),
                              data_load_and_process.result_adjust_radius,
                              data_load_and_process.missing_cell_value, -1)
                    SHOW_IMG(data_load_and_process)
                    
            elif event == cv.EVENT_LBUTTONUP:
                if data_load_and_process.result_adjust_mouse_begin==True:
                    data_load_and_process.result_adjust_mouse_begin=False
                    print('drawing missing cells; loc: ('+str(y)+' ,'+str(x)+'); cell value: '+\
                          str(data_load_and_process.missing_cell_value))
                    cv.circle(data_load_and_process.show_count_on_rgb, (x, y),
                              data_load_and_process.result_adjust_radius,
                              data_load_and_process.result_adjust_draw['color'], -1)
                    cv.circle(data_load_and_process.single_cells, (x, y),
                              data_load_and_process.result_adjust_radius,
                              data_load_and_process.missing_cell_value, -1)
                    SHOW_IMG(data_load_and_process)
    
    cv.setMouseCallback(_SHOW_NAME_RAW_IMG_WITH_COUNT, ONMOUSE_ADJUST_RESULT)

"""
def ADD_CTRL_TO_WINDOW(data_load_and_process):
    switch = '0 : Delete noises \n1 : Add missing cells'
    cv.createTrackbar(switch, _SHOW_NAME_COUNT_RESULT, 0, 1,
                      lambda: TARCKBAR_ADJUST_RESULT(data_load_and_process))

def TARCKBAR_ADJUST_RESULT(data_load_and_process):
    pass
"""

def SHOW_IMG(data_load_and_process):
    cv.imshow(_SHOW_NAME_RAW_IMG_WITH_COUNT,
              np.array([data_load_and_process.show_count_on_rgb[:,:,2],
                        data_load_and_process.show_count_on_rgb[:,:,1],
                        data_load_and_process.show_count_on_rgb[:,:,0]], dtype=np.uint8).transpose(1,2,0))
    cv.imshow(_SHOW_NAME_COUNT_RESULT,
              data_load_and_process.single_cells)

def SET_WINDOW_AND_SHOW_IMG(data_load_and_process):
    CLOSE_WIN_FOR_IMG_SHOW(data_load_and_process)
    SET_WINDOW_TO_SHOW_IMG(data_load_and_process)
    SHOW_IMG(data_load_and_process)

def CLOSE_WIN_FOR_IMG_SHOW(data_load_and_process):
    try:
        cv.destroyAllWindows()
    except:
        pass

def FRAME_ADJUST_RESULT(data_load_and_process, root):
    frame = tkinter.Frame(root)#, bg='#DCDCDC')
    
    adjust_result_delete_noises_button = tkinter.Button(frame, text="Delete noises", relief=RAISED,
                                          command=lambda: SET_ADJUST_RESULT_STAT_DELETE_NOISES(data_load_and_process))
    adjust_result_delete_noises_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    adjust_result_add_missing_cells_button = tkinter.Button(frame, text="Add missing cells", relief=RAISED,
                                          command=lambda: SET_ADJUST_RESULT_STAT_ADD_CELLS(data_load_and_process))
    adjust_result_add_missing_cells_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    adjust_result_update_button = tkinter.Button(frame, text="Update adjustment", relief=RAISED,
                                          command=lambda: SET_ADJUST_RESULT_STAT_UPDATE_ADJUST(data_load_and_process))
    adjust_result_update_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    adjust_result_save_button = tkinter.Button(frame, text="Save", relief=RAISED,
                                          command=lambda: SAVE_RESULTS(data_load_and_process))
    adjust_result_save_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    adjust_result_show_button = tkinter.Button(frame, text="Show", relief=RAISED,
                                          command=lambda: SET_WINDOW_AND_SHOW_IMG(data_load_and_process))
    adjust_result_show_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    adjust_result_exit_button = tkinter.Button(frame, text="Exit", relief=RAISED,
                                          command=lambda: CLOSE_WIN_FOR_IMG_SHOW(data_load_and_process))
    adjust_result_exit_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    frame.pack(side=TOP, fill=X)

def SET_ADJUST_RESULT_STAT_DELETE_NOISES(data_load_and_process):
    if data_load_and_process.result_adjust_stat==_RESULT_ADJUST_SATA[2]:
        SET_ADJUST_RESULT_STAT_UPDATE_ADJUST(data_load_and_process)
    data_load_and_process.result_adjust_draw['color'] = data_load_and_process.result_adjust_color_of_noises
    data_load_and_process.result_adjust_draw['val'] = 1
    data_load_and_process.result_adjust_stat=_RESULT_ADJUST_SATA[1]
    messagebox.showinfo(title='Hello',
                        message='Please mark the noises with your mouse on '+_SHOW_NAME_RAW_IMG_WITH_COUNT+' and click \'Update adjustment\' when you finish marking.')

def SET_ADJUST_RESULT_STAT_ADD_CELLS(data_load_and_process):
    if data_load_and_process.result_adjust_stat==_RESULT_ADJUST_SATA[1]:
        SET_ADJUST_RESULT_STAT_UPDATE_ADJUST(data_load_and_process)
    data_load_and_process.result_adjust_draw['color'] = data_load_and_process.result_adjust_color_of_missing_cells
    data_load_and_process.result_adjust_draw['val'] = 2
    data_load_and_process.result_adjust_stat=_RESULT_ADJUST_SATA[2]
    messagebox.showinfo(title='Hello',
                        message='Please mark the missing cells with your mouse on '+_SHOW_NAME_RAW_IMG_WITH_COUNT+' and click \'Update adjustment\' when you finish marking.')

def SET_ADJUST_RESULT_STAT_UPDATE_ADJUST(data_load_and_process):
    data_load_and_process.recount_cell_instances()
            
    SHOW_IMG(data_load_and_process)
    UPDATE_CELL_NUM_AND_SHOW_IN_MSG(data_load_and_process)
    
    data_load_and_process.result_adjust_draw['val'] = 0
    data_load_and_process.result_adjust_stat=_RESULT_ADJUST_SATA[0]
    messagebox.showinfo(title='Hello', message='Finish adjustment')

def FRAME_SAVE_GMM_TO_GIVEN_PATH(data_load_and_process, root):
    frame = tkinter.Frame(root)#, bg='#DCDCDC')
    
    options = {}
    options['defaultextension'] = ".pkl"
    options['filetypes'] = [('pkl files', '.pkl'), ('all files', '.*')]
    def CHANGE_SAVE_PATH(data_load_and_process, frame, filepath_for_show, options):
        chosen_filepath = asksaveasfilename(**options)
        filepath_for_show.set(chosen_filepath)
        frame.update()
        
    frame_foreground = tkinter.Frame(frame)#, bg='#DCDCDC')
    foreground_label = ttk.Label(frame_foreground, text="Foreground (cell) model")
    foreground_label.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    foreground_gmm_filepath = tkinter.StringVar()
    foreground_gmm_filepath.set(data_load_and_process.foreground_gmm_save_file_path)
    foreground_entry = tkinter.Entry(frame_foreground,textvariable=foreground_gmm_filepath)
    foreground_entry.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    foreground_button = tkinter.Button(frame_foreground, text="Change path", relief=RAISED,
                                       command=lambda: CHANGE_SAVE_PATH(data_load_and_process,
                                                                        frame_foreground,
                                                                        foreground_gmm_filepath,
                                                                        options))
    foreground_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_foreground.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    frame_background = tkinter.Frame(frame)#, bg='#DCDCDC')
    background_label = ttk.Label(frame_background, text="Background model")
    background_label.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    background_gmm_filepath = tkinter.StringVar()
    background_gmm_filepath.set(data_load_and_process.background_gmm_save_file_path)
    background_entry = tkinter.Entry(frame_background,textvariable=background_gmm_filepath)
    background_entry.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    background_button = tkinter.Button(frame_background, text="Change path", relief=RAISED,
                                       command=lambda: CHANGE_SAVE_PATH(data_load_and_process,
                                                                        frame_background,
                                                                        background_gmm_filepath,
                                                                        options))
    background_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_background.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    def SAVE_GMMS(data_load_and_process, foreground_gmm_filepath, background_gmm_filepath):
        data_load_and_process.foreground_gmm_save_file_path = foreground_gmm_filepath.get()
        data_load_and_process.background_gmm_save_file_path = background_gmm_filepath.get()
        data_load_and_process.save_GMMs()
        
    save_gmm_button = tkinter.Button(frame, text="Save the model", relief=RAISED,
                                       command=lambda: SAVE_GMMS(data_load_and_process,
                                                                 foreground_gmm_filepath,
                                                                 background_gmm_filepath))
    save_gmm_button.pack(side=RIGHT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    frame.pack(side=TOP, fill=X)

def FRAME_LOAD_GMM_BY_GIVEN_PATH(data_load_and_process, root):
    frame = tkinter.Frame(root)#, bg='#DCDCDC')
    
    options = {}
    options['defaultextension'] = ".pkl"
    options['filetypes'] = [('pkl files', '.pkl'), ('all files', '.*')]
    def CHANGE_LOAD_PATH(data_load_and_process, frame, filepath_for_show, options):
        chosen_filepath = askopenfilename(**options)
        filepath_for_show.set(chosen_filepath)
        frame.update()
        
    frame_foreground = tkinter.Frame(frame)#, bg='#DCDCDC')
    foreground_label = ttk.Label(frame_foreground, text="Foreground (cell) model")
    foreground_label.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    foreground_gmm_filepath = tkinter.StringVar()
    foreground_gmm_filepath.set(data_load_and_process.foreground_gmm_save_file_path)
    foreground_entry = tkinter.Entry(frame_foreground,textvariable=foreground_gmm_filepath)
    foreground_entry.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    foreground_button = tkinter.Button(frame_foreground, text="Change path", relief=RAISED,
                                       command=lambda: CHANGE_LOAD_PATH(data_load_and_process,
                                                                        frame_foreground,
                                                                        foreground_gmm_filepath,
                                                                        options))
    foreground_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_foreground.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    frame_background = tkinter.Frame(frame)#, bg='#DCDCDC')
    background_label = ttk.Label(frame_background, text="Background model")
    background_label.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    background_gmm_filepath = tkinter.StringVar()
    background_gmm_filepath.set(data_load_and_process.background_gmm_save_file_path)
    background_entry = tkinter.Entry(frame_background,textvariable=background_gmm_filepath)
    background_entry.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    background_button = tkinter.Button(frame_background, text="Change path", relief=RAISED,
                                       command=lambda: CHANGE_LOAD_PATH(data_load_and_process,
                                                                        frame_background,
                                                                        background_gmm_filepath,
                                                                        options))
    background_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_background.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    frame.pack(side=TOP, fill=X)
    
    return foreground_gmm_filepath, background_gmm_filepath

def UPDATE_CELL_NUM_AND_SHOW_IN_MSG(data_load_and_process):
    data_load_and_process.num_of_cells = np.max([0, len(np.unique(data_load_and_process.single_cells))-1])
    data_load_and_process.msg.set("There are "+str(data_load_and_process.num_of_cells)+" cells.")

def TRAIN(data_load_and_process, win):
    CLOSE_ALL_TOPLEVEL_WINS(data_load_and_process)
    CLOSE_WIN_FOR_IMG_SHOW(data_load_and_process)
    
    data_load_and_process.msg.set("Training")
    win.update()
    if data_load_and_process.train():
        UPDATE_CELL_NUM_AND_SHOW_IN_MSG(data_load_and_process)
        win.update()
        
        # show results
        SET_WINDOW_TO_SHOW_IMG(data_load_and_process)
        SHOW_IMG(data_load_and_process)
        
        # save the trained GMMs
        root_save_gmms = tkinter.Toplevel()
        root_save_gmms.title("Save the trained model")
        
        data_load_and_process.toplevel_win_list.append(root_save_gmms)
        
        FRAME_SAVE_GMM_TO_GIVEN_PATH(data_load_and_process, root_save_gmms)
        root_save_gmms.mainloop()
        
    else:
        data_load_and_process.msg.set("Something wrong with training")
        win.update()
    
def COUNT(data_load_and_process, win):
    CLOSE_ALL_TOPLEVEL_WINS(data_load_and_process)
    CLOSE_WIN_FOR_IMG_SHOW(data_load_and_process)
    
    data_load_and_process.msg.set("Counting")
    win.update()
    if data_load_and_process.count():
        UPDATE_CELL_NUM_AND_SHOW_IN_MSG(data_load_and_process)
        win.update()
        
        # show results
        SET_WINDOW_TO_SHOW_IMG(data_load_and_process)
        SHOW_IMG(data_load_and_process)
        
        # control panel for result adjustment
        root_result_adjust = tkinter.Toplevel()
        root_result_adjust.title("Result adjustment")
        
        data_load_and_process.toplevel_win_list.append(root_result_adjust)
        
        FRAME_ADJUST_RESULT(data_load_and_process, root_result_adjust)
        root_result_adjust.mainloop()
        
    else:
        data_load_and_process.msg.set("Something wrong with counting")
        win.update()

def RETRAIN(data_load_and_process, win):
    CLOSE_ALL_TOPLEVEL_WINS(data_load_and_process)
    CLOSE_WIN_FOR_IMG_SHOW(data_load_and_process)
    
    data_load_and_process.msg.set("Re-training")
    win.update()
    if data_load_and_process.retrain():
        UPDATE_CELL_NUM_AND_SHOW_IN_MSG(data_load_and_process)
        win.update()
        
        # show results
        SET_WINDOW_TO_SHOW_IMG(data_load_and_process)
        SHOW_IMG(data_load_and_process)
        
        # save the trained GMMs
        root_save_gmms = tkinter.Toplevel()
        root_save_gmms.title("Save the re-trained model")
        
        data_load_and_process.toplevel_win_list.append(root_save_gmms)

        FRAME_SAVE_GMM_TO_GIVEN_PATH(data_load_and_process, root_save_gmms)
        root_save_gmms.mainloop()
        
    else:
        data_load_and_process.msg.set("Something wrong with re-training")
        win.update()

def FRAME_COMMON_SETTINGS(data_load_and_process, root):
    frame = tkinter.Frame(root)#, bg='#DCDCDC')
    
    min_mask_size_frame = tkinter.Frame(frame)
    min_mask_size_label = ttk.Label(min_mask_size_frame, text="Min size for one cell")
    min_mask_size_label.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    min_mask_size = tkinter.StringVar()
    min_mask_size.set(data_load_and_process.min_mask_size)
    min_mask_size_entry = tkinter.Entry(min_mask_size_frame,textvariable=min_mask_size, width=5)
    min_mask_size_entry.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    min_mask_size_frame.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    space_label = ttk.Label(frame, text=" ")
    space_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    #sep = ttk.Separator(frame, orient='horizontal')
    #sep.pack(padx=_PADDING_PIX, fill=X)
    
    dl_model_label = ttk.Label(frame, text="Deep learning model for pre-processing")
    dl_model_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    dl_model_frame = tkinter.Frame(frame)
    which_dl_model_label = ttk.Label(dl_model_frame, text="Which model")
    which_dl_model_label.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    which_dl_model = tkinter.StringVar()
    which_dl_model_comboxlist=ttk.Combobox(dl_model_frame,textvariable=which_dl_model, width=10)
    which_dl_model_comboxlist["values"]=_DL_MODEL_NAME_LIST
    #which_dl_model_comboxlist.current(0)
    which_dl_model.set(data_load_and_process.DL_model_name)
    #dl_model_comboxlist.bind("<<ComboboxSelected>>", FUNC)
    which_dl_model_comboxlist.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    which_layer_label = ttk.Label(dl_model_frame, text="Which layer")
    which_layer_label.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    which_layer = tkinter.StringVar()
    which_layer.set(data_load_and_process.which_DL_layer )
    which_layer_entry = tkinter.Entry(dl_model_frame,textvariable=which_layer, width=5)
    which_layer_entry.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    dl_model_frame.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    space_label = ttk.Label(frame, text=" ")
    space_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    color_of_target_on_countimg_label = ttk.Label(frame, text="Color of the target cells in the input labels")
    color_of_target_on_countimg_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    color_of_target_on_countimg_frame = tkinter.Frame(frame)
    #R
    R_frame = tkinter.Frame(color_of_target_on_countimg_frame)
    R_label = ttk.Label(R_frame, text="R")
    R_label.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    R_scale = tkinter.Scale(R_frame, from_=0, to=255, orient="horizontal")
    R_scale.set(data_load_and_process.color_of_the_target[0])
    R_scale.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    R_frame.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    #G
    G_frame = tkinter.Frame(color_of_target_on_countimg_frame)
    G_label = ttk.Label(G_frame, text="G")
    G_label.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    G_scale = tkinter.Scale(G_frame, from_=0, to=255, orient="horizontal")
    G_scale.set(data_load_and_process.color_of_the_target[1])
    G_scale.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    G_frame.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    #B
    B_frame = tkinter.Frame(color_of_target_on_countimg_frame)
    B_label = ttk.Label(B_frame, text="B")
    B_label.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    B_scale = tkinter.Scale(B_frame, from_=0, to=255, orient="horizontal")
    B_scale.set(data_load_and_process.color_of_the_target[2])
    B_scale.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    B_frame.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    color_of_target_on_countimg_frame.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    frame.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    return min_mask_size, which_dl_model, which_layer, R_scale, G_scale, B_scale

def FRAME_TRAIN_SETTINGS(data_load_and_process, root_train_settings):
    min_mask_size, which_dl_model, which_layer, R_scale, G_scale, B_scale = \
        FRAME_COMMON_SETTINGS(data_load_and_process, root_train_settings)
    
    #
    space_label = ttk.Label(root_train_settings, text=" ")
    space_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    #sep = ttk.Separator(root_train_settings, orient='horizontal')
    #sep.pack(padx=_PADDING_PIX, fill=X)
    # 
    
    #
    frame_need_train_based_on_existing_gmm = tkinter.Frame(root_train_settings)
    
    need_train_based_on_existing_gmm_radio_label = ttk.Label(frame_need_train_based_on_existing_gmm,
                                                             text="Train based on existing model?")
    need_train_based_on_existing_gmm_radio_label.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    need_train_based_on_existing_gmm_radio_list = tkinter.IntVar()
    yes_train_based_on_existing_gmm_radiobutton = \
        tkinter.Radiobutton(frame_need_train_based_on_existing_gmm,
                            variable=need_train_based_on_existing_gmm_radio_list,
                            value=True, text="Yes")
    no_train_based_on_existing_gmm_radiobutton = \
        tkinter.Radiobutton(frame_need_train_based_on_existing_gmm,
                            variable=need_train_based_on_existing_gmm_radio_list,
                            value=False, text="No")
    need_train_based_on_existing_gmm_radio_list.set(data_load_and_process.need_train_based_on_existing_gmm)
    yes_train_based_on_existing_gmm_radiobutton.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    no_train_based_on_existing_gmm_radiobutton.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_need_train_based_on_existing_gmm.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    #
    
    #
    space_label = ttk.Label(root_train_settings, text=" ")
    space_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    #
    
    #
    load_gmm_label = ttk.Label(root_train_settings, text="Set the model loading path")
    load_gmm_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    foreground_gmm_filepath, background_gmm_filepath = \
        FRAME_LOAD_GMM_BY_GIVEN_PATH(data_load_and_process, root_train_settings)
    #
    
    #
    space_label = ttk.Label(root_train_settings, text=" ")
    space_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    #
    
    def APPLY_SETTINGS(data_load_and_process,
                       min_mask_size,
                       which_dl_model,
                       which_layer,
                       R_scale,
                       G_scale,
                       B_scale,
                       need_train_based_on_existing_gmm_radio_list,
                       foreground_gmm_filepath,
                       background_gmm_filepath,
                       win):
        data_load_and_process.min_mask_size = int(min_mask_size.get())
        data_load_and_process.DL_model_name = which_dl_model.get()
        data_load_and_process.which_DL_layer = int(which_layer.get())
        data_load_and_process.color_of_the_target = [int(R_scale.get()),
                                                     int(G_scale.get()),
                                                     int(B_scale.get())]
        data_load_and_process.need_train_based_on_existing_gmm = \
            (need_train_based_on_existing_gmm_radio_list.get()==1)
        data_load_and_process.foreground_gmm_load_file_path = \
            foreground_gmm_filepath.get()
        data_load_and_process.background_gmm_load_file_path = \
            background_gmm_filepath.get()
        win.update()
        messagebox.showinfo(title='Hello',
                            message='Changes are applied.')
        
    apply_button = tkinter.Button(root_train_settings, text="Apply setting changes", relief=RAISED,
                                  command=lambda: APPLY_SETTINGS(data_load_and_process,
                                                     min_mask_size,
                                                     which_dl_model,
                                                     which_layer,
                                                     R_scale,
                                                     G_scale,
                                                     B_scale,
                                                     need_train_based_on_existing_gmm_radio_list,
                                                     foreground_gmm_filepath,
                                                     background_gmm_filepath,
                                                     root_train_settings))
    apply_button.pack(side=RIGHT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
def TRAIN_SETTINGS(data_load_and_process, win):
    data_load_and_process.msg.set("Settings for training and re-training")
    win.update()
    
    root_train_settings = tkinter.Toplevel()
    root_train_settings.title("Settings for training and re-training")
    
    data_load_and_process.toplevel_win_list.append(root_train_settings)
    
    FRAME_TRAIN_SETTINGS(data_load_and_process, root_train_settings)
    
    root_train_settings.mainloop()

def FRAME_COUNT_SETTINGS(data_load_and_process, root_count_settings):
    min_mask_size, which_dl_model, which_layer, R_scale, G_scale, B_scale = \
        FRAME_COMMON_SETTINGS(data_load_and_process, root_count_settings)
    
    #
    space_label = ttk.Label(root_count_settings, text=" ")
    space_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    #sep = ttk.Separator(root_count_settings, orient='horizontal')
    #sep.pack(padx=_PADDING_PIX, fill=X)
    #
    
    #
    load_gmm_label = ttk.Label(root_count_settings, text="Set the model loading path")
    load_gmm_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    foreground_gmm_filepath, background_gmm_filepath = \
        FRAME_LOAD_GMM_BY_GIVEN_PATH(data_load_and_process, root_count_settings)
    #
    
    #
    space_label = ttk.Label(root_count_settings, text=" ")
    space_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    #
    
    def APPLY_SETTINGS(data_load_and_process,
                       min_mask_size,
                       which_dl_model,
                       which_layer,
                       R_scale,
                       G_scale,
                       B_scale,
                       foreground_gmm_filepath,
                       background_gmm_filepath,
                       win):
        data_load_and_process.min_mask_size = int(min_mask_size.get())
        data_load_and_process.DL_model_name = which_dl_model.get()
        data_load_and_process.which_DL_layer = int(which_layer.get())
        data_load_and_process.color_of_the_target = [int(R_scale.get()),
                                                     int(G_scale.get()),
                                                     int(B_scale.get())]
        data_load_and_process.foreground_gmm_load_file_path = \
            foreground_gmm_filepath.get()
        data_load_and_process.background_gmm_load_file_path = \
            background_gmm_filepath.get()
        win.update()
        messagebox.showinfo(title='Hello',
                            message='Changes are applied.')
        
    apply_button = tkinter.Button(root_count_settings, text="Apply setting changes", relief=RAISED,
                                  command=lambda: APPLY_SETTINGS(data_load_and_process,
                                                     min_mask_size,
                                                     which_dl_model,
                                                     which_layer,
                                                     R_scale,
                                                     G_scale,
                                                     B_scale,
                                                     foreground_gmm_filepath,
                                                     background_gmm_filepath,
                                                     root_count_settings))
    apply_button.pack(side=RIGHT, padx=_PADDING_PIX, pady=_PADDING_PIX)

def COUNT_SETTINGS(data_load_and_process, win):
    data_load_and_process.msg.set("Settings for cell counting")
    win.update()
    
    root_count_settings = tkinter.Toplevel()
    root_count_settings.title("Settings for cell counting")
    
    data_load_and_process.toplevel_win_list.append(root_count_settings)
    
    FRAME_COUNT_SETTINGS(data_load_and_process, root_count_settings)
    
    root_count_settings.mainloop()

def RESET(data_load_and_process):
    CLOSE_ALL_TOPLEVEL_WINS(data_load_and_process)
    CLOSE_WIN_FOR_IMG_SHOW(data_load_and_process)
    data_load_and_process.reset()
    data_load_and_process.msg.set("Reset all")

def FRAME_LOAD_RGB_IMGS(data_load_and_process, win):
    R_button = tkinter.Button(win, text="Red channel", relief=RAISED,
                                  command=lambda: LOAD_RED_IMG(data_load_and_process))
    R_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    G_button = tkinter.Button(win, text="Green channel", relief=RAISED,
                                  command=lambda: LOAD_GREEN_IMG(data_load_and_process))
    G_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    B_button = tkinter.Button(win, text="Blue channel", relief=RAISED,
                                  command=lambda: LOAD_BLUE_IMG(data_load_and_process))
    B_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)

def TRAIN_COMPLETE_FUNC(data_load_and_process, win):
    CLOSE_ALL_TOPLEVEL_WINS(data_load_and_process)
    CLOSE_WIN_FOR_IMG_SHOW(data_load_and_process)
    
    data_load_and_process.msg.set("Train pipeline")
    win.update()
    
    root_train_pipeline = tkinter.Toplevel()
    root_train_pipeline.title("Train pipeline")
    label = ttk.Label(root_train_pipeline, text="Load images")
    label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_load_imgs = tkinter.Frame(root_train_pipeline)
    FRAME_LOAD_RGB_IMGS(data_load_and_process, frame_load_imgs)
    load_count_img_button = tkinter.Button(frame_load_imgs, text="Count image", relief=RAISED,
                                  command=lambda: LOAD_COUNT_IMG(data_load_and_process))
    load_count_img_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_load_imgs.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    #
    space_label = ttk.Label(root_train_pipeline, text=" ")
    space_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    sep = ttk.Separator(root_train_pipeline, orient='horizontal')
    sep.pack(padx=_PADDING_PIX, fill=X)
    #
    
    label = ttk.Label(root_train_pipeline, text="Execution")
    label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_exec = tkinter.Frame(root_train_pipeline)
    train_button = tkinter.Button(frame_exec, text="Train", relief=RAISED,
                                  command=lambda: TRAIN(data_load_and_process, win))
    train_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    save_button = tkinter.Button(frame_exec, text="Save results", relief=RAISED,
                                  command=lambda: SAVE_RESULTS(data_load_and_process))
    save_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_exec.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    #
    space_label = ttk.Label(root_train_pipeline, text=" ")
    space_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    sep = ttk.Separator(root_train_pipeline, orient='horizontal')
    sep.pack(padx=_PADDING_PIX, fill=X)
    #
    
    FRAME_TRAIN_SETTINGS(data_load_and_process, root_train_pipeline)
    
def COUNT_COMPLETE_FUNC(data_load_and_process, win):
    CLOSE_ALL_TOPLEVEL_WINS(data_load_and_process)
    CLOSE_WIN_FOR_IMG_SHOW(data_load_and_process)
    
    data_load_and_process.msg.set("Count pipeline")
    win.update()
    
    root_count_pipeline = tkinter.Toplevel()
    root_count_pipeline.title("Count pipeline")
    label = ttk.Label(root_count_pipeline, text="Load images")
    label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_load_imgs = tkinter.Frame(root_count_pipeline)
    FRAME_LOAD_RGB_IMGS(data_load_and_process, frame_load_imgs)
    frame_load_imgs.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    #
    space_label = ttk.Label(root_count_pipeline, text=" ")
    space_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    sep = ttk.Separator(root_count_pipeline, orient='horizontal')
    sep.pack(padx=_PADDING_PIX, fill=X)
    #
    
    label = ttk.Label(root_count_pipeline, text="Execution")
    label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_exec = tkinter.Frame(root_count_pipeline)
    count_button = tkinter.Button(frame_exec, text="Count", relief=RAISED,
                                  command=lambda: COUNT(data_load_and_process, win))
    count_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    save_button = tkinter.Button(frame_exec, text="Save results", relief=RAISED,
                                  command=lambda: SAVE_RESULTS(data_load_and_process))
    save_button.pack(side=LEFT, padx=_PADDING_PIX, pady=_PADDING_PIX)
    frame_exec.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    #
    space_label = ttk.Label(root_count_pipeline, text=" ")
    space_label.pack(side=TOP, padx=_PADDING_PIX, pady=_PADDING_PIX)
    
    sep = ttk.Separator(root_count_pipeline, orient='horizontal')
    sep.pack(padx=_PADDING_PIX, fill=X)
    #
    
    FRAME_COUNT_SETTINGS(data_load_and_process, root_count_pipeline)
    
def SAVE_RESULTS(data_load_and_process):
    options = {}
    options['defaultextension'] = ".tif"
    options['filetypes'] = [('all files', '.*')]
    options['initialfile'] = "Count_result.tif"
    try:
        save_path = asksaveasfilename(**options)
        save_mask = Image.fromarray(np.array(data_load_and_process.single_cells>0, dtype=np.uint8)*255)
        save_mask.save(save_path);
        messagebox.showinfo("Hello", "The result has been saved.")
    except:
        messagebox.showinfo("Hello", "Something wrong with result saving.")

def HELP(data_load_and_process):
    webbrowser.open_new("https://github.com/AntonotnaWang/CellCounter")
    
def ABOUT(data_load_and_process):
    messagebox.showinfo("Hello", "This software is designed and implemented by Andong Wang.\n"+\
                        "It comes out from a collaborative project with Gao Lab at Southern Medical University in Guangzhou.\n"+\
                            "Please contact wangad@connect.hku.hk if you have any questions.")

# ----------------------------------------------------------------------
# Calls  main  to start the ball rolling.
# ----------------------------------------------------------------------
main()