import numpy as np
#from PIL import Image
from sklearn.cluster import DBSCAN
from func.grab_gmm import grabcut
from skimage.measure import label, regionprops
import math

IMG_MAX_SIZE_FOR_ONE_PROCESS=1500
MIN_MASK_SIZE=10
GMM_COMPONENTS_DEFAULT = 5

def find_count_map(count_im_arr, color_of_the_target = [255, 255, 255]):
	# INPUT
    # count_im_arr: a map which contains the label of the target; np.array; shape: [nx, ny, 3]
    # color_of_the_target: shape: [3, ], indicating the RGB of the target
    
    # OUTPUT
    # count_map: shape: [nx, ny], where 0 indicates background and 1 indicates the target
    
	count_loc=np.where(np.logical_and(np.logical_and(count_im_arr[:,:,0]==color_of_the_target[0],
                                                  count_im_arr[:,:,1]==color_of_the_target[1]),
                                   count_im_arr[:,:,2]==color_of_the_target[2]))
	count_loc=np.array([count_loc[0],count_loc[1]]).transpose(1,0)
	count_map=np.zeros((count_im_arr.shape[0], count_im_arr.shape[1]), dtype=np.float)
	for i in np.arange(count_loc.shape[0]):
		count_map[count_loc[i,0], count_loc[i,1]]=1
	return count_map

def find_count_map_and_get_cell_instances(count_im_arr, color_of_the_target = [255, 255, 255]):
    # INPUT
    # count_im_arr: a map which contains the label of the target; np.array; shape: [nx, ny, 3]
    # color_of_the_target: shape: [3, ], indicating the RGB of the target
    
    # OUTPUT
    # count_map: shape: [nx, ny], where 0 indicates background and 1 indicates the target
    
    count_loc=np.where(np.logical_and(np.logical_and(count_im_arr[:,:,0]==color_of_the_target[0],
                                                  count_im_arr[:,:,1]==color_of_the_target[1]),
                                   count_im_arr[:,:,2]==color_of_the_target[2]))
    count_loc=np.array([count_loc[0],count_loc[1]]).transpose(1,0)
    
    clustering = DBSCAN(eps=1, min_samples=2).fit(count_loc)
    labels=clustering.labels_
    count_loc=count_loc[labels>=0,:]
    labels=labels[labels>=0]
    labels=labels+1
    label_unique_value=np.unique(labels)
    
    count_map=np.zeros((count_im_arr.shape[0], count_im_arr.shape[1]), dtype=np.float)
    
    for idx, value in enumerate(label_unique_value):
        temp_locs=count_loc[labels==value,:]

        max_x=np.max(temp_locs[:,0])
        min_x=np.min(temp_locs[:,0])
        max_y=np.max(temp_locs[:,1])
        min_y=np.min(temp_locs[:,1])

        x_len=max_x-min_x
        y_len=max_y-min_y

        max_x_new=np.clip(max_x+np.int(x_len), 0, count_im_arr.shape[0]-1)
        min_x_new=np.clip(min_x-np.int(x_len), 0, count_im_arr.shape[0]-1)
        max_y_new=np.clip(max_y+np.int(y_len), 0, count_im_arr.shape[1]-1)
        min_y_new=np.clip(min_y-np.int(y_len), 0, count_im_arr.shape[1]-1)
        
        count_map[min_x_new:max_x_new, min_y_new:max_y_new]=1
    
    return count_map

def get_01_segment_of_raw_img_without_sample_counts(raw_img,
                                                    foreground_gmm,
                                                    background_gmm,
                                                    img_max_size_for_one_process=IMG_MAX_SIZE_FOR_ONE_PROCESS):
    raw_img_shape=raw_img.shape
    
    if raw_img_shape[0]<=img_max_size_for_one_process and \
        raw_img_shape[1]<=img_max_size_for_one_process:
        GRABCUT = grabcut(img_input=raw_img,
          foreground_gmm=foreground_gmm,
          background_gmm=background_gmm,
          mask_input=None)
        mask_output = GRABCUT.mask
    else:
        raw_img_shape_min = np.min([raw_img_shape[0], raw_img_shape[1]])
        crop_cube_size=np.array([np.min([img_max_size_for_one_process, raw_img_shape_min]),
                                 np.min([img_max_size_for_one_process, raw_img_shape_min])])
        mask_output, _, _=crop_and_count(raw_img,crop_cube_size,
                                         grabcut,foreground_gmm,background_gmm)
    return mask_output

def count_with_01_segment(mask_output, min_mask_size=MIN_MASK_SIZE):
    #use BDSCAN
    single_cells, clustering_labels_unique, clustering_labels_counts=\
        DBSCAN_mask_output(mask_output, threshold=min_mask_size, dbscan_eps=1, dbscan_min_samples=1)
    
    return single_cells, clustering_labels_unique, clustering_labels_counts

def learn_gmms(raw_img, mask_input, need_train_based_on_existing_gmm=False, foreground_gmm=None, background_gmm=None, gmm_components_kn_input=GMM_COMPONENTS_DEFAULT):
    GRABCUT = grabcut(img_input=raw_img, mask_input=mask_input,
    foreground_gmm=foreground_gmm, background_gmm=background_gmm,
    gmm_components_kn_input=gmm_components_kn_input, need_train_based_on_existing_gmm=need_train_based_on_existing_gmm)
    mask_output = GRABCUT.mask
    foreground_gmm=GRABCUT.foreground_gmm
    background_gmm=GRABCUT.background_gmm
    
    return mask_output, foreground_gmm, background_gmm
    
def crop_and_count(raw_img,
                   crop_cube_size,
                   grabcut_func,
                   foreground_gmm,
                   background_gmm,
                   mask_input=None, need_shuffle=False):
    raw_img_shape=raw_img.shape
    mask_output=np.zeros((raw_img_shape[0], raw_img_shape[1]))
    
    arr_i = np.arange(0, raw_img_shape[0], crop_cube_size[0])
    arr_j = np.arange(0, raw_img_shape[1], crop_cube_size[1])
    
    if need_shuffle:
        np.random.shuffle(arr_i)
        np.random.shuffle(arr_j)
    
    for i in arr_i:
        for j in arr_j:
            if i+crop_cube_size[0]<=raw_img_shape[0]:
                x_start=i
                x_end=i+crop_cube_size[0]
            else:
                x_start=raw_img_shape[0]-crop_cube_size[0]
                x_end=raw_img_shape[0]
            
            if j+crop_cube_size[1]<=raw_img_shape[1]:
                y_start=j
                y_end=j+crop_cube_size[1]
            else:
                y_start=raw_img_shape[1]-crop_cube_size[1]
                y_end=raw_img_shape[1]
            
            raw_img_crop=raw_img[x_start:x_end, y_start:y_end, :]
            
            if mask_input is not None:
                mask_input_crop=mask_input[x_start:x_end, y_start:y_end]
                GRABCUT = grabcut_func(img_input=raw_img_crop,
                  foreground_gmm=foreground_gmm,
                  background_gmm=background_gmm,
                  mask_input=mask_input_crop)
                foreground_gmm=GRABCUT.foreground_gmm
                background_gmm=GRABCUT.background_gmm
                mask_output_crop = GRABCUT.mask
            else:
                GRABCUT = grabcut_func(img_input=raw_img_crop,
                  foreground_gmm=foreground_gmm,
                  background_gmm=background_gmm,
                  mask_input=None)
                mask_output_crop = GRABCUT.mask
                
            mask_output_temp=np.zeros((raw_img_shape[0], raw_img_shape[1]))
            mask_output_temp[x_start:x_end, y_start:y_end]=mask_output_crop
            mask_output[x_start:x_end, y_start:y_end]=\
            mask_output[x_start:x_end, y_start:y_end]+\
            mask_output_temp[x_start:x_end, y_start:y_end]
                
            del GRABCUT
    return mask_output, foreground_gmm, background_gmm

def DBSCAN_mask_output(mask_output, threshold=1, dbscan_eps=1, dbscan_min_samples=1):
    cell_locs=np.where(mask_output>0)
    cell_locs_x=cell_locs[0]
    cell_locs_y=cell_locs[1]
    cell_locs_len=cell_locs[0].shape[0]
    cell_locs_reshape=np.concatenate((cell_locs[0].reshape(cell_locs_len,1),
                                      cell_locs[1].reshape(cell_locs_len,1)),axis=1)
    clustering = DBSCAN(eps=dbscan_eps,
                        min_samples=dbscan_min_samples,
                        metric='euclidean').fit(cell_locs_reshape)
    clustering_labels=clustering.labels_
    clustering_labels_unique,clustering_labels_counts=np.unique(clustering_labels, return_counts=True)
    
    clustering_labels_counts=clustering_labels_counts[clustering_labels_unique>0]
    clustering_labels_unique=clustering_labels_unique[clustering_labels_unique>0] # delete background and noise
    
    clustering_labels_unique=clustering_labels_unique[np.where(clustering_labels_counts>threshold)]
    clustering_labels_counts=clustering_labels_counts[np.where(clustering_labels_counts>threshold)]
    
    single_cells=np.zeros(mask_output.shape)
    for i in range(0, len(clustering_labels_unique)):
        temp_label=clustering_labels_unique[i]
        temp_label_locs=np.where(clustering_labels==temp_label)
        single_cells[cell_locs_x[temp_label_locs],cell_locs_y[temp_label_locs]]= \
        clustering_labels_unique[i]
    
    return single_cells, clustering_labels_unique, clustering_labels_counts

def delete_those_do_not_have_the_target_roundness(mask, threshold_lower = 0, threshold_upper = 10):
    if threshold_lower>=threshold_upper:
        threshold_lower = 0
        threshold_upper = 10
    
    mask = label(mask)
    mask_props = regionprops(mask)
    roundness_dict = {}
    
    for idx, mask_prop in enumerate(mask_props):
        roundness = (4 * math.pi * mask_prop['area']) / (mask_prop['perimeter']**2)
        roundness_dict[idx+1] = roundness
    
    for idx in roundness_dict.keys():
        if roundness_dict[idx]>threshold_upper or roundness_dict[idx]<threshold_lower:
            mask[mask==idx] = 0
    
    return mask