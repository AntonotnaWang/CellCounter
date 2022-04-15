import numpy as np
from torch import device as torch_device
from torch import load as torch_load
from torch.nn import AdaptiveMaxPool2d as torch_nn_AdaptiveMaxPool2d
from torch import from_numpy
from torchvision import models
from sklearn.cluster import DBSCAN
from PIL import Image

class FeatureMapExtractor(object):
    def __init__(self, model, target_layers, device=torch_device('cpu')):
        self.device = device
        self.model = model.to(self.device)
        self.target_layers = target_layers # the target layers can be list of layer names or indexes
        self.fmap_pool = dict()
        self.handlers = []
        def forward_hook(key):
            def forward_hook_(module, input_im, output_im):
                self.fmap_pool[key] = output_im.detach().clone().cpu().numpy()
            return forward_hook_
        
        for idx, (name, module) in enumerate(self.model.named_modules()):
            if name in self.target_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                print("hook layer "+str(name))
            elif idx in self.target_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(idx)))
                print("hook layer "+str(name))
    
    def reset(self):
        self.fmap_pool = dict()
        self.handlers = []
    
    def __call__(self, x):
        return self.model(x.to(self.device))

def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
        
    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    
    im_as_arr=im_as_arr.transpose(1,2,0)

    return im_as_arr

def use_DL_model_to_preprocess_img(raw_img_input, which_layers=[4], pre_trained_model_name="vgg19", device=torch_device('cpu')):
    if pre_trained_model_name=="vgg19":
        para_path='DL_model_para/vgg19_part_para.pth'
        model_para = torch_load(para_path)
        input_model=models.vgg19(pretrained=False)
        input_model.load_state_dict(model_para, strict=False)
        
    feature_map_extractor=FeatureMapExtractor(input_model, which_layers, device)
    
    # raw_img_input: np.array [H, W, Channel]
    raw_img_input_tensor=raw_img_input.transpose(2,0,1)
    raw_img_input_tensor=raw_img_input_tensor.reshape(1,raw_img_input_tensor.shape[0],
                                                      raw_img_input_tensor.shape[1],
                                                      raw_img_input_tensor.shape[2])
    raw_img_input_tensor=from_numpy(raw_img_input_tensor).float().to(device)
    
    # forward pass
    _ = feature_map_extractor(raw_img_input_tensor)
    
    # get the target feature map
    feature_map_as_arr = {}
    for idx, which_layer in enumerate(which_layers):
        feature_map_as_arr[which_layer]=feature_map_extractor.fmap_pool[which_layer][0]
        if type(feature_map_as_arr[which_layer])!=np.ndarray:
            feature_map_as_arr[which_layer]=np.array(feature_map_as_arr[which_layer].cpu().detach().numpy(), dtype=np.float)
        feature_map_as_arr[which_layer]=feature_map_as_arr[which_layer].transpose(1,2,0)
    
    return feature_map_as_arr

def resize_img(img_input, resize_shape):
    img_output = Image.fromarray(np.uint8(img_input))
    img_output = img_output.resize(resize_shape)
    img_output = np.array(img_output, dtype=np.float)
    return img_output

def resize_img_pool(img_input, resize_shape):
    img_input_tensor=from_numpy(img_input.reshape(1,1,img_input.shape[0],img_input.shape[1])).float()
    pool=torch_nn_AdaptiveMaxPool2d(resize_shape)
    img_output_tensor=pool(img_input_tensor)
    img_output=np.array(img_output_tensor.cpu().detach().numpy(), dtype=np.float)[0][0]
    return img_output

if __name__ == '__main__':
    device = torch_device("cpu")
    input_model_part=models.vgg19(pretrained=True)
    print("model arch:")
    print(input_model_part)
    which_layers = [4]
    feature_map_extractor=FeatureMapExtractor(input_model_part, which_layers, device)
    feature_map_extractor.reset()
    