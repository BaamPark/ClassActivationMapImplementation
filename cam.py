import numpy as np
import cv2
import torch
from torch import topk
import torch.nn.functional as F
from hook_features import SaveFeatures
import tqdm
from PIL import Image



def main():
    #model: pytorch resnet model
    #dataset_test: pytorch dataset object
    #transform: defined torchvision transform instance

    #Instructions:
    # 1. train your model
    # 2. set your model to eval mode after training
    # 3. uncommnet the display_cam function

    #note:
    # getitem method of pytorch dataset must return numpy array image and label tensor must be onehot-encoded

    # display_cam(model, dataset_test, target_list, transform)
    return

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w))) #This line computes the weighted sum of the feature maps for the target class. Let me explain it step by step
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam) #min max scaler normalization
    cam_img = np.uint8(255 * cam_img)
    return cam_img

def cam_pass_input(model, image, square):
    img_tensor = torch.unsqueeze(image, dim = 0) 
    final_layer = model._modules.get('layer3') #layer3 is the last conv layer/block of your model
    activated_features = SaveFeatures(final_layer)
    logit = model(img_tensor)
    pred_probabilities = F.softmax(logit, dim=1)
    activated_features.remove()
    class_idx = topk(pred_probabilities, 1)[1].int()
    conf_tensor = topk(pred_probabilities, 1)[0]
    confidence = conf_tensor[0][0].item()
    fc_params= list(model._modules.get('fc').parameters())
    fc_weight = fc_params[0].data.numpy() #weight_softmax_params[0] is weight matrix and weight_softmax_params[1] is bias
    overlay = getCAM(activated_features.features, fc_weight, class_idx)
    cv2.applyColorMap(cv2.resize(overlay,(square, square)), cv2.COLORMAP_JET)
    return overlay, class_idx, confidence

def put_text(heatmap, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = int((heatmap.shape[1] - text_size[0]) / 2)  # center the text horizontally
    y = int(text_size[1] * 2)  # place the text above the image
    result = cv2.putText(heatmap, text, (x, y), font, font_scale, color, thickness)
    return result

def display_cam(model, dataset_test, target_list, transform):
    model.to('cpu')
    print("Class activation map generation progress")
    for i, (image, label, path) in enumerate(tqdm(dataset_test)):
        tensor_img = transform(Image.fromarray(image)) #transform() only take PIL instance
        cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_size = 512
        cv_img = cv2.resize(cv_img, (img_size, img_size))
        overlay, pred_class_idx, confidence = cam_pass_input(model, tensor_img, img_size)
        pred_index = pred_class_idx[0][0].item()
        heatmap = cv2.applyColorMap(cv2.resize(overlay,(img_size, img_size)), cv2.COLORMAP_JET)
        result = cv2.addWeighted(cv_img, 0.55, heatmap, 0.45, 0.0)
        max_index = np.argmax(label, axis=0)
        target_text = target_list[max_index]
        pred_text = target_list[pred_index]
        text = "target: {}, pred: {}, conf: {:.2f}".format(target_text, pred_text, confidence)
        result= put_text(result, text)
        result_concat = cv2.vconcat([result, cv_img]) #concatenate cam to original image
        cv2.imwrite('/home/usr/project/heatmaps//cam{}.jpg'.format(i), result_concat)

if __name__ == '__main__':
    main()
