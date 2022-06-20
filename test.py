from create_dataset import CustomDataset
import random as r
import glob
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import onnxruntime as ort
from torch.utils.data import DataLoader
from utils import *


def to_numpy_for_test_data(tensor):
    return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )
def blur(path_img):

    orig_img = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
    frame_height, frame_width = orig_img.shape[:2]
    image = orig_img
    image = cv2.resize(image, (256, 256))
    image = np.rollaxis(image, 2, 0)
    image = torch.Tensor(image)
    image = image.to(DEVICE).unsqueeze(0)

    onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy_for_test_data(image)}
    onnx_output = onnx_session.run(None, onnx_inputs)
    pred_mask = onnx_output[0]
    print(pred_mask.shape)

    pred_mask = pred_mask.squeeze(0)

    pred_mask = np.transpose(pred_mask, (1, 2, 0))
    pred_mask = cv2.resize(pred_mask, (frame_width, frame_height))
    pred_mask = cv2.threshold(pred_mask, 0.5, 1, cv2.THRESH_BINARY)[1]
    pred_mask = pred_mask.astype(np.uint8)

    output_image = cv2.GaussianBlur(orig_img, (-1, -1), 20)
    output_image[pred_mask == 1] = orig_img[pred_mask == 1]

    plt.imshow(pred_mask)
    plt.show()
    plt.imshow(orig_img)
    plt.show()
    plt.imshow(output_image)
    plt.show()




if __name__ == "__main__":


    x_test_dir='img_path'
    y_test_dir='mask_path'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load weights
    if os.path.exists('path_to_file_with_weights.pth'):
        best_model = torch.load('path_to_file_with_weights.pth', map_location=DEVICE)
        print('Loaded DeepLabV3+ model from this run.')

    # load from onnx
    onnx_session= ort.InferenceSession("Deep_Lab.onnx")
    test_dataset=CustomDataset(x_test_dir,y_test_dir)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)



    for idx in range(len(test_dataset)):
        image, gt_mask = test_dataset[idx]
        image, gt_mask = image.to(DEVICE).unsqueeze(0), gt_mask.to(DEVICE)
        onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy_for_test_data(image)}
        onnx_output = onnx_session.run(None, onnx_inputs)

        pred_mask = onnx_output[0]
        pred_mask = pred_mask.squeeze(0)
        pred_mask = np.transpose(pred_mask, (1, 2, 0))


        image=image.squeeze(0)
        image,gt_mask=from_tensor_to_numpy(image,gt_mask)


        visualize(
            original_image=image,
            ground_truth_mask=gt_mask,
            predict_mask=pred_mask,
        )


    path_img='path_to_the_image_for_the_blurr.jpg'
    blur(path_img)


