import random as r
import os, cv2
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from create_dataset import CustomDataset
from utils import *
from model import model_DeepLabV3


x_train_dir='img_path'
y_train_dir='mask_patj'

x_valid_dir='img_path'
y_valid_dir='mask_path'




dataset = CustomDataset(x_train_dir, y_train_dir)
random_idx = r.randint(0, len(dataset) - 1)
image, mask = dataset[random_idx]

image_arr,mask_arr=from_tensor_to_numpy(image,mask)

visualize(
    original_image=image_arr,
    ground_truth_mask=(mask_arr),
    binar_mask=cv2.threshold(mask_arr, 0.5, 255, cv2.THRESH_BINARY)[1]
)

# create segmentation model with pretrained encoder
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset=CustomDataset(x_train_dir,y_train_dir)
valid_dataset=CustomDataset(x_valid_dir,y_valid_dir)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

model=model_DeepLabV3(ENCODER,ENCODER_WEIGHTS,ACTIVATION)

# Set num of epochs
EPOCHS = 15

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define loss function
loss = smp.utils.losses.DiceLoss()

# define metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# define optimizer
optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

# define learning rate scheduler (not used in this NB)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5,)

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)


if __name__ == "__main__":

    print('MAIN')
    best_iou_score = 0.0
    train_logs_list, valid_logs_list = [], []

    for i in range(0, EPOCHS):

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        train_logs_list.append(train_logs)
        valid_logs_list.append(valid_logs)

        # Save model if a better val IoU score is obtained

    torch.save(model, './DeepLab_model_1channel_mask_andrey_dataset.pth')#save weights

    # save if onnx
    model.eval()
    model.to("cpu")
    dummy_input=torch.randn(1,3,256,256)
    input_names = [ "actual_input" ]
    output_names = [ "output" ]
    torch.onnx.export(model,
                      dummy_input,
                      "Deep_Lab.onnx",
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      export_params=True,
                      do_constant_folding=True,
                      opset_version=11
                      )
