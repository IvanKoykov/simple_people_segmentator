
import segmentation_models_pytorch as smp


def model_DeepLabV3(ENCODER,ENCODER_WEIGHTS,ACTIVATION):
    model = smp.DeepLabV3(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=1,
    activation=ACTIVATION,
)
    return model
