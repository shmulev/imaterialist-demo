import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.onnx
import torch.onnx.symbolic_opset9 as onnx_symbolic
import segmentation_models_pytorch as smp
from preprocessing import get_preprocessing
from predict import predict_mask


DEVICE = 'cuda'

# predict function
def predict_mask_from_raw(image_numpy, model, activation=None, device=DEVICE):
    ENCODER = 'resnext50_32x4d'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    preprocessing = get_preprocessing(preprocessing_fn)
    img_p = preprocessing(image=image_numpy)['image']
    pr_mask = predict_mask(img_p, model, activation=activation, device=device)
    
    return pr_mask


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# visualization of predictions
def show_mask(image_orig, mask):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_orig)
    plt.imshow(mask, cmap='magma_r', alpha=0.4)
    plt.show()
    return plt


def show_mask_2(image_orig, mask):
    plt.figure(figsize=(10, 10))
    mask_ext = np.zeros((*mask.shape, 3))
    mask_ext[..., 0] = mask * 237
    mask_ext[..., 1] = mask * 184
    mask_ext[..., 2] = mask * 5

    blend = np.clip(image_orig * 0.8 + mask_ext * 0.5, 0, 255).astype('uint8')
    plt.imshow(blend.astype('uint8'))
    plt.show()


# function to collapse several binary masks into colored-image
def melt_masks(masks, class_values):
    mask3D = np.zeros((*masks[0].shape, 3))
    for ix, mask in enumerate(masks):
        mask3D[mask == 1] = np.array(class_values[ix])
    return mask3D.astype('uint8')


# function to save PyTorch model to ONNX
def drop_onnx(model, model_path, input_tensor_example):
    """
    Save PyTorch model to ONNX format
    :param model: pytorch model, instance of nn.Module
    :param model_path: file path for saving
    :param input_tensor_example: torch.Tensor input example
    :param output_numpy: np.array output example
    :return: None
    """
    print('Exporting model to ONNX...')
    def upsample_nearest2d(g, input, output_size):
        # Currently, TRT 5.1/6.0 ONNX Parser does not support all ONNX ops
        # needed to support dynamic upsampling ONNX forumlation
        # Here we hardcode scale=2 as a temporary workaround
        scales = g.op("Constant", value_t=torch.tensor([1.,1.,2.,2.]))
        return g.op("Upsample", input, scales, mode_s="nearest")
    
    onnx_symbolic.upsample_nearest2d = upsample_nearest2d
    
    if input_tensor_example is not None:
        input_tensor_example = input_tensor_example.to('cpu')
       
    torch.onnx.export(model.to('cpu'),  # model being run
                      input_tensor_example,  # model input (or a tuple for multiple inputs)
                      f"{model_path}",  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      # do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                      #               'output': {0: 'batch_size'}}
                     )
    print(f'Model is successfully saved to ONNX: {model_path}.')
    
    
def load_model(model_path, Model, model_params, device):
    model = Model(**model_params)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    model = model.to(device)
    model = model.eval()
    return model
