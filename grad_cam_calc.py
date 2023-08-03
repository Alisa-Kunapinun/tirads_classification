

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import deprocess_image
# from torchvision.models import resnet50
import cv2

# model = resnet50(pretrained=True)
# target_layers = [model.layer4[-1]]

# rgb_img = cv2.imread("example01.jpeg", 1)[:, :, ::-1]
# rgb_img = np.float32(rgb_img) / 255
# input_tensor = preprocess_image(rgb_img,
#                                 mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
# Note: input_tensor can be a batch tensor with several images!

class gradcam:
    def __init__(self, model, target_layers):
        self.model = model
        # self.target_layers = [model.layer4[-1]]
        self.target_layers = target_layers
        # Construct the CAM object once, and then re-use it on many images:
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)

        # pred_composition = classify_output[:,0:4]
        # pred_echoginicity = classify_output[:,4:8]
        # pred_shape = classify_output[:,8:10]
        # pred_margin = classify_output[:,10:14]
        # pred_macrocal = classify_output[:,14:16]
        # pred_peripheral = classify_output[:,16:18]
        # pred_microcal = classify_output[:,18:20]
        # pred_comet = classify_output[:,20:22]
        # pred_is_malignant = classify_output[:,22:24]
        self.composition_cystic = 0
        self.composition_spongiform = 1
        self.composition_mixed = 2
        self.composition_solid = 3
        self.echoginicity_anechoic = 4
        self.echoginicity_hyperechoic = 5
        self.echoginicity_hypoechoic = 6
        self.echoginicity_very_hypoechoic = 7
        self.shape_wider = 8
        self.shape_taller = 9
        self.margin_smooth = 10
        self.margin_ill_defined = 11
        self.margin_irregular = 12
        self.margin_extension = 13
        self.macrocal = 15
        self.peripheral = 17
        self.microcal = 19
        self.comet = 21
        self.benign = 22
        self.malignant = 23

    def find_target(self, input_tensor, target, is_malignant = 1.0, threshold = 0.8):
        targets = [ClassifierOutputTarget(target)]

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

        # In this example grayscale_cam has only one image in the batch:
        # this is mask
        if is_malignant >= threshold:
            is_malignant = threshold
        is_malignant = is_malignant / threshold

        grayscale_cam = grayscale_cam[0, :] * is_malignant

        transparent = np.copy(grayscale_cam)
        transparent[transparent >= 0.2] = 1.0
        # print(np.max(transparent))
        # print(np.min(transparent))

        # print(real_image.shape)
        # print(gray.shape)
        rgb = input_tensor[0,0,:,:].detach().cpu().numpy()
        rgb_img = cv2.merge((rgb, rgb, rgb))
        rgb_img = np.float32(rgb_img) / 255
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.8)

        # print(np.max(rgb))
        rgba = np.uint8(rgb * transparent)
        # print(np.max(rgba))
        self.rgba_img = cv2.merge((rgba, rgba, rgba))
        # self.rgba_img = np.float32(self.rgba_img) / 255

        self.cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=self.model, use_cuda=True)
        gb = gb_model(input_tensor, target_category=None)

        self.cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        self.cam_gb = deprocess_image(self.cam_mask * gb)
        self.gb = deprocess_image(gb)

    def save(self, cam_image, gb, cam_gb, cam_transparent = None):
        cv2.imwrite(cam_image, self.cam_image)
        cv2.imwrite(gb, self.gb)
        cv2.imwrite(cam_gb, self.cam_gb)
        if (cam_transparent is not None):
            cv2.imwrite(cam_transparent, self.rgba_img)

    def save_nodule_and_mask(self, input_tensor, nodule, mask):
        rgb = input_tensor[0,0,:,:].numpy()
        rgb_img = cv2.merge((rgb, rgb, rgb))
        self.nodule = rgb_img
        self.mask = input_tensor[0,2,:,:].numpy()

        cv2.imwrite(nodule, self.nodule)
        cv2.imwrite(mask, self.mask)

