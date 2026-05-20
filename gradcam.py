import torch
import numpy as np
import cv2
from PIL import Image


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.backbone.features.denseblock4

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, tensor, class_idx):
        """Generate Grad-CAM heatmap for given class index."""
        self.model.eval()

        # Need gradients for Grad-CAM
        tensor = tensor.clone().requires_grad_(True)

        output = self.model(tensor)
        self.model.zero_grad()

        score = output[0, class_idx]
        score.backward()

        # Pool gradients across channels
        gradients = self.gradients[0]       # [C, H, W]
        activations = self.activations[0]   # [C, H, W]

        weights = gradients.mean(dim=[1, 2])  # [C]
        cam = (weights[:, None, None] * activations).sum(dim=0)  # [H, W]
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def overlay_heatmap(self, pil_image, cam, alpha=0.4):
        """Overlay Grad-CAM heatmap on original image."""
        img = np.array(pil_image.resize((224, 224)).convert('RGB'))

        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = (alpha * heatmap + (1 - alpha) * img).astype(np.uint8)
        return Image.fromarray(overlay)
