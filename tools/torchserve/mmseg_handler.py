import base64
import os

import cv2
import mmcv
import torch
from mmengine.model.utils import revert_sync_batchnorm
from ts.torch_handler.base_handler import BaseHandler

from mmseg.apis import inference_model, init_model


class MMsegHandler(BaseHandler):
    """TorchServe handler for MMSegmentation models."""

    def initialize(self, context):
        """Initialize model and configuration."""
        properties = context.system_properties
        self._setup_device(properties)
        self.manifest = context.manifest

        model_dir = properties.get('model_dir')
        self._load_model(model_dir)

        self.initialized = True

    def _setup_device(self, properties):
        """Setup computation device."""
        self.map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

        if torch.cuda.is_available():
            gpu_id = str(properties.get('gpu_id', 0))
            self.device = torch.device(f'{self.map_location}:{gpu_id}')
        else:
            self.device = torch.device(self.map_location)

    def _load_model(self, model_dir):
        """Load model from checkpoint and config."""
        serialized_file = self.manifest['model']['serializedFile']
        checkpoint_path = os.path.join(model_dir, serialized_file)
        config_path = os.path.join(model_dir, 'config.py')

        self.model = init_model(config_path, checkpoint_path, self.device)
        self.model = revert_sync_batchnorm(self.model)

    def preprocess(self, data):
        """Preprocess input data."""
        images = []

        for row in data:
            image_data = row.get('data') or row.get('body')
            image = self._decode_image(image_data)
            images.append(image)

        return images

    def _decode_image(self, image_data):
        """Decode image from various formats."""
        if isinstance(image_data, str):
            # Base64 encoded string
            image_data = base64.b64decode(image_data)

        return mmcv.imfrombytes(image_data)

    def inference(self, data, *args, **kwargs):
        """Run model inference."""
        return [inference_model(self.model, img) for img in data]

    def postprocess(self, data):
        """Postprocess inference results."""
        return [self._encode_result(image_result) for image_result in data]

    def _encode_result(self, image_result):
        """Encode segmentation result as PNG."""
        # Get the first element which contains the segmentation mask
        segmentation_mask = image_result[0].astype('uint8')

        # Encode as PNG
        success, buffer = cv2.imencode('.png', segmentation_mask)
        if not success:
            raise RuntimeError('Failed to encode segmentation result as PNG')

        return buffer.tobytes()