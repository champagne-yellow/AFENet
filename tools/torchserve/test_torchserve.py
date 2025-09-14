from argparse import ArgumentParser
from io import BytesIO

import matplotlib.pyplot as plt
import mmcv
import requests

from mmseg.apis import inference_model, init_model


def parse_args():
    """Parse command line arguments for model comparison."""
    parser = ArgumentParser(
        description='Compare result of torchserve and pytorch, and visualize them.'
    )
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('model_name', help='The model name in the server')
    parser.add_argument(
        '--inference-addr',
        default='127.0.0.1:8080',
        help='Address and port of the inference server'
    )
    parser.add_argument(
        '--result-image',
        type=str,
        default=None,
        help='save server output in result-image'
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device used for inference'
    )

    return parser.parse_args()


def get_torchserve_prediction(image_path, inference_addr, model_name):
    """Get prediction from TorchServe server."""
    url = f'http://{inference_addr}/predictions/{model_name}'

    with open(image_path, 'rb') as image_file:
        response = requests.post(url, image_file)

    if response.status_code != 200:
        raise RuntimeError(
            f'TorchServe request failed with status code: {response.status_code}'
        )

    return response.content


def save_and_display_result(image_content, result_image_path=None):
    """Save and display the result image."""
    if result_image_path:
        with open(result_image_path, 'wb') as out_file:
            out_file.write(image_content)
        image_data = mmcv.imread(result_image_path, 'grayscale')
    else:
        image_data = plt.imread(BytesIO(image_content))

    plt.imshow(image_data)
    plt.title('TorchServe Prediction')
    plt.axis('off')
    plt.show()


def get_local_prediction(config_path, checkpoint_path, image_path, device):
    """Get prediction from local PyTorch model."""
    model = init_model(config_path, checkpoint_path, device)
    image = mmcv.imread(image_path)
    return inference_model(model, image)


def display_local_prediction(result):
    """Display local PyTorch prediction result."""
    plt.imshow(result[0])
    plt.title('Local PyTorch Prediction')
    plt.axis('off')
    plt.show()


def main(args):
    """Main function to compare TorchServe and local predictions."""
    # Get and display TorchServe prediction
    try:
        torchserve_content = get_torchserve_prediction(
            args.img, args.inference_addr, args.model_name
        )
        save_and_display_result(torchserve_content, args.result_image)
    except Exception as e:
        print(f"Error getting TorchServe prediction: {e}")
        return

    # Get and display local PyTorch prediction
    try:
        local_result = get_local_prediction(
            args.config, args.checkpoint, args.img, args.device
        )
        display_local_prediction(local_result)
    except Exception as e:
        print(f"Error getting local prediction: {e}")


if __name__ == '__main__':
    args = parse_args()
    main(args)