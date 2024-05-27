import os
import sys
import cv2
import time
import pickle
import numpy as np
import albumentations as A
from PIL import Image
from io import BytesIO


transforms = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0,
            always_apply=True,
        ),
    ]
)


def get_size_mb(data):
    memory_size = len(pickle.dumps(data))
    return memory_size / 2**20


def cv2_preload(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    return image

def cv2_access(image):
    image = transforms(image=image)["image"]


def cv2_preload_compressed(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    _, buffer = cv2.imencode('.jpg', image)
    return buffer

def cv2_access_compressed(buffer):
    image = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
    image = transforms(image=image)["image"]


def pil_preload(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    return image

def pil_access(image):
    image = np.array(image)
    image = transforms(image=image)["image"]


def pil_preload_compressed(image_path):
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((224, 224))
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    return buffer.getvalue()

def pil_access_compressed(buffer):
    image = Image.open(BytesIO(buffer))
    image = np.array(image)
    image = transforms(image=image)["image"]


def test_lib(fn_preload, fn_access, lib_name, image_directory, max_image_number):
    images = []
    start_time = time.time()
    for filename in os.listdir(image_directory)[:max_image_number]:
        if filename.endswith('.jpeg') or filename.endswith('.jpg'):
            image_path = os.path.join(image_directory, filename)
            image = fn_preload(image_path)
            images.append(image)
    preload_time = time.time() - start_time
    memory = get_size_mb(images)
    print(f"{lib_name} preload pipeline: {preload_time:.2f} seconds, {memory:.2f} MB")

    start_time = time.time()
    for i in range(len(images)):
        fn_access(images[i])
    access_time = time.time() - start_time
    print(f"{lib_name} access pipeline: {access_time:.2f} seconds\n")


def main(image_directory, max_image_number=1000):
    lib_args = [(cv2_preload, cv2_access, "OpenCV (noncompressed)"),
            (cv2_preload_compressed, cv2_access_compressed, "OpenCV (compressed)"),
            (pil_preload, pil_access, "Pillow (noncompressed)"),
            (pil_preload_compressed, pil_access_compressed, "Pillow (compressed)")]
    for args in lib_args:
        test_lib(*args, image_directory, max_image_number)


if __name__ == '__main__':
    if not 2 <= len(sys.argv) <= 3:
        print(f"Usage: {sys.argv[0]} <image_directory> [<max_image_number>]")
        sys.exit(1)
    
    image_directory = sys.argv[1]
    if len(sys.argv) == 3:
        max_image_number = int(sys.argv[2])
        main(image_directory, max_image_number)
    else:
        main(image_directory)
