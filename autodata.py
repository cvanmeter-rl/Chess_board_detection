import os
from PIL import Image, ImageDraw
from realesrgan import RealESRGANer
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
import time
import random
import shutil
from tqdm import tqdm

def load_upsampler(upsampler_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = os.path.basename(upsampler_path)

    if model_name == 'realesr-general-wdn-x4v3.pth':
        # Initialize the model
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type='prelu'
        )

        upsampler = RealESRGANer(
            scale=4,
            model_path=upsampler_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=device
        )
    
    elif model_name == 'RealESRGAN_x4plus.pth':
        # Initialize the model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )

        upsampler = RealESRGANer(
            scale=4,
            model_path=upsampler_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=device
        )

    elif model_name == 'RealESRGAN_x2plus.pth':
        # Initialize the model for scale 2
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2
        )

        upsampler = RealESRGANer(
            scale=2,
            model_path=upsampler_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=device
        )

    return model, upsampler

def upscale_image(img, upsampler):
    # Convert the image to RGB and numpy array
    img_np = np.array(img.convert('RGB'))

    # Perform upscaling
    output, _ = upsampler.enhance(img_np, outscale=4)

    # Convert back to PIL Image and save
    upscaled_img = Image.fromarray(output)

    return upscaled_img


def add_rounded_corners(im, radius):
    # Ensure image has an alpha channel
    im = im.convert("RGBA")

    # Create a same-sized mask with transparent background
    mask = Image.new('L', im.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw a rounded rectangle on the mask
    width, height = im.size
    draw.rounded_rectangle([(0, 0), (width, height)], radius=radius, fill=255)

    # Apply the mask to the image
    im.putalpha(mask)

    return im

def bbox_to_pixel(bbox, img_width, img_height):
    _, x_center, y_center, width, height = bbox

    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # -- Calculate px coordinates of bbox corners
    x1 = int(x_center - width / 2)
    x2 = int(x_center + width / 2)
    y1 = int(y_center - height / 2)
    y2 = int(y_center + height / 2)

    return x1, x2, y1, y2


def paste_board(screenshot_path, 
                label_path, 
                board_path, 
                upsampler, 
                output_path,
                board_idx):
    '''
    Given a base screenshot_path with its corresponding board bbox label,
    paste the board from board_path over the screenshot and output the 
    new image to output_path
    '''
    
    # -- Read screenshot (RGBA for overlay)
    screenshot = Image.open(screenshot_path).convert('RGBA')
    img_width, img_height = screenshot.size

    # -- Read board to paste (RGBA for overlay)
    board = Image.open(board_path).convert('RGBA')

    # -- Upscale the board using Real-ESRGAN
    if upsampler != None:
        board = upscale_image(board, upsampler)
        board = add_rounded_corners(board, 10)
    else:
        board = add_rounded_corners(board, 5)

    # -- Read screenshot board bbox
    with open(label_path, 'r') as f:
        line = f.readline().strip()
        bbox = list(map(float, line.split()))

    # -- Convert bbox to pixel coordinates
    x1, x2, y1, y2 = bbox_to_pixel(bbox, img_width, img_height)
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # -- Resize board to the bounding box dimensions
    board = board.resize((bbox_width, bbox_height), Image.Resampling.LANCZOS)

    # -- Paste board over the screenshot
    screenshot.paste(board, (x1, y1), board)

    # -- Save the new screenshot
    screenshot_filename = os.path.basename(screenshot_path)
    screenshot_filename_new = str.split(screenshot_filename, '.')[0] + f'_{board_idx}.jpg'
    screenshot = screenshot.convert('RGB')
    screenshot.save(os.path.join(output_path, 'images', screenshot_filename_new))

    # -- Copy the corresponding label
    label_filename = os.path.basename(label_path)
    label_filename_new = str.split(label_filename, '.')[0] + f'_{board_idx}.txt'
    shutil.copy(label_path, os.path.join(output_path, 'labels', label_filename_new))
    
def generate_dataset(dataset):

    # -- Get paths of base screenshot images/labels
    images_dir = f'datasets/YOLOv11/{dataset}/images'
    labels_dir = f'datasets/YOLOv11/{dataset}/labels'
    images = [os.path.join(images_dir, file) for file in os.listdir(images_dir)]
    labels = [os.path.join(labels_dir, file) for file in os.listdir(labels_dir)]

    print(f"{dataset} -- Num Images: {len(images)}, Num Labels: {len(labels)}")
    num_images = len(images)

    # -- Get paths of board images
    if dataset == 'train':
        boards_dir = f'kaggle_dataset/train'
        boards_files = os.listdir(boards_dir)
        boards = [os.path.join(boards_dir, file) for file in boards_files]
    else:
        boards_dir = f'kaggle_dataset/test'
        boards_files = os.listdir(boards_dir)

        # -- Shuffle test set
        random.seed(42)
        random.shuffle(boards_files)

        # -- Split first half of test set for testing, second half for validation
        half = len(boards_files) // 2
        if dataset == 'test':
            boards = [os.path.join(boards_dir, file) for file in boards_files[:half]]
        else:
            boards = [os.path.join(boards_dir, file) for file in boards_files[half:]]

    print(f"{dataset} -- Num Boards: {len(boards)}")
    num_boards = len(boards)

    # -- Split boards into len(images) random groups
    random.shuffle(boards)

    group_size = int(num_boards / num_images) # 1000 (80,000 / 80)
    split_boards = [boards[i:i + group_size] for i in range(0, num_boards, group_size)]

    print(f"{dataset} -- split_boards length: {len(split_boards)}")

    '''
    Dataset Creation
    '''
    upsampler_path = 'realesr-general-wdn-x4v3.pth'
    model, upsampler = load_upsampler(upsampler_path)

    # -- For each image, paste all of its corresponding boards as new images
    for img_idx in range(num_images):

        for board_idx in tqdm(range(group_size), desc=f'Image {img_idx} of {num_images}...'):
            screenshot_path = images[img_idx]
            label_path = labels[img_idx]
            board_path = split_boards[img_idx][board_idx]
            output_path = f'datasets/chessboard_data/{dataset}'

            paste_board(screenshot_path=screenshot_path, 
                        label_path=label_path, 
                        board_path=board_path, 
                        upsampler=upsampler,
                        output_path=output_path,
                        board_idx=board_idx)

if __name__ == "__main__":
    datasets = ['test', 'valid', 'train']

    for dataset in datasets:
        generate_dataset(dataset)
