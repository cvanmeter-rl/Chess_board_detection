import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from torchvision import transforms
import torch
import torch.nn.functional as F
# -- Models
from ultralytics import YOLO
from torchvision.models.resnet import ResNet
# -- Utils
import pyautogui
from skimage.util import view_as_blocks

def preprocess_square(square):
    '''
    Applies the proper transformations to prepare a chessboard square
    to be used as input for the pretrained ResNet-18 model (using ImageNet values)

    Parameters:
    - square (square_size, square_size, 3): Single square image cropped from a chessboard

    Returns:
    - square (1, square_size, square_size, 3): Single square image with ImageNet-based
    augmentations/normalization applied and batch dimension added for ResNet-18 compatibility
    '''
    square = Image.fromarray(square)

    data_transforms = transforms.Compose([
        transforms.Resize(224),  # Resize to match training
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # Mean for normalization
                             [0.229, 0.224, 0.225])  # Std for normalization
    ])

    square = data_transforms(square)
    square = square.unsqueeze(0)

    return square

def plot_pipeline():
    # -- Plot combined plot of piece segmentation pipeline
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(nrows=4, ncols=2, height_ratios=[1, 1, 1, 1])

    screenshot = Image.open('results/screenshot.png')
    annotation = Image.open('results/annotation.png')
    board = Image.open('results/board.png')
    board_cropped = Image.open('results/board_cropped.png')
    pieces = Image.open('results/pieces.png')

    axs0 = fig.add_subplot(gs[0, :])
    axs0.imshow(screenshot)
    axs0.set_title('Screenshot')
    axs0.axis('off')

    axs1 = fig.add_subplot(gs[1, :])
    axs1.imshow(annotation)
    axs1.set_title('Chessboard Detection')
    axs1.axis('off')

    axs2 = fig.add_subplot(gs[2, 0])
    axs2.imshow(board)
    axs2.set_title('Detected Board')
    axs2.axis('off')

    axs3 = fig.add_subplot(gs[2, 1])
    axs3.imshow(board_cropped)
    axs3.set_title('Cropped Board')
    axs3.axis('off')

    axs4 = fig.add_subplot(gs[3, :])
    axs4.imshow(pieces)
    axs4.set_title('Piece Classification')
    axs4.axis('off')

    fig.savefig('results/pipeline.png', bbox_inches='tight', pad_inches=0, dpi=300)

def main():
    # -- Load YOLO11n chessboard detection model
    detector_model = 'epoch27.pt'
    print(f"Loading YOLO model ({detector_model})...")
    board_detector = YOLO(f'models/board_detector/{detector_model}')
    print(f"YOLO model ({detector_model}) loaded successfully.")

    # -- Take screenshot
    print("\nTaking screenshot...")
    
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)

    # -- Detect chessboard from the screenshot
    print("\nDetecting chessboard...")

    results = board_detector(screenshot, max_det=1)[0]
    bbox = results.boxes

    # If no objects are found, inform the user
    if bbox.conf.numel() == 0:
        print("\n-- No chessboards detected! --")
        return

    x1, y1, x2, y2 = map(int, bbox.xyxy[0])

    # -- Crop chessboard from screenshot using bbox
    board = screenshot[y1 : y2 + 1, x1 : x2 + 1]
    print(f"\nboard.shape: {board.shape}")

    # -- Resize/crop board to be evenly segmented into an 8x8 grid
    board_shape_avg = (board.shape[0] + board.shape[1]) / 2
    board_size = round(board_shape_avg / 8) * 8
    board_resized = cv2.resize(board, (board_size, board_size))

    board_w, board_h = board_resized.shape[1], board_resized.shape[0]
    square_w, square_h = board_w // 8, board_h // 8

    board_cropped = board_resized[:square_h * 8, :square_w * 8]
    print(f"board_cropped.shape: {board_cropped.shape}")

    # -- Segment squares from chessboard
    print("\nSegmenting pieces...")
    
    squares = view_as_blocks(board_cropped, (square_h, square_w, 3))
    print(f'squares shape: {squares.shape}')

    # -- Classify chessboard squares
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    classifier_model = 'classifier_epoch1.pth'
    print(f"\nLoading ResNet-18 model ({classifier_model})...")
    piece_classifier = torch.load(f'models/piece_classifier/{classifier_model}', map_location=device)
    print(f"ResNet-18 model ({classifier_model}) loaded successfully.")
    print(f"type: {type(piece_classifier)}")

    piece_classifier.to(device)
    piece_classifier.eval()

    square_preds = np.full((8, 8), "  ", dtype=object)

    # class_map = {
    #     0 : 'b',
    #     1 : 'k',
    #     2 : 'n',
    #     3 : 'p',
    #     4 : 'q',
    #     5 : 'r',
    #     6 : ' ',
    #     7 : 'B',
    #     8 : 'K',
    #     9 : 'N',
    #     10: 'P',
    #     11: 'Q',
    #     12: 'R'
    # }

    class_map = {
        0 : ' ',
        1 : 'N',
        2 : 'P',
        3 : 'Q',
        4 : 'R',
        5 : 'b',
        6 : 'k',
        7 : 'n',
        8 : 'p',
        9 : 'q',
        10: 'r',
        11: 'B',
        12: 'K'
    }

    for i in range(8):
        for j in range(8):
            square = squares[i][j][0]
            square = preprocess_square(square)

            with torch.no_grad():
                outputs = piece_classifier(square)
                probs = F.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, dim=1)

            square_preds[i][j] = class_map[int(pred)]

    print(f"\nPiece Classifications: \n{square_preds}")

    # -- Convert piece classifications to FEN


    # -- Save images
    cv2.imwrite('results/screenshot.png', cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
    cv2.imwrite('results/annotation.png', cv2.cvtColor(results.plot(), cv2.COLOR_RGB2BGR))
    cv2.imwrite('results/board.png', cv2.cvtColor(board, cv2.COLOR_RGB2BGR))
    cv2.imwrite('results/board_cropped.png', cv2.cvtColor(board_cropped, cv2.COLOR_RGB2BGR))

    # -- Plot segmented squares
    fig, axs = plt.subplots(8, 8)
    for i in range(8):
        for j in range(8):
            square = squares[i][j][0]
            axs[i][j].imshow(square)
            axs[i][j].set_xlabel(square_preds[i][j])
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])

    fig.subplots_adjust(hspace=1.0, wspace=1.0)
    fig.savefig('results/pieces.png', bbox_inches='tight')
    plt.close(fig)

    plot_pipeline()

if __name__ == "__main__":
    main()