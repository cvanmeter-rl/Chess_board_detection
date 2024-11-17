import cv2
from PIL import Image
import numpy as np
import time
# -- PyTorch
from torchvision import transforms
import torch
import torch.nn.functional as F
# -- Models
from ultralytics import YOLO
from torchvision.models.resnet import ResNet
from stockfish import Stockfish
# -- Utils
import pyautogui
from skimage.util import view_as_blocks
import msvcrt

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


def detect_board(board_detector: YOLO, screenshot: np.ndarray) -> np.ndarray:
    '''
    Given a screenshot, detects and returns a chessboard from the
    screen using a YOLO11n model

    Parameters:
    - screenshot (monitor_h, monitor_w, 3): Screenshot of the primary monitor,
    presumably containing a chessboard
    - detector: YOLO11n model trained to detect chessboards from screenshots
    with precise bounding boxes

    Returns:
    - board (board_h, board_w, 3): Chessboard detected by the YOLO11n board
    detection model, cropped from the original screenshot using YOLO bounding box
    '''
    # -- Detect chessboard from screenshot
    results = board_detector(screenshot, max_det=1, verbose=False)[0]
    bbox = results.boxes

    # -- If no objects are found, inform the user
    if bbox.conf.numel() == 0:
        print("\n-- No chessboards detected! --")
        return None
    
    # -- Crop chessboard from screenshot using bbox
    x1, y1, x2, y2 = map(int, bbox.xyxy[0])
    board = screenshot[y1 : y2 + 1, x1 : x2 + 1]

    return board


def segment_board(board: np.ndarray) -> np.ndarray:
    '''
    Given a detected board, resizes and segments the 
    board into a uniform 8x8 grid to prepare for piece classification

    Parameters:
    - board (board_h, board_w, 3): Chessboard detected by the YOLO11n board
    detection model, cropped from the original screenshot using YOLO bounding box

    Returns:
    - squares (8, 8, square_size, square_size, 3): Numpy array of the segmented
    square images cropped from the detected chessboard
    '''
    # -- Resize board to be evenly segmented into an 8x8 grid
    board_shape_avg = (board.shape[0] + board.shape[1]) / 2
    board_size = round(board_shape_avg / 8) * 8

    board_resized = cv2.resize(board, (board_size, board_size))

    # -- Segment the board into an 8x8 uniform grid of squares (shape = (8, 8, square_size, square_size, 3))
    square_size = board_size // 8
    squares = view_as_blocks(board_resized, (square_size, square_size, 3)).squeeze()

    return squares


def classify_pieces(piece_classifier: ResNet, squares: np.ndarray) -> np.ndarray:
    '''
    Given an array of segmented chessboard squares in the shape:
    - (8, 8, square_size, square_size, 3)
    Classifies each piece using a ResNet-18 model and returns an
    8x8 array of piece classifications for Forsyth-Edwards Notation (FEN)

    Parameters:
    - squares (8, 8, square_size, square_size, 3): Numpy array of the segmented
    square images cropped from the detected chessboard
    - classifier: ResNet-18 model trained to classify pieces from chip-level images
    of chessboard squares

    Returns:
    - pieces (8, 8): Numpy array of the FEN classification for each square
    '''
    # -- Initialize empty piece classification array
    pieces = np.full((8, 8), fill_value=' ', dtype=object)

    # -- Define mapping from ResNet-18 model classes [0, 12] to FEN piece notation
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

    # -- Classify each square and store in pieces array
    for i in range(8):
        for j in range(8):
            # Preprocess square image
            square = squares[i][j]
            square = preprocess_square(square)

            # Run forward pass and obtain prediction
            with torch.no_grad():
                outputs = piece_classifier(square)
                probs = F.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, dim=1)

            # Store FEN classification in pieces array
            pieces[i][j] = class_map[int(pred)]

    return pieces


def indices_to_alg(indices):
    '''
    Converts board indices ([0, 7], [0, 7]) to algebraic notation ([8, 1], [a, h])

    Parameters:
    - indices: Tuple of 0-indexed board position (row, col)

    Returns:
    - position: String representing a square of a board in algebraic notation
    '''
    row, col = indices

    # -- Row (Rank)
    rank = str(8 - row)

    # -- Col (File)
    file = chr(col + ord('a'))

    # -- Concatenate file and rank
    position = file + rank

    return position


def infer_move(pieces_1: np.ndarray, pieces_2: np.ndarray, active_color: str):
    '''
    Given two board states with a single move in between, infers the
    intermediate move and returns a string in long algebraic notation:
    - Normal Move:
        * "<start_pos><end_pos>"
    - Castle:
        * "O-O" (Kingside)
        * "O-O-O" (Queenside)

    Parameters:
    - pieces_1 (8, 8): Board position preceding the intermediate move
    - pieces_2 (8, 8): Board position following the intermediate move

    Returns:
    - move: String of the starting/ending positions of the moved piece
    '''
    # -- Determine if the move is a normal move (1 new space) or a castle (2 new spaces)
    new_empty = len(np.argwhere((pieces_1 != ' ') & (pieces_2 == ' ')))

    # -- Normal Move
    if new_empty == 1:
        # Define move as <start_pos><end_pos> in algebraic notation
        move_start = np.argwhere((pieces_1 != pieces_2) & (pieces_2 == ' '))[0]
        move_end = np.argwhere((pieces_1 != pieces_2) & (pieces_2 != ' '))[0]

        move = indices_to_alg(move_start) + indices_to_alg(move_end)

    # -- Castle
    elif new_empty == 2:
        # Define rook notation for the inferred color
        if active_color == 'w':
            rook = 'r'
        elif active_color == 'b':
            rook = 'R'

        # Determine number of rook moves
        rook_start = np.argwhere((pieces_1 == rook) & (pieces_2 == ' '))[0]
        rook_end   = np.argwhere((pieces_1 == ' ') & (pieces_2 == rook))[0]
        rook_moves = abs(rook_end[1] - rook_start[1])

        # Kingside castle if rook_moves = 2, Queenside castle if rook_moves = 3
        if rook_moves == 2:
            move = 'O-O'
        elif rook_moves == 3:
            move = 'O-O-O'

    return move


def generate_fen(stockfish: Stockfish,
                 pieces: np.ndarray, 
                 active_color: str) -> str:
    '''
    Converts 8x8 array of piece classifications to FEN notation
    for compatibility with Stockfish

    Parameters:
    - pieces (8, 8): Numpy array of strings classifying each
    square as a piece or empty according to FEN notation

    Returns:
    - fen: String of FEN notation compatible with Stockfish (6 required terms)
        * Example:
            - "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
            - "<piece notation> <active color> <castling rights> <en passant> <halfmove clock> <fullmove number>" 
        * Definitions:
            - <piece notation>:
                * Piece notation left to right per row (black lowercase, white uppercase)
                * Consecutive empty squares designated by numbers (e.g. 3 = 3 empty in a row)
                * Rows separated by '/' (for Stockfish API)
            - <active color>
                * Lowercase w for white's turn, lowercase b for black's turn
            - <castling rights>
                1. Uppercase KQ for white's kingside and queenside castling rights, respectively
                2. Lowercase kq for black's kingside and queenside castling rights, respectively
                * Cannot castle if castled previously or if king moves through/into check
            - <en passant>
                * Explanation:
                    - A pawn is eligible for an en passant capture on the current move iff it moved 
                    two squares on the *previous* turn
                        - Therefore, only one pawn is eligible to be en passant captured per move
                    - e.g. white's pawn moves e2->e4, black can only en passant white's pawn on its *next move*
                * Notation:
                    - The square behind the pawn eligible for en passant capture
                        * e.g. pawn moves e2->e4, <en passant> = e3
            - <halfmove clock>
                * How many moves *both* players have made since the last pawn move
                or piece capture (any pawn move, any piece capturing any other piece)
                    - Used to enforce the 50-move draw rule (halfmove counter = 100; 50 moves each)
            - <fullmove number>
                * The number of *completed* turns in the game; incremented by one every time black moves
                    - e.g. <fullmove number> = 2 means either:
                        1. white/black have both moved twice
                        2. white moved 3 times/black moved 2 times (will increment on black's next move)
    '''
    fen_fields = []

    # -- Piece Notation
    piece_notation = ""

    for row in range(pieces.shape[0]):
        num_empty = 0

        for piece in pieces[row]:
            # Count consecutive empty squares
            if piece == ' ':
                num_empty += 1
            # On an initial non-empty square, record num_empty first
            elif num_empty > 0:
                piece_notation += str(num_empty)
                piece_notation += piece
                num_empty = 0
            # Otherwise, simply record piece notation
            else:
                piece_notation += piece

        # Include final consecutive empty squares
        if num_empty > 0:
            piece_notation += str(num_empty)

        # Separate rows with slashes (/)
        if row != 7:
            piece_notation += '/'
    
    fen_fields.append(piece_notation)

    # -- Active Color
    fen_fields.append(active_color)

    # -- Castling Rights (must infer for other player)
    '''
    How?
    - User will set their color (White (w) or Black (b))
    - Assume (really required) that users will run the pipeline *at least* once per "turn" (White move and Black move)
        * Two scenarios (let user's color = 'w'):
            - 'w' -> 'w'
                * Program must infer Black's move that occurred between
            - 'w' -> 'b'
                * Program recommends a move for Black, but will have to infer Black's real move on the next 'w' run
    - Inferring Black's move:
        * Know:
            - Board position after white's first move (w_1 = detected board position + Stockfish move)
            - Board position after black's move (b_1 = detected board position)
        * Therefore, b_1 - w_1 will give you the change from black's move
            - Standard move:
                * A changed square where b_1 IS empty shows where black's piece came from
                    - indices = np.where((w_1 != b_1) & (b_1 == ' '))[0]
                * A changed square where b_1 IS NOT empty shows where black's piece went
                    - indices = np.where((w_1 != b_1) & (b_1 != ' '))[0]
            - Castle:
                * Two new empty squares show that black castled
                    - Kingside (short): Rook moves two columns
                    - Queenside (long): Rook moves three columns
                * Check number of columns rook moves:
                    - rook_start = np.where((w_1 == 'r') & (b_1 == ' '))[0]
                    - rook_end   = np.where((w_1 == ' ') & (b_1 == 'r'))[0]
                    - rook_moves = rook_end[1] - rook_start[1]
    '''


    # -- En Passant

    # -- Halfmove Clock

    # -- Fullmove Number

    # -- Concatenate FEN into space-separated string for Stockfish
    fen = ' '.join(fen_fields)

    return fen


def run_pipeline(board_detector: YOLO, 
                 piece_classifier: ResNet, 
                 stockfish: Stockfish,
                 active_color: str):
    '''
    Detect Chessboard
    '''
    print("Detecting chessboard...\n")

    # -- Take screenshot of primary monitor
    screenshot = np.array(pyautogui.screenshot())

    # -- Detect chessboard from screenshot
    board = detect_board(board_detector, screenshot)

    '''
    Classify Pieces
    '''
    print("Classifying pieces...\n")

    # -- Segment squares from detected chessboard
    squares = segment_board(board)

    # -- Classify the piece of each square in FEN notation
    pieces = classify_pieces(piece_classifier, squares)

    '''
    Recommend Move
    '''
    print("Recommending a move...\n")

    # -- Determine FEN notation for current game position
    fen = generate_fen(stockfish, pieces, active_color)
    print(fen)

    # -- Pass FEN notation to Stockfish to generate move recommendation


def main():
    '''
    Load Models
    '''
    print("Loading models...")

    # -- Load YOLO11n chessboard detection model (epoch27.pt)
    board_detector = YOLO(f'models/board_detector/epoch27.pt')

    # -- Load ResNet-18 piece classification model (classifier_epoch1.pth)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    piece_classifier = torch.load(f'models/piece_classifier/classifier_epoch1.pth', map_location=device).eval()

    # -- Load Stockfish17 engine
    stockfish = Stockfish(path='models/stockfish/stockfish-windows-x86-64-avx2')

    print("Models successfully loaded.\n")

    '''
    Run pipeline on Enter, exit on Escape
    '''
    while True:
        # -- If a key is pressed...
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # Run pipeline for White on 'w'
            # if key == b'w':
            #     run_pipeline(board_detector, 
            #                  piece_classifier, 
            #                  stockfish,
            #                  active_color='w')
            # # Run pipeline for Black on 'b'
            # elif key == b'b':
            #     run_pipeline(board_detector,
            #                  piece_classifier,
            #                  stockfish,
            #                  active_color='b')

            # Run pipeline on Enter
            if key == b'\r':
                run_pipeline(board_detector, 
                             piece_classifier, 
                             stockfish,
                             active_color='w')
            # Exit program on Escape
            elif key == b'\x1b':
                break


if __name__ == "__main__":
    main()