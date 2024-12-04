import cv2
from PIL import Image
import numpy as np
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    return square.to(device)


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
        print("-- No chessboards detected! --\n")
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


def classify_pieces(piece_classifier: ResNet, squares: np.ndarray, user_color: str) -> np.ndarray:
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

    # -- If Black is on bottom (user_color == 'b'), flip the board
    if user_color == 'b':
        pieces = pieces[::-1, ::-1]

    return pieces


def array_to_fen(pieces: np.ndarray):
    '''
    Converts numpy array of pieces to a FEN position string

    Parameters:
    - pieces (8, 8): Chessboard position

    Returns:
    - fen_position: String of FEN notation compatible with Stockfish
    '''
    # -- Piece Notation
    fen_position = ""

    for row in range(pieces.shape[0]):
        num_empty = 0

        for piece in pieces[row]:
            # Count consecutive empty squares
            if piece == ' ':
                num_empty += 1
            # On an initial non-empty square, record num_empty first
            elif num_empty > 0:
                fen_position += str(num_empty)
                fen_position += piece
                num_empty = 0
            # Otherwise, simply record piece notation
            else:
                fen_position += piece

        # Include final consecutive empty squares
        if num_empty > 0:
            fen_position += str(num_empty)

        # Separate rows with slashes (/)
        if row != 7:
            fen_position += '/'

    return fen_position


def fen_to_array(fen: str):
    '''
    Converts string FEN board position to a numpy array where 
    pieces are written in FEN and empty squares are ' '

    Parameters:
    - fen: String of FEN notation compatible with Stockfish

    Returns:
    - pieces (8, 8): Chessboard position
    '''
    # -- Extract board position from FEN
    fen_position = fen.split(' ')[0]
    fen_rows = fen_position.split('/')

    # -- Initialize empty pieces array
    pieces = np.full((8, 8), fill_value=' ', dtype=object)

    # -- Copy FEN position to pieces array
    for row_idx, row in enumerate(fen_rows):
        # Character index of row string
        chr_idx = 0
        # Column index of pieces array
        col_idx = 0

        while col_idx < 8:
            piece = row[chr_idx]
            # A letter implies a piece
            if piece.isalpha():
                pieces[row_idx][col_idx] = piece
                col_idx += 1
            # A digit n implies n consecutive empty squares
            elif piece.isdigit():
                pieces[row_idx][col_idx : col_idx + int(piece)] = [' '] * int(piece)
                col_idx += int(piece)
            # Increment through row string
            chr_idx += 1

    return pieces


def indices_to_alg(indices: tuple):
    '''
    Converts board indices ([0, 7], [0, 7]) to algebraic notation ([a, h], [8, 1])
    - e.g. (4, 4) -> "e4"

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


def infer_castling(pieces: np.ndarray):
    '''
    Given a board state as an array, infer if White/Black have
    kingside/queenside castling rights based on the positions of 
    their kings and rooks
    - If King/Rook(s) are in initial positions, they have castling rights
    '''
    w_castle = ""
    b_castle = ""

    # -- White
    if pieces[7][4] == 'K':
        if pieces[7][7] == 'R':
            w_castle += 'K'
        if pieces[7][0] == 'R':
            w_castle += 'Q'

    # -- Black
    if pieces[0][4] == 'k':
        if pieces[0][7] == 'r':
            b_castle += 'k'
        if pieces[0][0] == 'r':
            b_castle += 'q'

    if not w_castle and not b_castle:
        return '-'
    else:
        return w_castle + b_castle


def infer_move(pieces_1: np.ndarray, pieces_2: np.ndarray, active_color: str):
    '''
    Given two board states with a single move in between, infers the
    intermediate move and returns a string in long algebraic notation:
    - Move:
        * "<start_pos><end_pos>"

    Parameters:
    - pieces_1 (8, 8): Board position preceding the intermediate move
    - pieces_2 (8, 8): Board position following the intermediate move

    Returns:
    - move: String of the starting/ending positions of the moved piece
    '''
    # -- If the two boards are identical, return 'none'
    if (pieces_1 == pieces_2).all():
        return 'none'

    # -- Define inactive color
    inactive_color = 'b' if active_color == 'w' else 'w'

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
        # Define rook notation for the inferred (inactive) color
        if inactive_color == 'w':
            rook = 'R'
        elif inactive_color == 'b':
            rook = 'r'

        # Determine start/end rook positions
        rook_start = np.argwhere((pieces_1 == rook) & (pieces_2 == ' '))
        rook_end   = np.argwhere((pieces_1 == ' ') & (pieces_2 == rook))
        
        # If there are two moved pieces but no rooks, there are multiple unknown moves
        if rook_start.size == 0 or rook_end.size == 0:
            return 'mult'

        # Determine number of rook moves
        rook_moves = abs(rook_end[0][1] - rook_start[0][1])

        # Kingside castle if rook_moves = 2
        if rook_moves == 2:
            if inactive_color == 'w':
                move = 'e1g1'
            elif inactive_color == 'b':
                move = 'e8g8'
        # Queenside castle if rook_moves = 3
        elif rook_moves == 3:
            if inactive_color == 'w':
                move = 'e1c1'
            elif inactive_color == 'b':
                move = 'e8c8'

    # -- 3+ empty squares means there was multiple moves
    else:
        return 'mult'

    return move


def update_stockfish(stockfish: Stockfish,
                     pieces: np.ndarray, 
                     active_color: str,
                     user_color: str):
    '''
    Given the array of classified pieces, update stockfish's internal gamestate

    Returns:
    - apply_move (bool): Boolean stating if the move produced by Stockfish should
    be applied to Stockfish's gamestate, i.e. if Stockfish is certain to be updated fully
        * False if opponent is active_color (can't know their real move)
        * 
    '''
    # -- Read Stockfish FEN as string and array
    fen = stockfish.get_fen_position()
    fen_pieces = fen_to_array(fen)

    # -- If it is the opponent's move, do not apply_move to Stockfish (real move is unknown)
    if active_color != user_color:
        return False
    
    # -- If it is the first move (fullmove=1, active_color=White, fen_pieces & pieces match), apply_move to Stockfish
    if (fen.split(' ')[-1] == '1' and active_color == 'w' and (fen_pieces==pieces).all()):
        return True
    
    # -- Otherwise, you must update stockfish by inferring the opponent's move
    opp_move = infer_move(pieces_1=fen_pieces,
                          pieces_2=pieces,
                          active_color=active_color)
    
    # -- If there is no change in the board, it must be a double run (do not apply_move)
    if opp_move == 'none':
        return False
    
    # -- Multiple moves so inferring isn't possible, update Stockfish position/castling and apply_move to Stockfish
    elif opp_move == 'mult':
        fen_position = array_to_fen(pieces)
        castling = infer_castling(pieces)

        stockfish.set_fen_position(f"{fen_position} {active_color} {castling} - 0 1")

        return True
    
    # -- If a move can be inferred, update Stockfish's game state and apply_move 
    else:
        stockfish.make_moves_from_current_position([opp_move])

        return True


def run_pipeline(board_detector: YOLO, 
                 piece_classifier: ResNet, 
                 stockfish: Stockfish,
                 active_color: str,
                 user_color: str):
    '''
    Detect Chessboard
    '''
    print("Detecting chessboard...\n")

    # -- Take screenshot of primary monitor
    screenshot = np.array(pyautogui.screenshot())

    # -- Detect chessboard from screenshot
    board = detect_board(board_detector, screenshot)

    if board is None:
        return None, None

    '''
    Classify Pieces
    '''
    print("Classifying pieces...\n")

    # -- Segment squares from detected chessboard
    squares = segment_board(board)

    # -- Classify the piece of each square in FEN notation
    pieces = classify_pieces(piece_classifier, squares, user_color)

    '''
    Recommend Move
    '''
    print("Recommending a move...\n")

    # -- If applicable, update Stockfish to the current position by inferring the opponent's move
    apply_move = update_stockfish(stockfish, pieces, active_color, user_color)

    # -- Generate the best move from Stockfish
    move = stockfish.get_best_move()

    # -- If apply_move == True, apply the move to Stockfish's game state
    if apply_move:
        stockfish.make_moves_from_current_position([move])
        print(f"Updated Stockfish Board:\n{stockfish.get_board_visual(perspective_white=(user_color=='w'))}")

    return move, pieces


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
    stockfish = Stockfish(
        path='models/stockfish/stockfish-windows-x86-64-avx2.exe',
        depth=24,
        parameters={
            "Threads" : 4,
            "Hash" : 2048,
            "Skill Level" : 20
        }
    )

    print("Models successfully loaded.\n")

    '''
    Run pipeline on Enter, exit on Escape
    '''
    user_color = input('Input your color (w/b): ')

    print(f"\n-------- You are playing as ({'White' if user_color == 'w' else 'Black'}) --------\n")

    print(f"-- Hit <Enter> to recommend a move for yourself ({'White' if user_color == 'w' else 'Black'})") 
    print(f"-- Hit <Space> to recommend a move for your opponent ({'Black' if user_color == 'w' else 'White'})\n")

    while True:
        # -- If a key is pressed...
        if msvcrt.kbhit():
            key = msvcrt.getwch().lower()
            # Hit Enter to run pipeline for user_color
            if key == '\r':
                print(f"-------- {{Fullmove {stockfish.get_fen_position().split(' ')[-1]}}} --------\n")

                move = run_pipeline(board_detector, 
                                    piece_classifier, 
                                    stockfish,
                                    active_color=user_color,
                                    user_color=user_color)
                
                print(f"----> {'White' if user_color == 'w' else 'Black'} Move: {move}\n")
            # Hit Space to run pipeline for opponent's color
            elif key == ' ':
                print(f"-------- {{Fullmove {stockfish.get_fen_position().split(' ')[-1]}}} --------\n")

                move = run_pipeline(board_detector, 
                                    piece_classifier, 
                                    stockfish,
                                    active_color='b' if user_color == 'w' else 'w',
                                    user_color=user_color)
                
                print(f"----> {'Black' if user_color == 'w' else 'White'} Move: {move}\n")
            # Exit program on Escape
            elif key == '\x1b':
                break


if __name__ == "__main__":
    main()