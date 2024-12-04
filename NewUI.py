import sys
from PyQt5.QtWidgets import (QLabel, QGridLayout, QApplication, QMainWindow, QPushButton, QTextEdit,
                             QVBoxLayout, QWidget, QHBoxLayout, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

import pipeline

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

current_color = 'w' #keeps track of users piece color

black_side = [
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
]

white_side = [
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],
]

class ChessBoardWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)

        # Remove gaps between rows and columns
        self.grid_layout.setSpacing(0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)

        self.squares = [[None for _ in range(8)] for _ in range(8)]
        self.create_board()

        # Set fixed size to prevent expansion
        self.setFixedSize(480, 480)  # 8 squares * 60 pixels each
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Set initial board
        self.update_board(white_side)

    def create_board(self):
        """Initialize the 8x8 chessboard with alternating colors."""
        for row in range(8):
            for col in range(8):
                square = QLabel()
                square.setFixedSize(60, 60)
                square.setFrameStyle(QLabel.NoFrame)

                # Alternate square colors
                if (row + col) % 2 == 0:
                    square.setStyleSheet("""
                        background-color: white;
                        border: none;
                        margin: 0px;
                        padding: 0px;
                    """)
                else:
                    square.setStyleSheet("""
                        background-color: gray;
                        border: none;
                        margin: 0px;
                        padding: 0px;
                    """)

                # Add square to the grid layout
                self.grid_layout.addWidget(square, row, col)
                self.squares[row][col] = square

    def update_board(self, board):
        """
        Update the chessboard based on the given board positions.

        :param board: A 2D list representing the board. Example:
                      [['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
                       ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
                       [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                       ...]
        """
        piece_images = {
            'r': "UI_chess_images/br.png",
            'n': "UI_chess_images/bn.png",
            'b': "UI_chess_images/bb.png",
            'q': "UI_chess_images/bq.png",
            'k': "UI_chess_images/bk.png",
            'p': "UI_chess_images/bp.png",
            'R': "UI_chess_images/wr.png",
            'N': "UI_chess_images/wn.png",
            'B': "UI_chess_images/wb.png",
            'Q': "UI_chess_images/wq.png",
            'K': "UI_chess_images/wk.png",
            'P': "UI_chess_images/wp.png",
        }

        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                square = self.squares[row][col]
                
                if (piece != ' '):
                    # Load the corresponding piece image
                    pixmap = QPixmap(piece_images[piece])
                    pixmap = pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # Adjusted size
                    square.setPixmap(pixmap)
                    square.setAlignment(Qt.AlignCenter)  # Center both horizontally and vertically
                else:
                    square.clear()  # Clear the square if empty

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle("Chess Move Recommender")
        self.setGeometry(800, 400, 600, 800)  # Increased height

        self.setMinimumSize(600, 800)  # Allow the window to be at least 600x800

        # Create main widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)  # Added margins for better spacing
        main_layout.setSpacing(10)  # Space between widgets
        central_widget.setLayout(main_layout)

        # Create toggle button layout
        toggle_layout = QHBoxLayout()
        self.toggle_button = QPushButton("White Pieces", self)
        self.toggle_button.setCheckable(True)  # Enable toggle functionality
        self.toggle_button.setStyleSheet("background-color: white; color: black;border: 1px solid #555555;")  # Initial color
        self.toggle_button.setFixedWidth(200)
        toggle_layout.addStretch()  # Add stretch to center the toggle button horizontally
        toggle_layout.addWidget(self.toggle_button)
        toggle_layout.addStretch()
        main_layout.addLayout(toggle_layout)

        # Create move buttons layout
        self.oppPredMoveButton = QPushButton("Predict Opponent Move")
        self.oppPredMoveButton.setFixedHeight(75)

        self.predMoveButton = QPushButton("Get Recommended Move")
        self.predMoveButton.setFixedHeight(75)

        # Create horizontal layout for move buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.predMoveButton)
        button_layout.addWidget(self.oppPredMoveButton)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)

        # Add chessboard widget with centering
        chessboard_layout = QHBoxLayout()
        chessboard_layout.addStretch()
        self.chessboard = ChessBoardWidget()
        chessboard_layout.addWidget(self.chessboard)
        chessboard_layout.addStretch()
        main_layout.addLayout(chessboard_layout)

        # Create output area (read-only text area)
        reccMove_boxes_layout = QHBoxLayout()
        self.move1 = QTextEdit(self)
        self.move1.setReadOnly(True)  # Make the text area read-only
        self.move1.setFixedHeight(100)
        self.move1.setPlaceholderText("1st Move:")
        reccMove_boxes_layout.addWidget(self.move1) 

        self.move2 = QTextEdit(self)
        self.move2.setReadOnly(True)  # Make the text area read-only
        self.move2.setFixedHeight(100)
        self.move2.setPlaceholderText("2nd Move:")
        reccMove_boxes_layout.addWidget(self.move2) 

        self.move3 = QTextEdit(self)
        self.move3.setReadOnly(True)  # Make the text area read-only
        self.move3.setFixedHeight(100) 
        self.move3.setPlaceholderText("3rd Move:")
        reccMove_boxes_layout.addWidget(self.move3)

        main_layout.addLayout(reccMove_boxes_layout)

        # Connect buttons to methods
        self.oppPredMoveButton.clicked.connect(self.getOppPredictedMove)
        self.predMoveButton.clicked.connect(self.getPredictedMove)

        # Connect toggle signal
        self.toggle_button.toggled.connect(self.toggle_state)

    def toggle_state(self, checked):

        if checked:  # Black pieces
            current_color = 'b'
            self.chessboard.update_board(black_side)
            self.toggle_button.setText("Black Pieces")
            self.toggle_button.setStyleSheet("background-color: #3C3F41; color: white;border: 1px solid #555555;")  # Updated color
        else:  # White pieces
            current_color = 'w'
            self.chessboard.update_board(white_side)
            self.toggle_button.setText("White Pieces")
            self.toggle_button.setStyleSheet("background-color: white; color: black;border: 1px solid #555555;")  # Initial color

    # Methods for button actions
    def getOppPredictedMove(self):
        stockfish.get_fen_position().split(' ')[-1]

        move,pieces = pipeline.run_pipeline(board_detector, 
                                     piece_classifier, 
                                     stockfish,
                                     active_color='b' if current_color == 'w' else 'w',
                                     user_color=current_color)

        if pieces is not None:
            self.chessboard.update_board(pieces)

            self.move1.setText(move)
            self.move2.setText(move)
            self.move3.setText(move)

    def getPredictedMove(self):
        stockfish.get_fen_position().split(' ')[-1]

        move,pieces = pipeline.run_pipeline(board_detector,
                                     piece_classifier, 
                                     stockfish,
                                     active_color=current_color,
                                     user_color=current_color)

        if pieces is not None:
            self.chessboard.update_board(pieces)

            self.move1.setText(move)
            self.move2.setText(move)
            self.move3.setText(move)


if __name__ == "__main__":
    '''
    Load Models
    '''
    print("Loading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Load YOLO11n chessboard detection model (epoch27.pt)
    board_detector = YOLO(f'models/board_detector/epoch27.pt')
    board_detector = board_detector.to(device)
    # -- Load ResNet-18 piece classification model (classifier_epoch1.pth)
    
    piece_classifier = torch.load(f'models/piece_classifier/classifier_epoch1.pth', map_location=device)
    piece_classifier = piece_classifier.to(device)
    piece_classifier.eval()

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


    app = QApplication(sys.argv)
    # Apply a dark theme using QSS
    dark_style = """
    QMainWindow {
        background-color: #2B2B2B;
        color: #FFFFFF;
    }
    QPushButton {
        background-color: #3C3F41;
        color: #FFFFFF;
        border: 1px solid #555555;
        padding: 5px;
    }
    QPushButton:hover {
        background-color: #505050;
    }
    QTextEdit {
        background-color: #3C3F41;
        color: #FFFFFF;
        border: 1px solid #555555;
    }
    """
    app.setStyleSheet(dark_style)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
