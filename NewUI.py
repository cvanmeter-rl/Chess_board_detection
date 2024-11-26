import sys
from PyQt5.QtWidgets import QLabel,QGridLayout,QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget, QHBoxLayout,QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage

black_side = [
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'],  
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],  
    ['.', '.', '.', '.', '.', '.', '.', '.'],  
    ['.', '.', '.', '.', '.', '.', '.', '.'],  
    ['.', '.', '.', '.', '.', '.', '.', '.'],  
    ['.', '.', '.', '.', '.', '.', '.', '.'],  
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],  
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],  
]

white_side = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
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

    def update_board(self,board):
        """
        Update the chessboard based on the given board positions.

        :param board: A 2D list representing the board. Example:
                      [['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
                       ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
                       ['.', '.', '.', '.', '.', '.', '.', '.'],
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

                if piece != '.':
                    # Load the corresponding piece image
                    pixmap = QPixmap(piece_images[piece])
                    pixmap = pixmap.scaled(60, 45, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    square.setPixmap(pixmap)
                    square.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
                else:
                    square.clear()  # Clear the square if empty

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle("Chess Move Recommender")
        self.setGeometry(800, 400, 600, 600) #xpos,ypos,width,height

        self.setFixedSize(600,600) #set window to fixed size, width, height

        # Create main widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Top, Left, Bottom, Right margins
        #layout.setSpacing(5)  # Space between widgets
        layout.addStretch(0)
        central_widget.setLayout(layout)

        #create toggle button
        toggle_layout = QHBoxLayout()
        self.toggle_button = QPushButton("White Pieces", self)
        self.toggle_button.setCheckable(True)  # Enable toggle functionality
        self.toggle_button.setStyleSheet("background-color: white; color: black;border: 1px solid #555555;")  # Initial color
        self.toggle_button.setFixedWidth(200)
        toggle_layout.addWidget(self.toggle_button,alignment=Qt.AlignTop | Qt.AlignHCenter)
        layout.addLayout(toggle_layout)

        #create move buttons
        self.oppPredMoveButton = QPushButton("Predict Opponent Move")
        self.oppPredMoveButton.setFixedHeight(75)

        self.predMoveButton = QPushButton("Get Recommended Move")
        self.predMoveButton.setFixedHeight(75)

        # Create horizontal layout for buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.predMoveButton)
        button_layout.addWidget(self.oppPredMoveButton)
        layout.addLayout(button_layout)

        #add chessboard widget

        self.chessboard = ChessBoardWidget()
        layout.addWidget(self.chessboard)

        self.chessboard.update_board(white_side)

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


        layout.addLayout(reccMove_boxes_layout)

        # Connect buttons to methods
        self.oppPredMoveButton.clicked.connect(self.getOppPredictedMove)
        self.predMoveButton.clicked.connect(self.getPredictedMove)

        # Connect toggle signal
        self.toggle_button.toggled.connect(self.toggle_state)

    def toggle_state(self, checked):
        if checked: #black pieces
            self.chessboard.update_board(black_side)
            self.toggle_button.setText("Black Pieces")
            self.toggle_button.setStyleSheet("background-color: #3C3F41; color: white;border: 1px solid #555555;")  # Initial color
        else: #white pieces
            self.chessboard.update_board(white_side)
            self.toggle_button.setText("White Pieces")
            self.toggle_button.setStyleSheet("background-color: white; color: black;border: 1px solid #555555;")  # Initial color
    # Methods for button actions
    def getOppPredictedMove(self):
        self.move1.setText("You clicked Opp pred move!")
        self.move2.setText("You clicked Opp pred move!")
        self.move3.setText("You clicked Opp pred move!")
    
    def getPredictedMove(self):
        self.move1.setText("You clicked predicted move!")
        self.move2.setText("You clicked predicted move!")
        self.move3.setText("You clicked predicted move!")




if __name__ == "__main__":
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