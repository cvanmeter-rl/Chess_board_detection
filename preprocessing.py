import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def segment_square(src,des):
    #change to test or train depending on what you want
    path = f'./archive/{src}'

    #get list of all images in the folder
    files = os.listdir(path)
    images_path = os.path.join(path)

    #iterate through all images
    for f in files:
        images_path = os.path.join(path,f)
        image = Image.open(images_path)
        #get the filename and split it based on the notation for each row
        filename = os.path.basename(image.filename)

        split = filename[:-5].split("-")

        labels = []
        for r in split:
            rows = []
            for c in r:
                if c.isdigit():
                    empty = int(c)
                    while empty != 0:
                        rows.append('empty')
                        empty -= 1
                else:
                    rows.append(c)
            labels.append(rows)

        labels = np.array(labels)

        image = np.array(image) 

        r = 0
        c = 0

        #iterate through each image and segment each square, rename the image to the row and column it belongs to along with it's board notation and store it in its respective folder
        #class 1 = empty square, 2 = black bishop, 3 = black king, 4 = black knight, 5 = black pawn, 6 = black queen, and 7 = black rook, 8 = white bishop, 9 = white king, 10 = white knight
        #11 = white pawn, 12 = white queen, and 13 = white rook
        for i in range(0,8):
            rows = []
            for j in range(0,8):
                square_image = Image.fromarray(image[r:r+50,c:c+50].astype(np.uint8))
                newFilename = f'{labels[i,j]}_{i}_{j}_{filename[:-5]}.jpeg'
                print(labels[i,j])

                if labels[i,j] == 'empty':
                    square_image.save(os.path.join(f'./archive/{des}/class_1',newFilename))
                elif labels[i,j] == 'p':
                    square_image.save(os.path.join(f'./archive/{des}/class_5',newFilename))
                elif labels[i,j] == 'P':
                    square_image.save(os.path.join(f'./archive/{des}/class_11',newFilename))
                elif labels[i,j] == 'n':
                    square_image.save(os.path.join(f'./archive/{des}/class_4',newFilename))
                elif labels[i,j] == 'N':
                    square_image.save(os.path.join(f'./archive/{des}/class_10',newFilename))
                elif labels[i,j] == 'b':
                    square_image.save(os.path.join(f'./archive/{des}/class_2',newFilename))
                elif labels[i,j] == 'B':
                    square_image.save(os.path.join(f'./archive/{des}/class_8',newFilename))
                elif labels[i,j] == 'r':
                    square_image.save(os.path.join(f'./archive/{des}/class_7',newFilename))
                elif labels[i,j] == 'R':
                    square_image.save(os.path.join(f'./archive/{des}/class_13',newFilename))
                elif labels[i,j] == 'q':
                    square_image.save(os.path.join(f'./archive/{des}/class_6',newFilename))
                elif labels[i,j] == 'Q':
                    square_image.save(os.path.join(f'./archive/{des}/class_12',newFilename))
                elif labels[i,j] == 'k':
                    square_image.save(os.path.join(f'./archive/{des}/class_3',newFilename))
                elif labels[i,j] == 'K':
                    square_image.save(os.path.join(f'./archive/{des}/class_9',newFilename))
                c += 50
                
            r += 50
            c = 0
    image.close()
        


    


segment_square('train','segmented_train') 
#segment_square('test','segmented_test')