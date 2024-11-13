import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import gc
import shutil

def segment_square(src,des):
    #change to test or train depending on what you want
    path = f'./archive/{src}'

    #get list of all images in the folder
    files = os.listdir(path)

    # Map labels to folder paths for quick access
    class_paths = {
        'empty': os.path.join(f'./archive/{des}/class_1'),
        'p': os.path.join(f'./archive/{des}/class_5'),
        'P': os.path.join(f'./archive/{des}/class_11'),
        'n': os.path.join(f'./archive/{des}/class_4'),
        'N': os.path.join(f'./archive/{des}/class_10'),
        'b': os.path.join(f'./archive/{des}/class_2'),
        'B': os.path.join(f'./archive/{des}/class_8'),
        'r': os.path.join(f'./archive/{des}/class_7'),
        'R': os.path.join(f'./archive/{des}/class_13'),
        'q': os.path.join(f'./archive/{des}/class_6'),
        'Q': os.path.join(f'./archive/{des}/class_12'),
        'k': os.path.join(f'./archive/{des}/class_3'),
        'K': os.path.join(f'./archive/{des}/class_9')
    }

    count = 1
    #iterate through all images
    for f in files[21560:]:
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
                    rows.extend(['empty'] * int(c))
                else:
                    rows.append(c)
            labels.append(rows)

        labels = np.array(labels)

        imageArr = np.array(image) 

        r = 0
        c = 0

        #iterate through each image and segment each square, rename the image to the row and column it belongs to along with it's board notation and store it in its respective folder
        #class 1 = empty square, 2 = black bishop, 3 = black king, 4 = black knight, 5 = black pawn, 6 = black queen, and 7 = black rook, 8 = white bishop, 9 = white king, 10 = white knight
        #11 = white pawn, 12 = white queen, and 13 = white rook
        #due to the large amount of files this function creates additional batch folders to improve runtime
        for i in range(0,8):
            rows = []
            for j in range(0,8):
                square_image = Image.fromarray(imageArr[r:r+50,c:c+50].astype(np.uint8))
                newFilename = f'{labels[i,j]}_{i}_{j}_{filename[:-5]}.jpeg'

                batch_num = (count // 10000) + 1
                batch_path = os.path.join(class_paths[labels[i, j]], f'batch_{batch_num}')
                os.makedirs(batch_path,exist_ok=True)

                square_image.save(os.path.join(batch_path, newFilename))

                c += 50
                
            r += 50
            c = 0
        print(count)
        count += 1
        image.close()
        

        # Delete variables and force garbage collection periodically
        if count % 100 == 0:
            del imageArr, square_image
            gc.collect()


def move_files_to_class_folder(class_dir):
    count = 1
    # Traverse through all batch subdirectories within the class directory
    for root, dirs, files in os.walk(class_dir):
        # Only move files if the root is a subdirectory (i.e., batch folder) of the class directory
        if root != class_dir:
            for file in files:
                # Construct full file path in both source and destination
                file_path = os.path.join(root, file)
                dest_path = os.path.join(class_dir, file)
                
                # Check if the file already exists in the class directory
                if os.path.exists(dest_path):
                    # If it exists in the class folder, delete the one in the batch folder
                    os.remove(file_path)
                else:
                    print(count)
                    count += 1
                    # Otherwise, move the file to the class directory
                    shutil.move(file_path, dest_path)
            # After moving files, remove the empty batch subdirectory
            os.rmdir(root)






#segment_square('train','segmented_train') #21560
#segment_square('test','segmented_test')

class_directories = [
    './archive/segmented_train/class_1',
    './archive/segmented_train/class_2',
    './archive/segmented_train/class_3',
    './archive/segmented_train/class_4',
    './archive/segmented_train/class_5',
    './archive/segmented_train/class_6',
    './archive/segmented_train/class_7',
    './archive/segmented_train/class_8',
    './archive/segmented_train/class_9',
    './archive/segmented_train/class_10',
    './archive/segmented_train/class_11',
    './archive/segmented_train/class_12',
    './archive/segmented_train/class_13'
]
for class_dir in class_directories:
    move_files_to_class_folder(class_dir)