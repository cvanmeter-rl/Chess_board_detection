from ultralytics import YOLO
import cv2
import os

def custom_test(weights_name):
    # -- Load YOLO model
    model = YOLO(f'models/board_detector/{weights_name}.pt')

    # -- Set annotations/test images directories
    anns_save_dir = f'custom_testing/annotations/{weights_name}'
    test_imgs_dir = f'custom_testing/test_images'
    os.makedirs(anns_save_dir, exist_ok=True)

    # -- Run inference on all test images and save annotations
    for i in range(len(os.listdir(test_imgs_dir))):
        test_img_path = os.path.join(test_imgs_dir, f'test_img_{i}.png')

        # -- Perform inference on a test image
        results = model(test_img_path)

        # -- Save annotated image
        r = results[0]
        annotated_img = r.plot()
        
        cv2.imwrite(os.path.join(anns_save_dir, f'{weights_name}_test_img_{i}.png'), annotated_img)

def test(weights_name):
    # -- Load YOLO model
    model = YOLO(f'models/board_detector/{weights_name}.pt')

    # -- Test on specified split (test/val)
    split = 'test'

    results = model.val(
        data='data.yaml',
        split=split,
        name=f'{split}_{weights_name}'
    )

    # -- Print results
    results_dict = results.result_dict

    for metric, value in results_dict.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    weights_name = 'epoch27'

    custom_test(weights_name)
