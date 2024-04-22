Necessary packets:

pip install opencv-python
pip install deepface     
pip install scikit-image
pip install scikit-learn

Image set: https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k


Concerning cell 26:
# Data location
folder = "rvf10k/train"
dest_folder = "roi_dataset"
cascade_path = "haarcascade_frontalface_default.xml"

1. It may be necessary to create the according folder structure.
2. Path to cascade_path has to be set accordingly to the opencv-python package. E.g. cascade_path = "C:/Users/YourUsername/AppData/Local/Programs/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
