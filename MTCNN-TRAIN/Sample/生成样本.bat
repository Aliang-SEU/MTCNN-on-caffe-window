if exist mtcnn_train_12 rd /q /s mtcnn_train_12 
if exist mtcnn_train_24 rd /q /s mtcnn_train_24 
if exist mtcnn_train_48 rd /q /s mtcnn_train_48
start python gen_data_random_12.py 
start python gen_data_random_24.py 
start python gen_data_random_48.py 