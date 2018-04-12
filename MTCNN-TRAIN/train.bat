if exist models-12 rd /q /s models-12 
mkdir models-12
if exist models-24 rd /q /s models-24 
mkdir models-24
if exist models-48 rd /q /s models-48
mkdir models-48

start "training-12" %cd%\train-tools\caffe.exe train --solver=%cd%\solver-12.prototxt
start "training-24" %cd%\train-tools\caffe.exe train --solver=%cd%\solver-24.prototxt
start "training-48" %cd%\train-tools\caffe.exe train --solver=%cd%\solver-48.prototxt