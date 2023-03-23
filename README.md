# BillClassify

<br>1) Sets should be extracted to src/sets/Set X

<br><br>2) Install required packages
<br>In BillClassify directory (containing requirements.txt):
```
pip install -r requirements.txt
```
or
```
py -m pip install -r requirements.txt
```

<br>3) Generate annotations, train, and test
<br>Training requires an argument for number of epochs
```
py -u congruentgen.py
py -u t_alexTrain.py 1
py -u t_alexTest.py
```
