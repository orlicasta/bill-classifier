import os

epochs = 4

os.system("py -u congruentgen.py")
os.system("py -u xceptionTrain.py " + str(epochs))
os.system("py -u xceptionTest.py")

os.system("py -u congruentgen.py")
os.system("py -u xceptionTrain.py " + str(epochs))
os.system("py -u xceptionTest.py")

os.system("py -u congruentgen.py")
os.system("py -u res18Train.py " + str(epochs))
os.system("py -u res18Test.py")

os.system("py -u congruentgen.py")
os.system("py -u xceptionTrain.py " + str(epochs))
os.system("py -u xceptionTest.py")

os.system("py -u congruentgen.py")
os.system("py -u res18Train.py " + str(epochs))
os.system("py -u res18Test.py")

os.system("py -u congruentgen.py")
os.system("py -u xceptionTrain.py " + str(epochs))
os.system("py -u xceptionTest.py")

os.system("py -u congruentgen.py")
os.system("py -u res18Train.py " + str(epochs))
os.system("py -u res18Test.py")

os.system("py -u congruentgen.py")
os.system("py -u xceptionTrain.py " + str(epochs))
os.system("py -u xceptionTest.py")

quit()

for n in range(4):
    os.system("py -u congruentgen.py")
    os.system("py -u alexTrain.py " + str(epochs))
    os.system("py -u alexTest.py")

for n in range(4):
    os.system("py -u congruentgen.py")
    os.system("py -u vgg11Train.py " + str(epochs))
    os.system("py -u vgg11Test.py")

for n in range(4):
    os.system("py -u congruentgen.py")
    os.system("py -u xceptionTrain.py " + str(epochs))
    os.system("py -u xceptionTest.py")

for n in range(4):
    os.system("py -u congruentgen.py")
    os.system("py -u res18Train.py " + str(epochs))
    os.system("py -u res18Test.py")



epochs = 6

for n in range(4):
    os.system("py -u congruentgen.py")
    os.system("py -u alexTrain.py " + str(epochs))
    os.system("py -u alexTest.py")

for n in range(4):
    os.system("py -u congruentgen.py")
    os.system("py -u vgg11Train.py " + str(epochs))
    os.system("py -u vgg11Test.py")

for n in range(4):
    os.system("py -u congruentgen.py")
    os.system("py -u xceptionTrain.py " + str(epochs))
    os.system("py -u xceptionTest.py")

for n in range(4):
    os.system("py -u congruentgen.py")
    os.system("py -u res18Train.py " + str(epochs))
    os.system("py -u res18Test.py")
