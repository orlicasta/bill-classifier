import os
import random
import csv
#import shutil
#from datetime import datetime

# from all subfolders of directory '{cwd}/sets/' add file name and classification to a list, randomize, the write to a csv

#$ py -u generateFileList.py
#atmos Count is:  1098
#att Count is:  1169
#capital one Count is:  1083
#lowe Count is:  875
#kohl Count is:  1264

#TOD-DO ##########################
#train set of 800 each atmos, att, capitalone, kohl, other
#test set of 200 each atmos, att, capitalone, kohl, other


def main():

	# class 1
	atmosCount = 0
	# class 2
	attCount = 0
	# class 3
	capitaloneCount = 0
	# class 4
	kohlCount = 0

	folder = os.getcwd() + "/sets"

	fileNameList = []

	for subfolder in os.listdir(folder):
		for fileName in os.listdir(folder + "/" + subfolder):
			if "atmos" in fileName.lower():
				atmosCount = atmosCount + 1
				fileNameList.append(["sets/" + subfolder + "/" + fileName, 1])
			elif "att" in fileName.lower() or "at&t" in fileName.lower():
				attCount = attCount + 1
				fileNameList.append(["sets/" + subfolder + "/" + fileName, 2])
			elif "capital one" in fileName.lower():
				capitaloneCount= capitaloneCount + 1
				fileNameList.append(["sets/" + subfolder + "/" + fileName, 3])
			elif "kohl" in fileName.lower():
				kohlCount= kohlCount + 1
				fileNameList.append(["sets/" + subfolder + "/" + fileName, 4])
			else:
				fileNameList.append(["sets/" + subfolder + "/" + fileName, 0])				

	print("atmos Count is: ", atmosCount)
	print("att Count is: ", attCount)
	print("capital one Count is: ", capitaloneCount)
	print("kohl Count is: ", kohlCount)

	random.shuffle(fileNameList)

	atmosTrain = []
	attTrain = []
	capitaloneTrain = []
	kohlTrain = []
	otherTrain = []

	atmosTest = []
	attTest = []
	capitaloneTest = []
	kohlTest = []
	otherTest = []

	foundCount = 0
	for item in fileNameList:
		if item[1] == 1 and foundCount < 800:
			foundCount = foundCount + 1
			atmosTrain.append(item)
		elif item[1] == 1 and foundCount < 1000:
			foundCount = foundCount + 1
			atmosTest.append(item)

	foundCount = 0
	for item in fileNameList:
		if item[1] == 2 and foundCount < 800:
			foundCount = foundCount + 1
			attTrain.append(item)
		elif item[1] == 2 and foundCount < 1000:
			foundCount = foundCount + 1
			attTest.append(item)

	foundCount = 0
	for item in fileNameList:
		if item[1] == 3 and foundCount < 800:
			foundCount = foundCount + 1
			capitaloneTrain.append(item)
		elif item[1] == 3 and foundCount < 1000:
			foundCount = foundCount + 1
			capitaloneTest.append(item)

	foundCount = 0
	for item in fileNameList:
		if item[1] == 4 and foundCount < 800:
			foundCount = foundCount + 1
			kohlTrain.append(item)
		elif item[1] == 4 and foundCount < 1000:
			foundCount = foundCount + 1
			kohlTest.append(item)

	foundCount = 0
	for item in fileNameList:
		if item[1] == 0 and foundCount < 800:
			foundCount = foundCount + 1
			otherTrain.append(item)
		elif item[1] == 0 and foundCount < 1000:
			foundCount = foundCount + 1
			otherTest.append(item)

	trainList = atmosTrain + attTrain + capitaloneTrain + kohlTrain + otherTrain
	testList = atmosTest + attTest + capitaloneTest + kohlTest + otherTest

	random.shuffle(trainList)
	random.shuffle(testList)

	# backup existing csv files
	#backupDir = os.getcwd() + "/annotationsbackup"
	#d = datetime.now()
	#d = d.strftime('_%Y-%m-%d_%H-%M-%S')
	#shutil.copy('train.csv', backupDir + '/fullRandom' + d + '.csv')
	#shutil.copy('test.csv', backupDir + '/fullRandom' + d + '.csv')

	with open('train.csv', 'w', newline='') as file:
		wrtr = csv.writer(file)
		wrtr.writerow(['dir', 'label'])
		for entry in trainList:
			wrtr.writerow(entry)

	with open('test.csv', 'w', newline='') as file:
		wrtr = csv.writer(file)
		wrtr.writerow(['dir', 'label'])
		for entry in testList:
			wrtr.writerow(entry)


# Driver Code
if __name__ == '__main__':
	
	# Calling main() function
	main()
