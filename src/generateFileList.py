import os
import random
import csv
import shutil
from datetime import datetime

# from all subfolders of directory '{cwd}/sets/' add file name and classification to a list, randomize, the write to a csv

def main():

	# class 1
	atmosCount = 0
	# class 2
	attCount = 0
	# class 3
	capitaloneCount = 0
	# class 4
	loweCount = 0
	# class 5
	kohlCount = 0

	folder = os.getcwd() + "/sets"

	fileNameList = []

	for subfolder in os.listdir(folder):
		for fileName in os.listdir(folder + "/" + subfolder):
			if "directv" in fileName.lower():
				fileNameList.append(["sets/" + subfolder + "/" + fileName, 1])
				atmosCount = atmosCount + 1
			elif "att" in fileName.lower() or "at&t" in fileName.lower():
				attCount = attCount + 1
				fileNameList.append(["sets/" + subfolder + "/" + fileName, 2])
			elif "capital one" in fileName.lower():
				capitaloneCount= capitaloneCount + 1
				fileNameList.append(["sets/" + subfolder + "/" + fileName, 3])
			elif "lowe" in fileName.lower():
				loweCount= loweCount + 1
				fileNameList.append(["sets/" + subfolder + "/" + fileName, 4])
			elif "kohl" in fileName.lower():
				kohlCount= kohlCount + 1
				fileNameList.append(["sets/" + subfolder + "/" + fileName, 5])
			else:
				fileNameList.append(["sets/" + subfolder + "/" + fileName, 0])
				
	
	random.shuffle(fileNameList)

	print("atmos Count is: ", atmosCount)
	print("att Count is: ", attCount)
	print("capital one Count is: ", capitaloneCount)
	print("lowe Count is: ", loweCount)
	print("kohl Count is: ", kohlCount)

	quit()

	# backup existing csv file

	backupDir = os.getcwd() + "/annotationsbackup"
	d = datetime.now()
	d = d.strftime('_%Y-%m-%d_%H-%M-%S')
	shutil.copy('fullRandom.csv', backupDir + '/fullRandom' + d + '.csv')

	with open('fullRandom.csv', 'w', newline='') as file:
		wrtr = csv.writer(file)
		wrtr.writerow(['dir', 'label'])
		for entry in fileNameList:
			wrtr.writerow(entry)



# Driver Code
if __name__ == '__main__':
	
	# Calling main() function
	main()
