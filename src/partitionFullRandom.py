import os
import csv
import shutil
from datetime import datetime

def main():

	testSize = 1200
	lineList = []

	with open('fullRandom.csv', 'r', newline='') as file:
		rdr = csv.reader(file)
		for line in rdr:
			lineList.append(line)

	# backup current testing annotations
	backupDir = os.getcwd() + "/annotationsbackup"
	d = datetime.now()
	d = d.strftime('_%Y-%m-%d_%H-%M-%S')
	shutil.copy('test.csv', backupDir + '/test' + d + '.csv')

	with open('test.csv', 'w', newline='') as file:
		wrtr = csv.writer(file)
		for i, entry in enumerate(lineList):
			if i <= testSize:
				wrtr.writerow(entry)

	# backup current training annotations
	shutil.copy('train.csv', backupDir + '/train' + d + '.csv')

	with open('train.csv', 'w', newline='') as file:
		wrtr = csv.writer(file)
		wrtr.writerow(['fileName', 'class'])
		for i, entry in enumerate(lineList):
			if i > testSize:
				wrtr.writerow(entry)

# Driver Code
if __name__ == '__main__':
	
	# Calling main() function
	main()
