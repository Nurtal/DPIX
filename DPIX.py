import matplotlib
matplotlib.use('TkAgg')
import os
import getopt
import sys


import representation
import preprocessing
import classification
import manager


def main(argv):
	##
	## IN PROGRESS
	##
	## TODO : write help
	##
	##


	action = "NA"


	try:
		opts, args = getopt.getopt(argv,"ha:",["action="])

	except getopt.GetoptError:
		sys.exit(2)

	for opt, arg in opts:
		
		## Display Help
		if opt == '-h':
			print "choucroute"
			sys.exit()

		## Get action
		elif opt in ("-a", "--action"):
			action = arg

	if(action == "exemple"):

		## test
		## Prepare to predict data - give an empty array
		prediction_dataset = []

		## Build image map and save it
		preprocessing.reformat_input_datasets("datasets/creditcard_reduce.csv", 30, True)
		preprocessing.normalize_data("datasets/creditcard_reduce_reformated.csv")
		image_structure = representation.build_image_map("datasets/creditcard_reduce_reformated_scaled.csv", 5)
		manager.save_matrix_to_file(image_structure, "datasets/credit_image_structure.csv")
		

		## prepare train data
		representation.simple_conversion_to_img_matrix("datasets/creditcard_reduce_reformated_scaled.csv")
		representation.build_patient_representation("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
		real_data = representation.build_patient_matrix("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
		(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)
		
		## Run CNN
		classification.run_CNN(train_X, train_Y, test_X, test_Y, 20, prediction_dataset)
		
		## write report
		manager.write_report()





##------##
## MAIN ###################################################################
##------##


if __name__ == '__main__':
	main(sys.argv[1:])
