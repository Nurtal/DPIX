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
		image_structure = representation.build_image_map_GMA("datasets/creditcard_reduce_reformated_scaled.csv", 100)
		manager.save_matrix_to_file(image_structure, "datasets/credit_image_structure.csv")

		## prepare train data
		representation.simple_conversion_to_img_matrix("datasets/creditcard_reduce_reformated_scaled.csv")
		representation.build_patient_representation("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
		real_data = representation.build_patient_matrix("datasets/creditcard_reduce_reformated_scaled_interpolated.csv", image_structure)
		(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)

		## Run CNN
		classification.run_CNN(train_X, train_Y, test_X, test_Y, 20, prediction_dataset, image_structure)

		## write report
		manager.write_report()

	elif(action == "test"):
		print "Enter the Test Area"

		print "[TEST] Overclocked files"
		prediction_dataset = []
		image_structure = manager.load_matrix_from_file("datasets/creditcard_reduce_overclocked_reformated_scaled_saved_matrix.csv")
		representation.simple_conversion_to_img_matrix("datasets/creditcard_reduce_overclocked_reformated_scaled.csv")
		representation.build_patient_representation("datasets/creditcard_reduce_overclocked_reformated_scaled_interpolated.csv", image_structure)
		real_data = representation.build_patient_matrix("datasets/creditcard_reduce_overclocked_reformated_scaled_interpolated.csv", image_structure)
		(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)
		classification.run_CNN(train_X, train_Y, test_X, test_Y, 20, prediction_dataset, image_structure)



	elif(action == "rush"):

		## Prepare to predict data - give an empty array
		prediction_dataset = []

		## Build image map and save it
		preprocessing.reformat_input_datasets("datasets/rnaseq_data.csv", 0, True)
		preprocessing.normalize_data("datasets/rnaseq_data_reformated.csv")
		image_structure = representation.build_image_map_GMA("datasets/rnaseq_data_reformated_scaled.csv", 100)
		manager.save_matrix_to_file(image_structure, "datasets/rnaseq_image_structure.csv")

		## prepare train data
		representation.simple_conversion_to_img_matrix("datasets/rnaseq_reformated_scaled.csv")
		representation.build_patient_representation("datasets/rnaseq_reformated_scaled_interpolated.csv", image_structure)
		real_data = representation.build_patient_matrix("datasets/rnaseq_reformated_scaled_interpolated.csv", image_structure)
		(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)

		## Run CNN
		classification.run_CNN(train_X, train_Y, test_X, test_Y, 50, prediction_dataset, image_structure)

		## write report
		manager.write_report()

	elif(action == "HLA"):


		## pre-set for HLA data

		## Prepare to predict data - give an empty array
		prediction_dataset = []

		preprocessing.reformat_input_datasets("datasets/HLA_data_clean.csv", 562, True)
		preprocessing.normalize_data("datasets/HLA_data_clean_reformated.csv")
		image_structure = representation.build_image_map_GMA("datasets/HLA_data_clean_reformated_scaled.csv", 100)
		manager.save_matrix_to_file(image_structure, "HLA_image_structure_100i.csv")

		## prepare train data
		representation.simple_conversion_to_img_matrix("datasets//HLA_data_clean_reformated_scaled.csv")
		representation.build_patient_representation("datasets//HLA_data_clean_reformated_scaled_interpolated.csv", image_structure)
		real_data = representation.build_patient_matrix("datasets//HLA_data_clean_reformated_scaled_interpolated.csv", image_structure)
		(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.72)

		## Run CNN
		classification.run_CNN(train_X, train_Y, test_X, test_Y, 20, prediction_dataset, image_structure)


	elif(action == "run"):

			## run solver on the target file
			## IN PROGRESS

			## Prepare to predict data - give an empty array
			prediction_dataset = []

			## deal with file names
			target_file = argv[2]
			reformated_data_file_name = target_file.split(".")
			reformated_data_file_name = reformated_data_file_name[0]+"_reformated.csv"
			scaled_data_file_name = reformated_data_file_name.split(".")
			scaled_data_file_name = scaled_data_file_name[0]+"_scaled.csv"
			matrix_save_file_name = scaled_data_file_name.split(".")
			matrix_save_file_name = matrix_save_file_name[0]+"_saved_matrix.csv"
			interpolated_data_file_name = scaled_data_file_name.split(".")
			interpolated_data_file_name = interpolated_data_file_name[0]+"_interpolated.csv"

			## preprocessing
			last_pos = preprocessing.get_number_of_columns_in_csv_file(target_file) -1
			preprocessing.reformat_input_datasets(target_file, last_pos, True)
			preprocessing.normalize_data(reformated_data_file_name)
			image_structure = representation.build_image_map_GMA(scaled_data_file_name, 15)
			manager.save_matrix_to_file(image_structure, matrix_save_file_name)

			## prepare train data
			representation.simple_conversion_to_img_matrix(scaled_data_file_name)
			representation.build_patient_representation(interpolated_data_file_name, image_structure)
			real_data = representation.build_patient_matrix(interpolated_data_file_name, image_structure)
			(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.7)

			print train_Y

			## Run CNN
			classification.run_CNN(train_X, train_Y, test_X, test_Y, 75, prediction_dataset, image_structure)

			## write report
			manager.write_report()

	elif(action == "grid-checking"):

		## Checking the impact of learning the grid on the performances of
		## DPIX.

		print "[+] Testing the optimisation ..."

		## iteration to process
		iteration_list = []
		for x in xrange(10, 300, 10):
			iteration_list.append(x)


		## init log file
		impact_log_file = open("log/optimisation_impact.log", "w")

		for iteration in iteration_list:

			print "[+] Processing iteration "+str(iteration)

			## Prepare to predict data - give an empty array
			prediction_dataset = []

			## Load image structure and set filenames
			print "[+] Loading the grid ..."

			image_structure_file = "grids/all_rna_grid_GMA_opimized_iteration_"+str(iteration)+".csv"
			scaled_data_file = "datasets/rnaseq_data_2_reformated_scaled.csv"
			interpolated_data_file = "datasets/rnaseq_data_2_reformated_scaled_interpolated.csv"

			image_structure = manager.load_matrix_from_file(image_structure_file)

			## prepare train data
			print "[+] Processing input ..."
			representation.simple_conversion_to_img_matrix(scaled_data_file)
			representation.build_patient_representation(interpolated_data_file, image_structure)
			real_data = representation.build_patient_matrix(interpolated_data_file, image_structure)
			(train_X, train_Y), (test_X, test_Y) = classification.extract_data_for_cnn(real_data, 0.4)

			## Run CNN
			classification.run_CNN(train_X, train_Y, test_X, test_Y, 50, prediction_dataset, image_structure)

			## write log
			test_accuracy = -1
			test_loss = -1
			test_auc = -1
			epochs = -1
			observation_in_training_set = -1
			observation_in_test_set = -1
			model_log_file = open("log/model_training.log", "r")
			for line in model_log_file:
				line = line.replace("\n", "")
				line_in_array = line.split(";")
				if(line_in_array[0] == "test_accuracy"):
					test_accuracy = line_in_array[1]
				elif(line_in_array[0] == "test_loss"):
					test_loss = line_in_array[1]
				elif(line_in_array[0] == "test_auc"):
					test_auc = line_in_array[1]
				elif(line_in_array[0] == "epochs"):
					epochs = line_in_array[1]
				elif(line_in_array[0] == "observation_in_training"):
					observation_in_training_set = line_in_array[1]
				elif(line_in_array[0] == "observation_in_test"):
					observation_in_test_set = line_in_array[1]
			model_log_file.close()

			impact_log_file.write(str(iteration)+"\ttest_accuracy\t"+str(test_accuracy)+"\n")
			impact_log_file.write(str(iteration)+"\ttest_loss\t"+str(test_loss)+"\n")
			impact_log_file.write(str(iteration)+"\ttest_auc\t"+str(test_auc)+"\n")
			impact_log_file.write(str(iteration)+"\tepochs\t"+str(epochs)+"\n")
			impact_log_file.write(str(iteration)+"\tobservation_in_training_set\t"+str(observation_in_training_set)+"\n")
			impact_log_file.write(str(iteration)+"\tobservation_in_test_set\t"+str(observation_in_test_set)+"\n")

		## close log file
		impact_log_file.close()



##------##
## MAIN ###################################################################
##------##


if __name__ == '__main__':
	main(sys.argv[1:])
