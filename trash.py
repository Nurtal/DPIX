##-------------------##
## Trash file module ##
##-------------------##
## un-used and deprecated functions that might or might not be
## useful in further test phase ...
##
## MANIFEST:
##  -> generate_random_data
##  -> compute_matrix_score_old
##  -> valid_map_matrix
##  -> mutate_map_matrix
##  -> generate_new_matrix_from_parent
##  -> generate_new_matrix_from_parent_2
##  -> compute_population_score
##  -> generate_initial_population
##  -> generate_new_population
##  -> build_image_map
##  -> build_random_image_map
##  -> select_best_grid
##  -> select_parents
##


def generate_random_data(number_of_variables, number_of_patients):
	##
	## create a csv file with
	## random variables
	##

	## importation
	import random

	data_file_name = "trash_data.csv"
	variables_to_values = {}

	## Generate data
	for x in xrange(0, number_of_variables):
		variable_name = "variable_"+str(x)
		vector = []
		min_value = random.randint(0,25)
		max_value = random.randint(85,100)
		for y in xrange(0, number_of_patients):
			scalar = random.randint(min_value, max_value)
			vector.append(scalar)
		variables_to_values[variable_name] = vector

	## Write data
	output_file = open(data_file_name, "w")

	## write header
	header = ""
	for key in variables_to_values.keys():
		header+=str(key)+","
	header = header[:-1]
	output_file.write(header+"\n")

	## write patients
	for x in xrange(0, number_of_patients):
		patient = ""
		for y in xrange(0, number_of_variables):
			key = "variable_"+str(y)
			patient+=str(variables_to_values[key][x])+","
		patient = patient[:-1]
		output_file.write(patient+"\n")

	output_file.close()






def compute_matrix_score_old(corr_matrix, grid_matrix):
	##
	## Score is the sum of distance between each element of the grid
	## and its neighbour
	##
    ## => old version, before the implementation of multiproc
    ##

	## absolute value
	corr_matrix = abs(corr_matrix)

	total_score = 0.0

	## compute half size block value, must be an integer
	half_size_block = len(grid_matrix) / 2

	## for each pixel:
	for vector in grid_matrix:
		for scalar in vector:

			## get neighbour
			neighbours = get_neighbour(scalar, grid_matrix, half_size_block)

			## compute score for each pixel (the distance from each variable with their neighbours)
			for n in neighbours:
				total_score += corr_matrix[int(scalar)][int(n)]

	## return global score
	return total_score






def valid_map_matrix(map_matrix):
	##
	## => Check if the map matrix is valid, i.e
	## if the map contains the same variable more
	## than once.
	##
	## => return True if the map_matrix is valid, or False
	## if it's not.
	##

	## init stuff
	list_of_scalar = []
	valid_matrix = True

	## loop over the matrix
	for vector in map_matrix:
		for scalar in vector:

			if(float(scalar) == -1):
				valid_matrix = False
			elif(scalar not in list_of_scalar):
				list_of_scalar.append(scalar)
			else:
				valid_matrix = False

	## return the result of the test
	return valid_matrix







def mutate_map_matrix(map_matrix, number_of_mutation):
	##===================================================================
	## => [DEPRECATED] <=***********************************************#
	##	-> Part of the first grid optimisation implementation based     #
	##     on the use of a Genetic Algorithm, replaced by GMA.          #
	##===================================================================
	##
	## => Mutate the map_matrix : inverse 2 scalar position
	## randomly in the matrix, the number of inversions is the
	## number_of_mutation
	## => return the mutated matrix
	##

	## importation
	import random

	## perform a specific number of mutation
	for x in xrange(0,number_of_mutation):

		## locate random position
		y_position_start = random.randint(0, len(map_matrix)-1)
		x_position_start = random.randint(0, len(map_matrix[0])-1)

		y_position_end = random.randint(0, len(map_matrix)-1)
		x_position_end = random.randint(0, len(map_matrix[0])-1)

		## get the corrersping values
		value_start = map_matrix[y_position_start][x_position_start]
		value_end = map_matrix[y_position_end][x_position_end]

		## perform the inversion
		map_matrix[y_position_start][x_position_start] = value_end
		map_matrix[y_position_end][x_position_end] = value_start

	## return the mutated matrix
	return map_matrix



def generate_new_matrix_from_parent(parent_matrix_1, parent_matrix_2):
	##===================================================================
	## => [DEPRECATED] <=***********************************************#
	##	-> Part of the first grid optimisation implementation based     #
	##     on the use of a Genetic Algorithm, replaced by GMA.          #
	##===================================================================
	##
	## => A very simple reproduction function, create a new matrix
	##    and for each scalar run the dice to know if it should come
	##    from parent 1 or parent 2, check if variable not already in
	##    matrix
	##
	## => can return False Matrix
	##

	## importation
	import numpy
	import random

	child_matrix = numpy.zeros(shape=(len(parent_matrix_1),len(parent_matrix_1[0])))
	scalar_in_child = []
	errors_cmpt = 0
	errors_position_list = []

	for x in xrange(0,len(child_matrix)):
		for y in xrange(0, len(child_matrix[0])):

			## roll the dices
			random_choice = random.randint(0,100)

			if(random_choice > 50):
				scalar_to_add = parent_matrix_1[x][y]
				if(scalar_to_add not in scalar_in_child):
					child_matrix[x][y] = parent_matrix_1[x][y]
					scalar_in_child.append(scalar_to_add)
				else:
					if(parent_matrix_2[x][y] not in scalar_in_child):
						child_matrix[x][y] = parent_matrix_2[x][y]
						scalar_in_child.append(parent_matrix_2[x][y])
					else:
						child_matrix[x][y] = -1
						errors_position_list.append([x,y])
						errors_cmpt += 1
			else:
				scalar_to_add = parent_matrix_2[x][y]
				if(scalar_to_add not in scalar_in_child):
					child_matrix[x][y] = parent_matrix_2[x][y]
					scalar_in_child.append(scalar_to_add)
				else:
					if(parent_matrix_1[x][y]):
						child_matrix[x][y] = parent_matrix_1[x][y]
						scalar_in_child.append(parent_matrix_1[x][y])
					else:
						child_matrix[x][y] = -1
						errors_position_list.append([x,y])
						errors_cmpt += 1

	return child_matrix






def generate_new_matrix_from_parent_2(parent_tuple):
	##===================================================================
	## => [DEPRECATED] <=***********************************************#
	##	-> Part of the first grid optimisation implementation based     #
	##     on the use of a Genetic Algorithm, replaced by GMA.          #
	##===================================================================
	##
	## => A very simple reproduction function, create a new matrix
	##    and for each scalar run the dice to know if it should come
	##    from parent 1 or parent 2, check if variable not already in
	##    matrix
	##
	## => can return False Matrix
	##

	## importation
	import numpy

	parent_matrix_1 = parent_tuple[0]
	parent_matrix_2 = parent_tuple[1]

	child_matrix = numpy.zeros(shape=(len(parent_matrix_1),len(parent_matrix_1[0])))
	scalar_in_child = []
	errors_cmpt = 0
	errors_position_list = []

	for x in xrange(0,len(child_matrix)):
		for y in xrange(0, len(child_matrix[0])):

			## roll the dices
			random_choice = random.randint(0,100)

			if(random_choice > 50):
				scalar_to_add = parent_matrix_1[x][y]
				if(scalar_to_add not in scalar_in_child):
					child_matrix[x][y] = parent_matrix_1[x][y]
					scalar_in_child.append(scalar_to_add)
				else:
					if(parent_matrix_2[x][y] not in scalar_in_child):
						child_matrix[x][y] = parent_matrix_2[x][y]
						scalar_in_child.append(parent_matrix_2[x][y])
					else:
						child_matrix[x][y] = -1
						errors_position_list.append([x,y])
						errors_cmpt += 1
			else:
				scalar_to_add = parent_matrix_2[x][y]
				if(scalar_to_add not in scalar_in_child):
					child_matrix[x][y] = parent_matrix_2[x][y]
					scalar_in_child.append(scalar_to_add)
				else:
					if(parent_matrix_1[x][y]):
						child_matrix[x][y] = parent_matrix_1[x][y]
						scalar_in_child.append(parent_matrix_1[x][y])
					else:
						child_matrix[x][y] = -1
						errors_position_list.append([x,y])
						errors_cmpt += 1

	return child_matrix





def compute_population_score(dist_mat, population, use_multiproc):
	##===================================================================
	## => [DEPRECATED] <=***********************************************#
	##	-> Part of the first grid optimisation implementation based     #
	##     on the use of a Genetic Algorithm, replaced by GMA.          #
	##===================================================================
	##
	## => Compute the score for the population:
	## score is just the mean of the score for
	## each matrix.
	##

	## importation
	from multiprocessing import Pool

	## MULTIPROC
	if(use_multiproc):
		number_of_proc = 80
		p = Pool(number_of_proc)

		sub_populations_array = []
		sub_array = []

		for ind in population:
			if(len(sub_array) < number_of_proc):
				sub_array.append((ind, dist_mat))
			else:
				sub_populations_array.append(sub_array)
				sub_array = []

		scores = []
		for sub_array in sub_populations_array:
			scores_to_append = []
			scores_to_append = p.map(compute_matrix_score, sub_array)
			scores += scores_to_append

	## CLASSIC USE
	else:
		scores = []
		for ind in population:
			scores.append(compute_grid_score_wrapper(dist_mat, ind))

	return scores



def generate_initial_population(number_of_individuals, dist_mat, use_multiproc):
	##===================================================================
	## => [DEPRECATED] <=***********************************************#
	##	-> Part of the first grid optimisation implementation based     #
	##     on the use of a Genetic Algorithm, replaced by GMA.          #
	##===================================================================
	##
	## IN PROGRESS
	##


	if(use_multiproc):

		## importation
		from pathos.multiprocessing import ProcessingPool as Pool

		initial_population = []
		inputs = []
		for x in xrange(0,number_of_individuals):
			inputs.append(dist_mat)

		res = Pool().amap(init_grid_matrix, inputs)
		initial_population = res.get()

	else:

		initial_population = []
		for x in xrange(0,number_of_individuals):
			print "[INFO] - create grid "+str(x)
			random_grid = init_grid_matrix(dist_mat)
			initial_population.append(random_grid)

	return initial_population





def generate_new_population(population_size, mutation_rate, parents, use_multiproc):
	##===================================================================
	## => [DEPRECATED] <=***********************************************#
	##	-> Part of the first grid optimisation implementation based     #
	##     on the use of a Genetic Algorithm, replaced by GMA.          #
	##===================================================================
	##
	## IN PROGRESS
	##

	## importation
	import random
	from multiprocessing import Pool

	individual_cmpt = 0
	new_population = []
	tentative_cmpt = 0

	avorted_matrix = 0

	if(use_multiproc):



		patient_to_create = population_size
		patient_created = 0
		number_of_proc = 80
		p = Pool(number_of_proc)

		while(patient_to_create > patient_created):

			args_array = []
			created_child = []
			nb_of_proc_used = 0
			if( (patient_to_create - patient_created) > number_of_proc ):
				nb_of_proc_used = number_of_proc
			else:
				nb_of_proc_used = (patient_to_create - patient_created)

			for x in xrange(0, nb_of_proc_used):

				## determine the content of args_array
				parents_are_different = False
				male_index = -1
				female_index = -1
				while(not parents_are_different):

					male_index = random.randint(0,len(parents))
					female_index = random.randint(0,len(parents))

					if(male_index != female_index):
						parents_are_different = True

				parent_male = parents[random.randint(0,len(parents)-1)]
				parent_female = parents[random.randint(0,len(parents)-1)]
				args_array.append((parent_male, parent_female))


			## run multiproc on child creation
			created_child = p.map(generate_new_matrix_from_parent_2, args_array)

			## test if child is valid, run mutation
			## Sequential for now
			for child in created_child:
				status = "Failed"
				if(valid_map_matrix(child)):

					## Mutation
					if(random.randint(0,100) <= mutation_rate):
						child = mutate_map_matrix(child, 4)

						if(valid_map_matrix(child)):
							new_population.append(child)
							individual_cmpt += 1
							status = "Success"
					else:
						new_population.append(child)
						individual_cmpt += 1
						status = "Success"

				else:
					avorted_matrix += 1
				tentative_cmpt += 1


			## update number of created patients
			patient_created = len(new_population)

		print "[AVORTED] " +str(avorted_matrix)


	else:

		## Create the new generation
		while(individual_cmpt != population_size):

			## get the parents (random selection)
			parents_are_different = False
			male_index = -1
			female_index = -1
			while(not parents_are_different):

				male_index = random.randint(0,len(parents))
				female_index = random.randint(0,len(parents))

				if(male_index != female_index):
					parents_are_different = True

			parent_male = parents[random.randint(0,len(parents)-1)]
			parent_female = parents[random.randint(0,len(parents)-1)]

			## create the child
			child = generate_new_matrix_from_parent(parent_male, parent_female)

			status = "Failed"
			if(valid_map_matrix(child)):

				## Mutation
				if(random.randint(0,100) <= mutation_rate):
					child = mutate_map_matrix(child, 4)

					if(valid_map_matrix(child)):
						new_population.append(child)
						individual_cmpt += 1
						status = "Success"
				else:
					new_population.append(child)
					individual_cmpt += 1
					status = "Success"

			tentative_cmpt += 1

	return new_population






















def build_image_map(data_file, n_cycles):
	##===================================================================
	## => [DEPRECATED] <=***********************************************#
	##	-> Part of the first grid optimisation implementation based     #
	##     on the use of a Genetic Algorithm, replaced by GMA.          #
	##===================================================================
	##
	##
	##
	## => Core of the idea <=
	## This function perform the global operation
	## of finding the best possible grid.
	##
	## n_cycle is the number of cycles for the genetic algorithm.
	##
	## [STEP 1] => compute the distance matrix between
	## the variables, in this first version it's absolute
	## value of the correlation matrix.
	##
	## [STEP 2] => use a genetic algorithm to find the optimal
	## solution (i.e the best map to build the image)
	##
	## -> return the best grid found
	##


	## importation
	import manager
	from time import gmtime, strftime
	import datetime

	optimization_log_file = open("log/optimization.log", "w")
	use_multiproc = False


	## get the correlation matrix
	print "[COMPUTE DISTANCE - START] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
	a = datetime.datetime.now()
	corr_mat = get_correlation_matrix(data_file)
	b = datetime.datetime.now()
	c = b - a
	c = divmod(c.days * 86400 + c.seconds, 60)
	optimization_log_file.write("COMPUTE DISTANCE,"+str(c[0])+" - " +str(c[1])+"\n")
	print "[COMPUTE DISTANCE - END] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))


	## compute the distance between each variables
	## kind of "distance matrix"
	## just use the absolute value of correlation, in this case
	## we consider that a highly negative value (anticorrelation)
	## is associated with the variable of interest same as a hihgly correlated
	## variable
	print "[PROCESS MATRIX - START] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
	a = datetime.datetime.now()
	dist_mat = abs(corr_mat)
	b = datetime.datetime.now()
	c = b - a
	c = divmod(c.days * 86400 + c.seconds, 60)
	optimization_log_file.write("PROCESS MATRIX,"+str(c[0])+" - " +str(c[1])+"\n")
	print "[PROCESS MATRIX - END] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))





	##------------------------##
	## LEARN THE OPTIMAL GRID ##
	##------------------------##
	##
	## Create a population of random grid and then
	## use a genetic algorithm to learn the optimal grid
	##

	## init the log file
	log_file = open("learning_optimal_grid.log", "w")


	## Create initial population
	print "[GENERATE INIT POP - START] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
	a = datetime.datetime.now()
	initial_population = generate_initial_population(100, dist_mat, use_multiproc)
	b = datetime.datetime.now()
	c = b - a
	c = divmod(c.days * 86400 + c.seconds, 60)
	optimization_log_file.write("GENERATE INIT POP,"+str(c[0])+" - " +str(c[1])+"\n")
	print "[GENERATE INIT POP - END] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))


	## Run the genetic algorithm over
	## a number cycles
	number_of_cycles = n_cycles
	current_population = initial_population
	best_grid = select_best_grid(current_population, dist_mat)
	best_grid_score = compute_grid_score_wrapper(dist_mat, best_grid)

	for x in xrange(0, number_of_cycles):

		print "[GENERATION] ========= "+str(x)+ " ================="


		## debug
		print "[ENTER THE CYCLE] => " +str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

		## Select parents
		print "[SELECT PARENTS - START] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
		a = datetime.datetime.now()
		parents = select_parents(current_population, 20,2, dist_mat)
		b = datetime.datetime.now()
		c = b - a
		c = divmod(c.days * 86400 + c.seconds, 60)
		optimization_log_file.write("PARENTS SELECTION,"+str(c[0])+" - " +str(c[1])+"\n")
		print "[SELECT PARENTS - END] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))


		## parameters
		population_size = 100
		individual_cmpt = 0
		new_population = []
		tentative_cmpt = 0
		mutation_rate = 10

		## Create the new generation
		print "[GENERTE NEW POP - START] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
		a = datetime.datetime.now()
		new_population = generate_new_population(population_size, mutation_rate, parents, use_multiproc)
		b = datetime.datetime.now()
		c = b - a
		c = divmod(c.days * 86400 + c.seconds, 60)
		optimization_log_file.write("GENERATE NEW POP,"+str(c[0])+" - " +str(c[1])+"\n")
		print "[GENERATE NEW POP - END] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))

		## update population
		current_population = new_population

		## compute score for the population
		#pop_scores = compute_population_score(dist_mat, current_population)

		print "[COMPUTE SCORE - START] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))
		a = datetime.datetime.now()
		pop_scores = compute_population_score(dist_mat, current_population, use_multiproc)
		b = datetime.datetime.now()
		c = b - a
		c = divmod(c.days * 86400 + c.seconds, 60)
		optimization_log_file.write("COMPUTE SCORE,"+str(c[0])+" - " +str(c[1])+"\n")
		print "[COMPUTE SCORE - END] => "+str(str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())))

		pop_score = 0
		for ind_score in pop_scores:
			pop_score += ind_score
		pop_score = float(float(pop_score)/float(len(pop_scores)))
		print "[GLOBAL SCORE] "+ str(pop_score)

		## find best and worst individual in the population
		scores = pop_scores
		best_score = max(scores)
		worst_score = min(scores)

		print "[BEST SCORE] "+str(best_score)
		print "[WORST SCORE] "+str(worst_score)

		## save best solution (best grid)
		best_grid_candidate = select_best_grid(current_population, dist_mat)
		best_grid_candidate_score = best_score

		if(best_grid_candidate_score > best_grid_score):
			best_grid = best_grid_candidate
			best_grid_candidate_score = best_grid_score


		## Write all informations in a log file
		log_file.write(">generation "+str(x)+"\n")
		log_file.write("global_score;"+str(pop_score)+"\n")
		log_file.write("best_score;"+str(best_score)+"\n")
		log_file.write("worst_score;"+str(worst_score)+"\n")


		## save the generated grid at fix number of iteration
		iteration_checkpoint = [5,10,15,20,25,30,35,40,45,50]
		if(x in iteration_checkpoint):
			manager.save_matrix_to_file(best_grid, "grids/automatic_save_"+str(x)+".csv")

		## debug
		print "[EXIT THE CYCLE] => " +str(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))



	## close log file
	log_file.close()
	optimization_log_file.close()

	## return best solution found
	return best_grid



def build_random_image_map(data_file, n_cycle, batch_size):
	"""
	Random optimization to build
	the image map
	"""

	## importation
	import manager
	import datetime

	use_multiproc = True

	## get the correlation matrix
	print "[PROCESSING - START]"
	corr_mat = get_correlation_matrix(data_file)

	## compute the distance between each variables
	## kind of "distance matrix"
	## just use the absolute value of correlation, in this case
	## we consider that a highly negative value (anticorrelation)
	## is associated with the variable of interest same as a hihgly correlated
	## variable

	dist_mat = abs(corr_mat)
	print "[PROCESSING - END]"

	## init the log file
	log_file = open("learning_optimal_grid.log", "w")

	## Create initial population
	print "[DEBUG] - init pop start "+str(datetime.datetime.now())
	initial_population = generate_initial_population(batch_size, dist_mat, use_multiproc)
	print "[DEBUG] - init pop end "+str(datetime.datetime.now())

	## Run the genetic algorithm over
	## a number cycles
	number_of_cycles = n_cycle
	current_population = initial_population
	print "[DEBUG] - select best grid start "+str(datetime.datetime.now())
	best_grid = select_best_grid(current_population, dist_mat)
	print "[DEBUG] - select best grid end "+str(datetime.datetime.now())
	best_grid_score = compute_grid_score_wrapper(dist_mat, best_grid)

	for x in xrange(0, number_of_cycles):

		print "[ITERATION - "+str(x)+"]"
		print "[DEBUG] - iteration start "+str(datetime.datetime.now())

		print "[DEBUG] - generate population start "+str(datetime.datetime.now())
		current_population = generate_initial_population(batch_size, dist_mat, use_multiproc)
		print "[DEBUG] - generate pupulation end "+str(datetime.datetime.now())
		current_best_grid = select_best_grid(current_population, dist_mat)
		current_best_grid_score = compute_grid_score_wrapper(dist_mat, current_best_grid)

		if(current_best_grid_score > best_grid_score):
			best_grid = current_best_grid
			best_grid_score = current_best_grid_score

		## Write all informations in a log file
		log_file.write(">generation "+str(x)+"\n")
		log_file.write("global_score;"+"NA"+"\n")
		log_file.write("best_score;"+str(best_grid_score)+"\n")
		log_file.write("worst_score;"+"NA"+"\n")

		## save the generated grid at fix number of iteration
		iteration_checkpoint = [5,10,15,20,25,30,35,40,45,50]
		if(x in iteration_checkpoint):
			manager.save_matrix_to_file(best_grid, "grids/automatic_save_"+str(x)+".csv")

		print "[BEST SCORE - "+str(best_grid_score)+"]"
		print "[DEBUG] - iteration end "+str(datetime.datetime.now())

	## close log file
	log_file.close()

	## return best solution found
	return best_grid


def select_best_grid(population, dist_matrix):
	##
	## Select best grid ( according to score ) from
	## a population.
	## -> population is a list of grids
	## -> dist_matrix is by default the correlation matrix
	## -> return a grid
	## [TEST MULTI LEVEL OF MULTIPROC]
	##	=> bad idea : AssertionError: daemonic processes are not allowed to have children

	## importation
	from pathos.multiprocessing import ProcessingPool as Pool
	import operator

	## Compute the score for each individual (matrix) in the
	## population.
	ind_index_to_score = {}
	index_to_ind = {}
	ind_index = 0

	## Classic way
	for ind in population:
		ind_index_to_score[ind_index] = compute_grid_score_wrapper(dist_matrix, ind)
		index_to_ind[ind_index] = ind
		ind_index += 1

	## select best grid
	best_grid_index = max(ind_index_to_score.iteritems(), key=operator.itemgetter(1))[0]
	best_grid = index_to_ind[best_grid_index]

	return best_grid




def select_parents(population, good_parents, bad_parents, dist_matrix):
	##===================================================================
	## => [DEPRECATED] <=***********************************************#
	##	-> Part of the first grid optimisation implementation based     #
	##     on the use of a Genetic Algorithm, replaced by GMA.          #
	##===================================================================
	## Select parents from population
	## - population is a list of matrix
	## - good_parents is an int, number of good (best score)
	##   parents to return
	## - bad parents is an int, number of bad (low score)
	##   parents to return
	##

	## importation
	import operator

	## Compute the score for each individual (matrix) in the
	## population.
	ind_index_to_score = {}
	ind_index = 0
	for ind in population:
		ind_index_to_score[ind_index] = compute_grid_score_wrapper(dist_matrix, ind)
		ind_index += 1


	## Select good parents, i.e the top score
	all_good_parents_assigned = False
	number_of_good_parents = 0
	list_of_good_parents = []
	while(not all_good_parents_assigned):

		selected_parent = max(ind_index_to_score.iteritems(), key=operator.itemgetter(1))[0]
		del ind_index_to_score[selected_parent]
		list_of_good_parents.append(selected_parent)
		number_of_good_parents += 1

		if(number_of_good_parents == good_parents):
			all_good_parents_assigned = True

	## Select bad parents, i.e the low score
	all_bad_parents_assigned = False
	number_of_bad_parents = 0
	list_of_bad_parents = []
	while(not all_bad_parents_assigned):

		selected_parent = min(ind_index_to_score.iteritems(), key=operator.itemgetter(1))[0]
		del ind_index_to_score[selected_parent]
		list_of_bad_parents.append(selected_parent)
		number_of_bad_parents += 1

		if(number_of_bad_parents == bad_parents):
			all_bad_parents_assigned = True


	## Create the list of patient to return
	parents_id = list_of_good_parents + list_of_bad_parents
	parents = []
	for p_id in parents_id:
		parents.append(population[p_id])

	return parents
