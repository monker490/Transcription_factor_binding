import numpy as np

#	X_train = pd.read_csv('train.csv',header = None, skiprows = 1)
#	X_test = pd.read_csv('test.csv')

def loadData(X_train):

	X_seq = X_train[[1]]
	X_labels = X_train[[2]]

	X_proc = []

	for i, x in X_seq.itertuples():
		X_proc += list(x)

	X_labels = X_labels.values

	X_proc = np.array(X_proc)

	X_seq = []
	X_labs = []

	X_sendthis = []


	for i,x in enumerate(X_proc):
		for z in x:
			if (z == 'A'):
				#X_seq += [[1,0,0,0]]
				X_seq += [1]
			elif(z == 'C'):
				#X_seq += [[0,1,0,0]]
				X_seq += [2]
			elif (z == 'G'):
				#X_seq += [[0,0,1,0]]
				X_seq += [3]
			elif (z == 'T'):
				#X_seq += [[0,0,0,1]]
				X_seq += [4]

	for i, x in enumerate(X_labels):
		if (x == 0):
			X_labs += [[0,0,0,0]]
		elif (x == 1):
			X_labs += [[1,1,1,1]]
		
	for i, x in enumerate(X_labels):
		if (x == 0):
			X_sendthis += [[0,1]]
		elif (x == 1):
			X_sendthis += [[1,0]]

	X_sendthis = np.array(X_sendthis)
	#print (X_sendthis)
	

	X_labels = np.array(X_labs)	
	X_labels = X_labels.reshape(2000,1,4) ##softmax might require a change to 2 dimensional vectors
	X_seq = np.array(X_seq)
	#X_seq = X_seq.reshape(2000,14,4)
	X_seq = X_seq.reshape(2000,14)
	X_sendseq = np.array(X_seq)
	#print (np.shape(X_labels))
	#print (np.shape(X_seq))
	#print (np.shape(X_sendseq))

	# X_seq = np.append(X_seq, X_labels, axis = 1)

	# #print (np.shape(X_seq))
	# X_mini_train = []
	# X_mini_test = []

	# for i,x in enumerate(X_seq):
		
	# 	if (i%4 == 0):
	# 		X_mini_test += [x]
	# 	else:
	# 		X_mini_train += [x]

	# X_mini_test = np.array(X_mini_test)
	# X_mini_train = np.array(X_mini_train)
	# X_mini_labels = np.array(X_mini_train[:,14,:])
	# X_mini_labels = X_mini_labels.reshape(1500,4)

	# X_final_labels = []

	# for i,x in enumerate(X_mini_labels):
		
	# 	if (np.array_equal(x,[1,1,1,1])):
	# 		X_final_labels += [[1,0]]
	# 	elif (np.array_equal(x,[0,0,0,0])):
	# 		X_final_labels += [[0,1]]

	# X_final_labels = np.array(X_final_labels)
	# #print ((X_final_labels))
		

	#print (np.shape(X_mini_train))
	#print (np.shape(X_mini_test))
	#print ((X_mini_labels)) 

	#X_seq = np.random.shuffle(X_seq)

	#print (np.shape(X_seq))

	#return (X_seq, X_labels) ## this will be used in the proper running
	return (X_sendseq,X_sendthis) ##this function used on splitting the 2000 into 1500, and 500
	#return (1,1)


	


