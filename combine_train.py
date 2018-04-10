# organize imports
import numpy as np
import h5py
import os
import _pickle as cPickle
from sklearn.linear_model import LogisticRegression
# get the data path
data_path = "data/"
data_files = os.listdir(data_path)
#print "total files : {}".format(len(data_files))

#print data_files

# initialize two lists to hold feature files and label files
features_files  = []
label_files     = []
seed = 10
classifier_path = 'classifier.cpickle'
# loop through the files and get the features files
# and labels files
for file in data_files:
	print(file)
	if "featrures" in file:
		#print "welcome"
		features_files.append(file)
	elif "labels" in file:
		label_files.append(file)

# sort the features and labels files
features_files = sorted(features_files)
label_files    = sorted(label_files)

#print features_files
#print label_files

# master features and labels
mFeatures = np.empty(shape=(0,128))
mLabels   = np.empty(0)

# change the directory where features and labels exist
os.chdir("data/")

# loop through feature and its correspoding label
for f, l in zip(features_files, label_files):
	# read the h5 file
	print(f)
	print(l)
	h5f_data  = h5py.File(f, 'r')
	h5f_label = h5py.File(l, 'r')

	# get the data from the file
	data_string   = h5f_data['dataset_1']
	labels_string = h5f_label['dataset_1']

	# convert the data into numpy array
	data   = np.array(data_string)
	
	labels = np.array(labels_string)
	
	
	print(data.shape)
	print(labels.shape)

	# concatenate the features
	mFeatures = np.concatenate((mFeatures, data), axis=0)

	# append the labels
	mLabels = np.append(mLabels, [labels])

# display the features and labels shape
print(mFeatures.shape)
print(mLabels.shape)


print("[INFO] creating model...")
model = LogisticRegression(random_state=seed)
model.fit(mFeatures, mLabels)

# dump classifier to file
print("[INFO] dumping classifier...")
f = open(classifier_path, "wb")
f.write(cPickle.dumps(model))
f.close()

