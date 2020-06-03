import os
import sys
import re

directory = sys.argv[1]

import os


sub_dirs = [directory + '/concept_assertion_relation_training_data/beth/txt/mmi', directory + '/concept_assertion_relation_training_data/partners/txt/mmi', directory + '/test_data/mmi']

concepts_file = open("concepts.txt", "w")

for sub_dir in sub_dirs:
	for filename in os.listdir(sub_dir):
		print(filename)
		doc = open(sub_dir + '/' + filename, "r")
		for line in doc:
			tokens = line.split('|')
			concepts_file.write(tokens[3].strip() + '\n')


