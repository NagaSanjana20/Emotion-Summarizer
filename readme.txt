 	This is a Readme file for project which discuss about Emotional Driven Video Summarizer Using Vison and Audio Transformer (sentiment analysis of mp4 files based on video, audio and transcript)

	Author: 	Naga Sanjana Maddula, Manogyna sai patibandla, Sheetal gudepu, Phraharshith Krupal Gummadi


How to use this software:


1. Unzip into one directory and verify that the following files exist:
	readme.txt    				--> This is the current opened file
	phase1.py				--> This is the file contains all the source code
	labels_maps.py				--> This is the file contains all the emotion mappings
	batch_results.csv			--> This is the final result output, which shows the accuracy and confidence of each model emotion
	confusions.json				--> This is the confusion matric in json format
	project_documentation			--> This is the main documentation file which talk about our phases of implementations, challenges and all the related work



2. Running this code:

	a. follow the below folder structure and get the virtual environment setup libraries (.venv)

\\{
Folder Structure to follow:
AI_PERSONAL_ASSISTANT/
	.venv/
	oututs/
	phase1/
	test/
	
}//
	
	b. Activate the virtual environment
		Command: .venv\Scripts\Activate.ps1

	c. If the activation fails, give the permission and try again
		Command: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

	d. Run the specific mp4 file or a group of files using below command
		single file command: python phase1/phase1.py --video test/video1.mp4

		group file command: python phase1/phase1.py --dir test

	c. output batch_results.csv, confusions.json and result.json will be generated under output folder	

3. Done!