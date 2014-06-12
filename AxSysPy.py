# AxSysPy.py
# Leverages Python to quickly write AxSys protocol files in 
# text document format.

# - Changelog -
# Created on 06/12/2014 by BCP


# - Import required modules -


# - Python Function Definitions -
def AxSysWrite(fileName, ActionList, numberOfFunctions):
	"""Compiles a list of AxSys functions and actions into a 
	protocol in text document (.txt) format."""
	
	# NOTE: AxSys Functions (indicated throughout with capitalization) 
	# are distinctly different than Python functions. AxSys Functions
	# are unique, globally applied sets of AxSys Actions. Any
	# modifications of AxSys Functions are applied directly to the 
	# function library which is called on by the AxSys Action "Call ()".
	
	# Open protocol text file for appending
	protocol = open(str(fileName) + '.txt', 'a')
	
	
	
	# Initialize header for AxSys file
	AxSysFile = 'AD DOCUMENT VERSION: 11\nNUMBER OF COUNTERS:  0\n\n'
	
	# Initialize AxSys Function library
	AxSysFile += 'NUMBER OF FUNCTIONS:  ' + str(numberOfFunctions) + '\n\n'
	
	
	
	# Initialize main program - all following elements must be AxSys Actions
	AxSysFile += '__________________ MAIN PROGRAM _______________\n\nSTART OF PROGRAM LIST\n\n\nNUMBER OF ACTIONS:  ' + str(actions) + '\n\n'

	
	
# - Code -