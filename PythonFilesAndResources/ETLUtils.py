# Import necessary libraries
import MongoDBUtils
import csv
import numpy as np

# Author: Joojay Huyn joojayhuyn@utexas.edu
# Created on 3/2016
# Script contains utility functions to perform extraction, transformation, and loading operations on raw data stored
# in the MongoDB NoSQL database. These functions transform the raw data in MongoDB to a raw presentation on txt files.
# Then, this raw representation is transformed to a csv file, which represents a curated dataset.

# Get a set of distinct system calls given syscall_table.txt This syscall_table.txt is the original raw system call
# table created along with the MongoDB database. Obviously, this system call table file has redundancies, this function
# "cleans" up this raw system call table. Contact the UT ECE Spark Research Lab for access to this file.
def getSyscallNameSet(syscallTableFileName):

    # Initializing to empty set
    syscallNamesSet = set()

    # Get lines in file
    file = open(syscallTableFileName)
    fileLines = file.readlines()

    # For each line, extract system call name and add it to the system call name set
    for line in fileLines:
        lineParts = line.strip().split(" ")
        syscallName = lineParts[1]
        syscallNamesSet.add(syscallName)

    file.close() # Closing file

    # Adding a system call "other" to set to account for unrecognized system calls that may be encountered in raw data
    syscallNamesSet.add("other")

    # Return set of distinct system calls
    return syscallNamesSet

# Get a dictionary of key-value pairs, where key is a distinct system call name and value is an index assigned to the
# call
def getSyscallNameIndexDict(syscallTableFileName):

    # Initializing to empty dictionary
    syscallNameIndexDict = {}

    # Get a set of distinct system calls
    distinctSyscallNameList = getSyscallNameSet(syscallTableFileName)

    # Populating dictionary with keys (distinct system calls) and values (indices assigned to system calls)
    # It seems like the system calls are enumerated by order of hash code, which is implemented by the Python set
    for i, syscallName in enumerate(distinctSyscallNameList): syscallNameIndexDict[syscallName] = i

    # Return dictionary
    return syscallNameIndexDict

# Function extracts raw data from a MongoDB collection (Collections "Ben" and "Mal" refer to benign or malicious
# software processes respectively). Recall that the raw data is a list of system call processes, where each process
# is a sequence of system calls. The extracted raw data is written to a file. For example, the raw data in the "Ben"
# collection in MongoDB is written to a text file "Ben.txt". Each line in "Ben.txt" refers to a software process system
# trace in the "Ben" collection. Each line is a list of comma-separated system calls, with each call encoded as an index
# for data compression purposes.
def writeDataAsSeqToFileForCollection(dbName, collectionName, idKey, gridFSFileIdListKey, syscallTableFileName,
        dataFileDir):

    # Boiler plate code to obtain raw system call traces of each process in the given MongoDB collection
    client = MongoDBUtils.getMongoClient()
    db = MongoDBUtils.getDB(client, dbName)
    gridFSInstance = MongoDBUtils.getGridFSInstance(db)
    collection = MongoDBUtils.getCollection(db, collectionName)

    # Creating .txt file to hold raw data
    dataFile = open(dataFileDir + collectionName + ".txt", "wb", 1)

    # Getting dictionary of key-value pairs, where key is a distinct system call name and value is an index assigned to
    # the call. This dictionary will be used to represent the system calls in the .txt file as indices for data
    # compression purposes
    syscallNameIndexDict = getSyscallNameIndexDict(syscallTableFileName)

    i = 1 # Counter to monitor progress of writing to file

    # Iterating through each document in given MongoDB collection
    for doc in collection.find(no_cursor_timeout=True):

        # Get document id in collection and write id to txt file
        docId = MongoDBUtils.getDocValueForKey(doc, idKey)
        dataFile.write(str(docId))

        # Get list of pointers, that point to documents in the MongoDB fs.files collection. The fs.files collection
        # stores the raw trace data for each process
        gridFSFileIdList = MongoDBUtils.getGridFSFileIdList(doc, gridFSFileIdListKey)

        # Iterating through list of pointers
        for gridFSFileId in gridFSFileIdList:

            # Get entire system call trace for the current process. The trace represents a sequence of system calls
            gridFSFile = MongoDBUtils.getGridFSFile(gridFSInstance, gridFSFileId)
            gridFSFileContent = MongoDBUtils.getGridFSFileContent(gridFSFile)

            # Getting list of lines from trace data, where each line corresponds to a system call
            syscallLines = gridFSFileContent.split("\n")

            # Iterating through lines of trace data
            for syscallLine in syscallLines:

                # Really hacky way to obtain system call name in the current line. Unfortunately, ETL code tends to be
                # hacky since you are dealing with raw uncleansed data
                if syscallLine != "" and "(" in syscallLine:
                    syscallName = syscallLine.split("(")[0].split(" ")[1]

                    # Encoding system call name as a number, based on mapping in dictionary
                    if syscallName in syscallNameIndexDict.keys(): syscallNameIndex = syscallNameIndexDict[syscallName]
                    else: syscallNameIndex = syscallNameIndexDict["other"]

                    # Writing encoded system call name to file
                    dataFile.write("," + str(syscallNameIndex))

        # Start a new line for a new process
        dataFile.write("\n")

        # For every 1000 documents processed, printing out number of documents processed so far, like a progress bar
        if i % 1000 == 0: print(i)
        i += 1 # Increment counter

    # Close text file
    dataFile.close()

# Function that transforms the raw data in a .txt file to a curated dataset in a .csv file. The curated dataset
# represents the raw data, a sequence of system calls encoded as numbers, as a vector space model. The vector space
# model represents the raw data as a bag of words model or a distribution of system call occurrences. For example,
# column 3 in row 4 of the curated dataset represents the number of times the system call encoded as number 3 occurred
# in the trace of the 4th software process.
def getVecSpaceModelFromFile(syscallTableFileName, txtFileName, csvFileDir):

    # Getting csv file name. If the raw data was stored in "Ben.txt", the csv file will be named "Ben.csv"
    txtFileNameParts = txtFileName.split(".")
    txtFileNameNoExtParts = txtFileNameParts[len(txtFileNameParts)-2].split("/")
    csvFileName = txtFileNameNoExtParts[len(txtFileNameNoExtParts)-1]

    # Creating csv file
    csvDataFile = open(csvFileDir + csvFileName + ".csv", "wb")
    csvDataFileCSVWriter = csv.writer(csvDataFile)

    # Getting number of distinct system calls (should be 332)
    numDistinctSyscalls = len(getSyscallNameSet(syscallTableFileName))

    # Opening raw data text file
    txtFile = open(txtFileName, "r+")

    # Iterating through lines of text file. Recall that each line represents the trace (sequence of system calls) of
    # a process
    for line in txtFile.readlines():

        # Getting list of system call names (encoded as numbers) from current line
        lineParts = line.split(",")

        # Discard the first element of this list because the first element actually refers to the process's MongoDB
        # document id
        lineParts.pop(0)

        # Initializing a vector of 332 features to a zero array of length 332
        vector = np.zeros(numDistinctSyscalls)

        # Get sequence of system calls for current process trace
        sysCallSeq = lineParts

        # Iterating through each system call in sequence
        for sysCall in sysCallSeq:

            # Convert string to int and update vector accordingly
            sysCallNum = int(sysCall)
            vector[sysCallNum] += 1

        # Convert array to list and write to csv file
        vectorOfInts = map(lambda x: int(x), vector)
        csvDataFileCSVWriter.writerow(vectorOfInts)

    # Close both the raw data txt file and curated csv file
    txtFile.close()
    csvDataFile.close()