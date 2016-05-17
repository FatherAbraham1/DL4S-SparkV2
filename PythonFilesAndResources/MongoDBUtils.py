# Importing necessary libraries
from pymongo import MongoClient
import gridfs

# Author: Joojay Huyn joojayhuyn@utexas.edu
# Created on 3/2016
# Script contains utility functions to interact and communicate with a MongoDB instance via Python

# Get mongo client
def getMongoClient(): return MongoClient()

# Get specified database in specified mongo client
def getDB(client, dbName): return client[dbName]

# Get specified collection in specified database
def getCollection(db, collectionName): return db[collectionName]

# Get value of the given key in the given json document
def getDocValueForKey(doc, key): return doc[key]

# Get value (a list of grid fs file ids) of the given grid-fs-file-id-list key in the given json document
def getGridFSFileIdList(doc, gridFSFileIdListKey): return getDocValueForKey(doc, gridFSFileIdListKey)

# Get a gridfs instance from specified database
def getGridFSInstance(db): return gridfs.GridFS(db)

# Get a grid fs file from given a grid fs file id from a specified grid fs instance
def getGridFSFile(gridFSInstance, gridFSFileId): return gridFSInstance.get(gridFSFileId)

# Get the raw content from a grid fs file
def getGridFSFileContent(file): return file.read()
