#!/usr/bin/env python
# This Python file uses the following encoding: utf-8
# Evaluate_means.py
#Inputs:
#Case_name
#
#Outputs:
#Variables= [Flow weighted Mach Number, Flow weigthed totoal pressure, ... ]
#
# Variables = Evaluate_means(Case_name)
#
# Authors: Janry, 14/4/2018
import os         as         os
import numpy      as         np
from   getopt     import getopt
import sys        as         sys 

def POSTPROCESSOR(Case_directory,verbosity):


    # check the openfoam version
    # TODO: write the command to check the OpenFOAM version


    if verbosity > 1 :
        print("Now activate the post processors")
    
    
    #Doing postprocess with processors of OpenFOAM
    CASE_DIR = Case_directory
    os.chdir(CASE_DIR)
    
    os.system("sonicFoam -postProcess -func MachNo -latestTime > MachNoLatestTime")
    #os.system("wallCellVelocity -latestTime")

    # reading the latest time value from the file
    f_latest = open('MachNoLatestTime', 'r')
    for line in f_latest:
        if 'Create mesh for time =' in line:
            timeString = line

    TimeList = timeString.replace('\n','').split(' ')
    latestTime = TimeList[5]
    return latestTime

def File_Reader(Filed_data_type,Field_name,Case_directory,time,verbosity):


    scalar_list = []
    vector_list = []

    file_path = Case_directory + "/" + str(time) + "/" + str(Field_name)
    if verbosity>1:
        print("The field path is {}".format(file_path))

    f = open(file_path,'r')

    line_count = 0; start_line = -1000

    if Field_name == 'p':
        for line in f:
            line_count += 1
            if "o-00" in line:

                # This returns the starting line number 
                start_line = line_count

            if line_count == (start_line + 7):
                list_length = int(line.replace('\r','').replace('\n',''))
    else:
        for line in f:
            line_count += 1
            if "o-00" in line:

                # This returns the starting line number 
                start_line = line_count

            if line_count == (start_line + 4):
                list_length = int(line.replace('\r','').replace('\n',''))   

    print("list_length",list_length)     
    f.close()




    # read fields file and return a list
    if ("Scalar" == Filed_data_type) and (Field_name == 'p'):

        f_scalar = open(file_path,'r')
        line_count = 0
        for line in f_scalar:
            line_count += 1
            if (line_count >= (start_line + 9)) and (line_count <= (start_line + 9 + list_length -1)):
                scalar_list.append( float(line.replace('\r','').replace('\n','')))
        field_data_list = scalar_list
        #print field_data_list

    elif "Vector" == Filed_data_type:
        f_vector = open(file_path,'r')
        line_count = 0
        for line in f_vector:
            line_count += 1
            if (line_count >= (start_line + 6)) and (line_count <= (start_line + 6 + list_length -1)):
                temp = line.replace('(','').replace(')','').replace('\r','').replace('\n','').split(' ')
                for i in range(len(temp)):
                    temp[i] = float(temp[i])
                vector_list.append(temp)
        field_data_list = vector_list
        #print field_data_list
    elif ("Scalar" == Filed_data_type) and (Field_name != 'p'):
        f_scalar = open(file_path,'r')
        line_count = 0
        for line in f_scalar:
            line_count += 1
            if (line_count >= (start_line + 6)) and (line_count <= (start_line + 6 + list_length -1)):
                scalar_list.append( float(line.replace('\r','').replace('\n','')))
        field_data_list = scalar_list


    else:
        print "Please double check your "
    

    return  field_data_list

def Area_count(case_name,verbosity):

    Case_directory = case_name
    if verbosity>1 :
        print("The area caculator is come to the directory of {}".format(case_name))

    Constant_directory = case_name + '/constant'
    polyMesh_directory = Constant_directory + '/polyMesh'
    boundary_file      = polyMesh_directory + '/boundary'
    faces_file         = polyMesh_directory + '/faces'
    points_file        = polyMesh_directory + '/points'

    # open the boundary file and read the line number of nFaces and startFace
    f_boundary = open(boundary_file,'r')
    line_count = 0
    for line in f_boundary:
        line_count +=1
        if "o-00" in line:
            line_nFaces = line_count+3
            line_startface = line_count+4
    print line_nFaces
    f_boundary.close()

    # re-open the bounary file and get the value of nfaces and startFaces
    f_boundary = open(boundary_file,'r')
    line_count = 0
    for line in f_boundary:
        line_count += 1
        if line_count == line_nFaces:
            nFaces = float(line.replace('\r','').replace('\n','').replace(';','').split(' ')[-1])

        if line_count == line_startface:
            startFace = float(line.replace('\r','').replace('\n','').replace(';','').split(' ')[-1])
    f_boundary.close()

    if verbosity >1 :
        print("The numbe of faces is:{}".format(nFaces))
        print("The index of start face is:{}".format(startFace))



    # open the faces file:
    faces_list = []
    f_faces = open(faces_file,'r')
    line_count = 0
    for line in f_faces:
        line_count += 1
        if line_count >= (startFace + 21) and line_count < (startFace + 21 + nFaces):
            temp = line.replace('\r','').replace('\n','').replace('(',' ').replace(')',' ').split(' ') 
            temp.remove('')
            del temp[0]
            faces_list.append(  temp )
    f_faces.close()
    print faces_list,len(faces_list)

    # Reading the point coordinates into a list
    points_total = []
    f_points = open(points_file,'r')
    line_count = 0
    for line in f_points:
        line_count += 1
        if line_count == 19 :
            total_line = float(line.replace('\r','').replace('\n',''))
        if line_count >= 21 and line_count < (21 + total_line):
            temp = line.replace('\r','').replace('\n','').replace('(',' ').replace(')',' ').split(' ') 
            temp.remove('')
            temp.remove('')
            points_total.append(temp)
    f_points.close()
    #print total_line, points_total

    face_coordinates_list = []

    for i in range(len(faces_list)):
        facet_coordinates = []
        for j in range(len(faces_list[i])):
            index_temp = int(faces_list[i][j])
            coordinates_temp = points_total[index_temp]
            facet_coordinates.append(coordinates_temp)
        face_coordinates_list.append(facet_coordinates)

    print face_coordinates_list[-1]


    #then calculating areas of every facet

    # y
    # ^
    # | 
    # |  B--------C
    # |  |        |
    # |  A--------D
    # -----------------> z
    # A: face_coordinates_list[i][0]
    # B: face_coordinates_list[i][1]
    # C: face_coordinates_list[i][2]
    # D: face_coordinates_list[i][3]  
    # Thus for current approach, the simple way to calculate the area is |AB| * |BC|
    # AB = 
    area_list = []
    for i in range(len(face_coordinates_list)):
        AB = abs( float(face_coordinates_list[i][0][1]) - float(face_coordinates_list[i][1][1])   )
        BC = abs( float(face_coordinates_list[i][1][2]) - float(face_coordinates_list[i][2][2])   )
        area_list.append(AB*BC)
    return area_list

def Area_Weighted_Average_Scalar(scalar_list,area_list):


    if len(scalar_list) != len(area_list):
        raise MyError("The length of scalar_list is not equal to the area_list")
    
    numerator = []
    for i in range(len(scalar_list)):
        temp = scalar_list[i]*area_list[i]
        numerator.append(temp )
    average = sum(numerator)/sum(area_list)

    return average





def main(uoDict):

    verbosity = uoDict.get("--verbosity", 1)

    if verbosity >1 :
        print("Heading here for evaluation")

    # main file to be executed
    case_name = uoDict.get("--case_name", "test")

    if verbosity>1:
        print("Currently do post processing for {}".format(case_name))

    # do post processing and get the latestTime
    latestTime = POSTPROCESSOR(case_name,verbosity)
    if verbosity>1: 
        print("The latestTime is: {} ".format(latestTime))

    # Do post processing evalutation            
    if verbosity >1:
        print("Now reading the Mach numberi field and loading the Mach number list")


    Mach_list        = File_Reader("Scalar","Ma",case_name,latestTime,verbosity)
    if verbosity >1:
        print("Mach_list is {}".format(Mach_list))

    pressure_list    = File_Reader("Scalar","p",case_name,latestTime,verbosity)
    if verbosity >1:
        print("pressure_list is {}".format(pressure_list))    



    area_list        = Area_count(case_name,verbosity)
    if verbosity> 1 :
        print("The area_list is: {}".format(area_list))     

    average_Mach     = Area_Weighted_Average_Scalar(Mach_list,area_list)
    average_Pressure = Area_Weighted_Average_Scalar(pressure_list,area_list)
    Post_list = []
    Post_list.append(average_Mach)
    #Post_list.append(average_alpha)
    #Post_list.append(mass_outlet)
    #Post_list.append(average_total_pressure)


    if verbosity > 1:
        print("The area average Mach number is: {}".format(average_Mach))
        print("The area average pressure is: {}".format(average_Pressure))

    return Post_list

###
###
class MyError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
###
###
shortOptions = ""
longOptions = ["help", "case_name=", "verbosity="]
###
###
def printUsage():
    print("")
    print("Usage: Evaluate_means.py [--help] [--case_name=<jobFileName>] [--verbosity=<0,1,2>]")
    print("\n")
    print(" --help      Display help.")
    print("\n")
    print(" --case_name=      Directory for Case folder (absolute path)")
    print("\n")
    print(" --verbosity   Set level of screen output 0-none; 1-some; 2-all.")
    return
###
###
if __name__ == "__main__":

    userOptions = getopt(sys.argv[1:], shortOptions, longOptions)
    uoDict = dict(userOptions[0])

    if len(userOptions[0]) == 0 or "--help" in uoDict:
        printUsage()
        sys.exit(1)

    # execute the code
    try:
        main(uoDict)
        print("\n \n")
        print("SUCCESS.")
        print("\n \n")

    except MyError as e:
        print("This run has gone bad.")
        print(e.value)
        sys.exit(1)