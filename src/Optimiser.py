#!/usr/bin/env python
# This Python file uses the following encoding: utf-8
#
#
#
#    This file is used to read the dataList_New

import os         as         os
import numpy    as         np
import shutil as         sh
from     getopt import getopt
import sys        as         sys 
from scipy.optimize import minimize
from scipy.spatial import distance
from numpy.random import random
from scipy.interpolate import griddata
from scipy import integrate
import matplotlib.pyplot as plt 
import Evaluate_means as Evaluate_means


# Golable index

GLOBAL_INDEX = 0
GLOBAL_DIR   = ""
GLOBAL_COUNT = 0



def load_data(M,C,D):
    ##############################
    # Generate data base list:

    DATA_LIST = []
    # 将dataList中的已有的数据加载进来
    Index     = [] # List to store the index
    Status    = [] # List to store the Status
    Geometry  = [] # List to store the Geometry
    Post      = [] # List to store the results of post processing
    Cost      = [] # List to store the cost functions
    Directory = [] # List to store the case directory
    Method    = [] # List to store the method for getting the data, interpolation or evaluation


    # check that file exists
    if os.path.isfile(M.ROOT_DIR+D.filename):
        if M.verbosity > 1:
            print('Loading Inital data from {}'.format(D.filename)) 
        with open(M.ROOT_DIR+D.filename) as f:
            read_data = f.read()

        # remove brackets and other stuff from results lines
        for line in read_data:
            if line.startswith("["):
                temp = line.replace(" ", '').replace('[','').replace(']','').replace('\r','').replace('\n','').split(",")
                DATA_LIST.append(temp)     

        # Convert geometry data to floats   
        for i in range(len(DATA_LIST)):
            for j in range(2+M.Nvar+M.NPost+M.NCost):
                DATA_LIST[i][j] = float(DATA_LIST[i][j])

    #[1, 1 , 0.1, 0.0, -0.1, 0.07, 0.0682, 0.0012, 0.002, 0.00008,  70., 0.2,  0.02,  nan, nan, nan, nan, nan, nan, nan, nan, folder address for simaultion ]

    if D.Use_initial_simplex: # Add extra line corresponding to first line of simplex
        temp = np.append([0,0], D.initial_simplex[0])
        temp = np.append(temp, np.zeros(M.NPost+M.NCost+2))
        DATA_LIST.append(temp)

    if M.verbosity > 1:
        print('DATA_LIST:', DATA_LIST) 


    for i in range(len(DATA_LIST)):
        Index.append(DATA_LIST[i][0])
        Status.append(DATA_LIST[i][1])
        #将Geometry信息读取到Geometry Lsit中
        Geometry.append(DATA_LIST[i][2:2+M.Nvar])
        #将后处理信息读取到Post Lsit中
        Post.append(DATA_LIST[i][2+M.Nvar:2+M.Nvar+M.NPost])
        #将后COST信息读取到Post Lsit中
        Cost.append(DATA_LIST[i][2+M.Nvar+M.NPost:2+M.Nvar+M.NPost+M.NCost])
        #将后Directory信息读取到Directory Lsit中
        Directory.append(DATA_LIST[i][2+M.Nvar+M.NPost+M.NCost+1-1])  
        Method.append(DATA_LIST[i][2+M.Nvar+M.NPost+M.NCost+2-1])


    if M.verbosity > 1:
        print('Data Lists have been created.') 
        print('    Index    :', Index) 
        print('    Status   :', Status) 
        print('    Geometry :', Geometry) 
        print('    Post     :', Post) 
        print('    Cost     :', Cost) 
        print('    Directory:', Directory) 
        print('    Method   :', Method) 




    return DATA_LIST, Index, Status, Geometry, Post, Cost, Directory, Method

##############################
def EXECUTE(Geometry_list, Case_directory, neareast_DIR, M, C, D):

    #############################
    #Prepare base case
    print("...Creating case folder")

    #ROOT_DIR = M.ROOT_DIR #"/media/uqjqi/Janry_Research/Stator_Optimization/"
    #ROOT_DIR = "/home/uqjqi/Desktop/work2/Nozzle_Optimisation/examples/con-div-nozzle/."
    MASTER_DIR = M.ROOT_DIR + '/' + C.MasterCase_folder + '/'

    CASE_DIR = Case_directory
    if M.verbosity >1:
        print("The Root directory is:",M.ROOT_DIR)
        print("The case forder is:",CASE_DIR)
        print("The basic setting is stroed in:",MASTER_DIR)


    os.chdir(M.ROOT_DIR)

    print Case_directory, os.path.isfile(Case_directory)
    #判断一下是否已经完成计算
    if os.path.exists(Case_directory):
         raise MyError('Case_directory {} already exist'.format(Case_directory))
    else:
        os.mkdir(Case_directory)
        os.system("cp -r "+ MASTER_DIR + ". " + Case_directory )
        if M.verbosity > 1:
            print('Master_Case has been duplciated to {}.'.format(Case_directory))


        #############################
        #create updated Stator_job.py
        


        print "...Creating con-div-nozzle.lua"
        GeometryString = str(Geometry_list).replace('[','').replace(']','')
        print GeometryString
        job = open(Case_directory + '/' + C.Input_filename ,'r')
        newjob = open(Case_directory + "/newjob.lua",'w')
        for line in job:
            if "OP = {" in line:
                line = 'OP = {' + GeometryString + '}\r\n'
            
            newjob.write("%s" % line)
        job.close()
        newjob.close()

        os.chdir(CASE_DIR)
        os.system('rm '+ C.Input_filename)
        os.system('mv newjob.lua '+ C.Input_filename)
        print os.getcwd()

        # Create Mesh
        os.system('./' + C.setup_command)


        
        #############################
        # 这一步是将初始化的文档拷贝过去


        if None == neareast_DIR:
            os.system('./' + C.simulate_command)


        else: 
            raise MyError('dosent')
            os.system("cp -r " + neareast_DIR +"/100000/. ./0/.")
            os.system("rm -r ./0/uniform")


        
        #os.system("decomposePar")
        #os.system("paraFoam")
 
        # excute with 4 cores
        #os.system("mpirun -np 4 transonicMRFDyMFoam -parallel")

        # reconstruct the partial files
        #os.system("reconstructPar")

        # remove the processor files to save the memory
        #os.system("rm -r processor*")


        # enhance the co-number and rerun the simulation
        #os.system("mv ./system/controlDict.bak ./system/controlDict")
        #os.system("decomposePar")
        #os.system("mpirun -np 4 transonicMRFDyMFoam -parallel")
        #os.system("reconstructPar")
        #os.system("rm -r processor*")



        #############################
        #将最后一步结果拷贝到Initial_value中，方便下一个算例的初始化
        #os.system("cp -r ./15000/.  ../Initial_Value/." )
        #os.system("rm -r ../Initial_value/uniform")

        # change the root directory
        os.chdir(M.ROOT_DIR)

    return

def POSTPROCESSOR(Case_directory):
    
    #Doing postprocess with processors of OpenFOAM
    CASE_DIR = Case_directory
    ROOT_DIR = M.ROOT_DIR
    #ROOT_DIR = "/media/uqjqi/Janry_Research/Stator_Optimization/"
    os.chdir(CASE_DIR)
    
    os.system("sonicFoam -postProcess -func MachNo -latestTime > MachNoLatestTime")
    #os.system("wallCellVelocity -latestTime")

    #接下来这一部分是把T的进口边界条件改成ZeroGradient, 以用来使用Mach
    f_latest = open('MachNoLatestTime', 'r')
    for line in f_latest:
        if 'Create mesh for time =' in line:
            timeString = line

    TimeList = timeString.replace('\n','').split(' ')
    latestTime = TimeList[5]
    #print latestTime

    #os.chdir(CASE_DIR+ '/' +latestTime)
    #f_T = open('T','r')
    #f_T_new= open('Tnew','w')
    #for line in f_T:
    #    if 'isentropic' in line:
    #        line = "        type            zeroGradient;"
    #    f_T_new.write("%s" % line + '\r\n')
    #f_T.close()
    #f_T_new.close()
    #os.system("mv T OLDT")
    #os.system("mv Tnew T")
    #os.chdir(CASE_DIR)


    #os.system("entropyIdeal -latestTime")
    #os.system("Mach -latestTime")
    #os.system("massFlowRate -latestTime > mdot")

    
    os.chdir(ROOT_DIR)

    return latestTime




def File_Reader(Filed_data_type,Field_name,Case_directory,time):


    scalar_list = []
    vector_list = []

    file_path = Case_directory + "/" + str(time) + "/" + str(Field_name)

    f = open(file_path,'r')

    line_count = 0; start_line = -1000

    if Field_name == 'p':
        for line in f:
            line_count += 1
            if "OF_outlet_00" in line:

                # This returns the starting line number 
                start_line = line_count

            if line_count == (start_line + 8):
                list_length = int(line.replace('\r','').replace('\n',''))
    else:
        for line in f:
            line_count += 1
            if "OF_outlet_00" in line:

                # This returns the starting line number 
                start_line = line_count

            if line_count == (start_line + 4):
                list_length = int(line.replace('\r','').replace('\n',''))        
    f.close()


    # read fields file and return a list
    if ("Scalar" == Filed_data_type) and (Field_name == 'p'):

        f_scalar = open(file_path,'r')
        line_count = 0
        for line in f_scalar:
            line_count += 1
            if (line_count >= (start_line + 10)) and (line_count <= (start_line + 10 + list_length -1)):
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

def Flux_Weighted_Average_Scalar(density_list,velocity_list,area_list,face_normal_list,scalar_list):
    
    # First determin the length of each list is equal or not:

    len1 = len(density_list)
    len2 = len(velocity_list)
    len3 = len(area_list)
    len4 = len(face_normal_list)
    len5 = len(scalar_list)

    numerator_flux_weighted     = 0
    denominator_flux_weighted   = 0

    if (len1 == len2) and (len2 == len3) and (len3 == len4) and (len4 == len5):
        for i in range(len1):
            # Calculate the U component on the surface normal direction for each cell face:
            cell_absolute_velocity_meridional_component = (float(velocity_list[i][0])*float(face_normal_list[i][0]) +\
                                                         float(velocity_list[i][1])*float(face_normal_list[i][1]) +\
                                                         float(velocity_list[i][2])*float(face_normal_list[i][2]))/ \
                                                         area_list[i]

            # Sum of cell_velocity_meridional_component times the cell area, that is the area weighted avarage
            numerator_flux_weighted    = numerator_flux_weighted + cell_absolute_velocity_meridional_component* float(scalar_list[i]) * area_list[i] * float(density_list[i])
            denominator_flux_weighted  = denominator_flux_weighted + abs(cell_absolute_velocity_meridional_component) * area_list[i] * float(density_list[i])

        flux_weighted_average_scalar = numerator_flux_weighted/denominator_flux_weighted

    return flux_weighted_average_scalar



#def COST_EVALUATION(Mach_target, alpha_target, mdot_target, p0_target, Mach, alpha, mdot, p0):

def COST_EVALUATION(Mach_target, Mach):

    Cost_list = []

    Mach_cost    =  ((Mach_target - Mach)/Mach_target)**2

    #alpha_cost   =  ((alpha_target - alpha)/alpha_target)**2

    #mdot_cost    =  ((mdot_target - mdot)/mdot_target)**2

    #p0_cost      =  ((p0_target - p0)/p0_target)**2

    Cost_list.append(Mach_cost)
    #cost_list.append(alpha_cost)
    #cost_list.append(mdot_cost)
    #cost_list.append(p0_cost)

    return Cost_list



def COMPARE_VECTOR(List0,List1,VT):
    # Use Mahala-Nobis method to evaluate the distance
    # To be remind that the two list shoud have the same length
    return  distance.mahalanobis(List0,List1,VT)




def Intergrand(t, mu, sigma_sqaure):
    # Cumulative distribution function for the normal distribution
    return np.exp(- (t - mu)**2. / (2.*sigma_sqaure) )/ np.sqrt(2.*np.pi*sigma_sqaure)




def Calculation(x,M,C,D):

    # set the iteration counter
    D.Counter = D.Counter+1

    if M.verbosity > 1:
        print("------------------------------------------------")
        print('Starting Calculation Loop')  
        print('Iteration: {}'.format(D.Counter))
        print('Current Geometry:', x.tolist())




    #加上一些flag，来控制之后是否运行新的case
    flag = 'None'
    
    #Target value are defined here
    #在这儿定义目标值
    Mach_target  = 1.5 
    alpha_target = 69.  # degree
    mdot_target  = 0.036
    p0_target    = 20.e6

    #Weighting factors are defined here
    #定义weighting factor
    W_ma    = 20.
    W_alpha = 40.
    W_mdot  = 100.
    W_p0    = 20.




    # Changing the 
    # 将目标的 INDEX 数变成当前的路经值
    global GLOBAL_INDEX , GLOBAL_DIR , GLOBAL_COUNT 

    global ITERATION, RESIDUAL

    GLOBAL_DIR    = M.ROOT_DIR +  '/' + C.ChildCase_folder_base + str(GLOBAL_INDEX)

    print "-------------",GLOBAL_DIR

    # 对比当前的尺寸値的已经有列表值，如果已经存在，那么就用已经存的值替代当前值
    DATA_LIST, Index, Status, Geometry, Post, Cost, Directory, Method  = load_data(M,C,D)


    #这个地方使用不同的flag来标记最新的数据点是否根已有数据点的马式距离足够接近。如果足够接近的话，使用不同的策略进行计算
    # Mahalanobis norm Assessment
    if sum(Method) == 0:  # only need to do Mahalanobis Norm Assessment if Solutions exist that have been evaluated by CFD 
        # No previous data is available.
        flag = 'evaluation_new'
    else:
        if M.verbosity > 1:
            print("Heading to distance evaluation now...")

        raise MyError('Mahalanobis Calculation Currently not implemented. Need to ensure proper selection of reference cases.')

        #这一步将现在的Geometry与之前所有的进行比较，把得到的马式距离放到一个list里
        distance_list = []
        index = []
        for i in range(len(Geometry)):
            if Method[i] == 1: # only consider cases that have real evaluations
                # TODO: calculate Mahalanobis distance only using Geometries that have 'e'
                # TODO: Set-up correct inverse of covaruiance and check correct covariance is calculated
                # distance_list.append( COMPARE_VECTOR(x.tolist(),Geometry[i],COV_MATRIX.T) )#注意此处使用转置的斜方差矩阵
                distance_list.append( COMPARE_VECTOR(x.tolist(),Geometry[i],COV_MATRIX.T) )
                index.append(i)

        #得到所有的马式距离之后，就要看是否有点落到相应的区间了
        if M.verbosity > 1:
            print("After distance evaluation...")


        #找到最小的马式距离
        #find the minimum md distance
        minimum_distance = min(distance_list)  
        temp = distance_list.index(min(distance_list))
        min_index = index[temp]
        if M.verbosity > 1:
            print('Closest Geometry is: {i}, with Mahalanobis Norm: {f}'.format(min_index,minimum_distance))

        #    #把distance_list进行排序，返回从小到大的变量的index
        #    #Sort the distance_list, return an array of index
        #    sorted_index_array = np.argsort(distance_list,kind='quicksort')
        #    Geometry_New = []
        #    for i in range(len(sorted_index_array)):
        #        if i < 70:
        #            Geometry_New.append(Geometry[sorted_index_array[i]])
        #
        #    #print "The 80 geometry for interpolation:", Geometry_New,len(Geometry_New)

        if minimum_distance == 0.0 :
            # point has been visited previously.
            COST_FUNC = (Cost[min_index][0])* W_ma + (Cost[min_index][1]) * W_alpha + (Cost[min_index][2])*W_mdot + (Cost[min_index][3])*W_p0      

        elif minimum_distance < M.cutoff_0:
            # point is very close to previously visited. Always interpolate. 
            flag = 'interpolation'      

        elif  minimum_distance < M.cutoff_1: 
            # point is in proximite of previous evaluations. Toss coin.         
            print("minimum distance is:", minimum_distance)
            #这一步是让距离从0～0.01 整合为 0 ～0.1,以适应0～0.1的概率累计方程
            a = minimum_distance  * 10. 

            # cumulative distrubution function
            # we use the cumulative distrubution fuction for the normal distrubution.
            possibility_criteria =  (integrate.quad(Intergrand, -np.inf, a , args = (0.085 , 0.00005) ) )[0]

            # 开始扔硬币！
            b = random()
            print "The coin is:",b," and the possibility criteria is:", possibility_criteria

            if b >= possibility_criteria:
                print("-----Do the interpolation!")
                flag = 'interpolation'
            else:
                print("-----Run the case!")
                flag = 'evaluation'

        else:
            # point is far away. Always evaluate. 
            flag = 'evaluation'

    # Then starting interpolation or evaluation
    #接下来开始执行插值或者计算

    if M.allow_interpolation == False and 'interpolation' == flag:
        # overwrite interpolation by evaluation to ensure all functions are evaluated.  
        flag = 'evaluation'

    if 'interpolation' == flag:

        if M.verbosity > 1:
            print("Evaluation by Interpolation")

        raise MyError('interpolation not yet implemented')  

        temp_Ma = []
        temp_P  = []
        temp_Alpha = []
        temp_Mdot  = []


        #Geometry_fifty = []
        # 将距离最小的50个值的 后处理值给定 
        #
        for i in range(len(sorted_index_array)):
            if i < 70:
                #Geometry_fifty.append(Geometry[sorted_index_array[i]])
                temp_Ma.append( Post[sorted_index_array[i]][0]  )
                temp_Alpha.append(Post[sorted_index_array[i]][1]  )
                temp_Mdot.append( Post[sorted_index_array[i]][2])
                temp_P.append( Post[sorted_index_array[i]][3] )

            value_Ma     = np.array(temp_Ma)#; #print values
            value_Alpha  = np.array(temp_Alpha)
            value_Mdot   = np.array(temp_Mdot)
            value_P      = np.array(temp_P)

            xi  = x
            #print xi.shape

            #print "The geometry is:", Geometry, len(Geometry)
            #1/0
            Ma_interpolate    = griddata(Geometry_New, value_Ma,    xi, method='linear',rescale=True)
            Alpha_interpolate = griddata(Geometry_New, value_Alpha, xi, method='linear',rescale=True)
            Mdot_interpolate  = griddata(Geometry_New, value_Mdot,  xi, method='linear',rescale=True )
            P_interpolate     = griddata(Geometry_New, value_P,     xi, method='linear',rescale=True)


            #print "!!!!!Ma=", Ma_interpolate,Alpha_interpolate,Mdot_interpolate,P_interpolate
            Cost_list = []
            Cost_list = COST_EVALUATION(Mach_target, alpha_target, mdot_target, p0_target, float(Ma_interpolate.tolist()[0]), float(Alpha_interpolate.tolist()[0]),float(Mdot_interpolate.tolist()[0]), float(P_interpolate.tolist()[0])  )


            Post_list = []
            Post_list.append(float(Ma_interpolate.tolist()[0]))
            Post_list.append(float(Alpha_interpolate.tolist()[0]))
            Post_list.append(float(Mdot_interpolate.tolist()[0]))
            Post_list.append(float(P_interpolate.tolist()[0]))
            
      

            COST_FUNC = (Cost_list[0])* W_ma + (Cost_list[1]) * W_alpha + (Cost_list[2])*W_mdot + (Cost_list[3])*W_p0

            #如果插值在hall之外，那么就把flag变成evaluation

            if np.isnan( COST_FUNC ):
                print "TRY to extrapolation!!!!!! rerun the case."
                flag = 'evaluation'#'interpolation'

            else:
                print "The total cost is:", COST_FUNC, " and cost is, Ma:", Cost_list[0], " alpha:", Cost_list[1] , " mdot:", Cost_list[2], " p0:", Cost_list[3]

                ITERATION.append(GLOBAL_INDEX)
                RESIDUAL.append(COST_FUNC)


                write = [GLOBAL_INDEX] + [0] + x.tolist() + Post_list + Cost_list + [GLOBAL_DIR] + ['i']
                print write

                # 写文件，算一步写一步
                f = open("dataList_New",'a+')
                f.write("%s" % write + '\r\n')
                f.close()
                GLOBAL_INDEX += 1
           
                
    if 'evaluation' == flag or 'evaluation_new' == flag: 
        if M.verbosity > 1:
            print("Evaluation by Evaluation")

        if flag == 'evaluation' and C.Keep_ChildCase == True: # initiaise simualtion from nearest case
            raise MyError('evaluation, staring from nearest not yet implemented')           
            #找到最近的evaluation 的路径
            # Find the neareast distance directory

            #sorted_index_array = np.argsort(distance_list,kind='quicksort')
            for i in range(len(sorted_index_array)):
                if Method[sorted_index_array[i]] == 'e':
                    min_evaluation_index = sorted_index_array[i] 
                    break

            neareast_DIR = Directory[min_evaluation_index]
            print neareast_DIR, min_evaluation_index

            EXECUTE(x.tolist(), C.ChildCase_folder_base+str(D.Counter) , neareast_DIR, M,C,D)

        elif flag == 'evaluation_new' or C.Keep_ChildCase == False: 
            # initialise from MasterCase
            #raise MyError('Need to Continue from here') 
            if M.verbosity >1:
                print("Start a new evaluation...")

            EXECUTE(x.tolist(), C.ChildCase_folder_base+str(D.Counter) ,None,M,C,D)
        
            if M.verbosity >1:
                print("Case {} has been evaluated".format(C.ChildCase_folder_base+str(D.Counter)))

        else:
            raise MyError('setting for flag not supported')

        print M.ROOT_DIR + '/' + C.ChildCase_folder_base+str(D.Counter)
        # Execute postprocessing routine to evaluate conditions of current simualtion.
        uoDict = {'--verbosity': M.verbosity, '--case_name': M.ROOT_DIR + '/' + C.ChildCase_folder_base+str(D.Counter)}
        Post_list = Evaluate_means.main(uoDict)

        if M.verbosity > 1:
            print('Evaluate_means.py has been exexcuted on directory {}.'.format(C.ChildCase_folder_base+str(D.Counter)))
            print('Post_list: {}'.format(Post_list))


        Cost_list = []
        Cost_list = COST_EVALUATION(Mach_target, Post_list[0])

        # Updating the cost function
        #Cost[i] = Cost_list

        print Cost_list

        print("index = ", GLOBAL_INDEX)
        print("dir =", GLOBAL_DIR) 
        print("count = ", GLOBAL_COUNT)


        write = [D.Counter] + [0] + x.tolist() + Post_list + Cost_list + [M.ROOT_DIR + '/' + C.ChildCase_folder_base+str(D.Counter)] + ['e']




        # 写文件，算一步写一步

        f = open(M.ROOT_DIR + "/dataList",'a+')
        f.write("%s" % write + '\r\n')
        f.close()


        #COST_FUNC = float(Cost_list[0])* W_ma + float(Cost_list[1]) * W_alpha + float(Cost_list[2])*W_mdot + float(Cost_list[3])*W_p0
        COST_FUNC = float(Cost_list[0])
        #GLOBAL_INDEX += 1

        if M.verbosity >1 :
            print("The case counter is now: {}".format(D.Counter))


    else:
        print " you use wrong entry for the flag"
        pass

    D.Iteration.append(D.Counter)
    D.Residual.append(COST_FUNC)

    #print "The cost is:", COST_FUNC
    return COST_FUNC



###
###
class Model:
    def __init__(self):
        self.test = 0
    ##
    def check(self):
        if not self.test == 0:
            raise MyError('M.testincorrect not specified')
        # add additional checks to assess that basics have been set-up correct;y
###
###
class Case:
    def __init__(self):
        self.MasterCase_folder = []
        self.ChildCase_folder_base = []
        self.Input_filename = []
        self.Keep_ChildCase = []
        self.setup_command = []
        self.simulate_command = []
        self.postprocess_command = []
    ##
    def check(self):
        if not self.MasterCase_folder:
            raise MyError('C.MasterCase_folder not specified')
        if not self.ChildCase_folder_base:
            raise MyError('C.ChildCase_folder_base not specified')
        if not self.Input_filename:
            raise MyError('C.Input_filename not specified')
        if not (self.Keep_ChildCase == True or self.Keep_ChildCase == False):
            raise MyError('C.Keep_ChildCase has to be True or False')
        if not self.setup_command:
            raise MyError('C.setup_command not specified')
        if not self.simulate_command:
            raise MyError('C.simulate_command not specified')
        if not self.postprocess_command:
            raise MyError('C.postprocess_command not specified')
    # TODO: need to add routines tho check that folders and files exist.
###
###
class Data:
    def __init__(self):
        self.Counter = 0
        self.Use_initial_simplex = []
        self.initial_simplex_filename = []
        self.write_data = []
        self.filename = []
        self.Iteration = []
        self.Residual  = []
    ##
    def check(self):
        if not (self.Use_initial_simplex == True or self.Use_initial_simplex == False):
            raise MyError('D.Use_initial_simplex has to be True or False')
        if not self.initial_simplex_filename:
            raise MyError('D.initial_simplex_filename not specified')
        if not (self.write_data == True or self.Use_inputarray == False):
            raise MyError('D.write_data has to be True or False')
        if not self.filename:
            raise MyError('C.filename not specified')

    # TODO: need to add routines tho check that folders and files exist.
###
###
def plot_outcomes(M,C,D,Geometry):

    Iteration = D.Iteration
    Residual  = D.Residual

    plt.figure(figsize=(8,4))
    plt.grid()
    plt.plot(Iteration,Residual,'k-',markersize=9,markerfacecolor='w',markeredgewidth=1.5,linewidth=1.5)
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.show()
    print('Plotting function currently not implemented')

    return 0
###
###
def main(uoDict):
    """
    main function
    """
    # create string to collect warning messages
    warn_str = "\n"

    # main file to be executed
    jobFileName = uoDict.get("--job", "test")

    # strip .py extension from jobName
    jobName = jobFileName.split('.')[0]

    # create classes to store input data
    M = Model() # data definig optimisation run.
    C = Case()  # data that defines CFD Case that is to be executed
    D = Data()  # Results from Optimisation process

    # set verbosity (can be overwritten from jobfile)
    M.verbosity = 1
    if "--verbosity" in uoDict:
        M.verbosity = int(uoDict.get("--verbosity", 1))

    # set ROOT 
    M.ROOT_DIR = os.getcwd()
    if M.verbosity > 1:
        print('M.ROOT_DIR set to: {}'.format(M.ROOT_DIR))  


    # Execute jobFile, this creates all the variables
    exec(open(M.ROOT_DIR+'/'+jobFileName).read(),globals(),locals())  

    # check that input data has been set correctly.
    M.check()
    C.check()
    D.check()    

    # create initial simplex to start optimiser   
    if D.Use_initial_simplex:
        # Load initialarray from file
        with open(M.ROOT_DIR+'/'+D.initial_simplex_filename) as f:
            read_data = f.read()
        if M.verbosity > 1:
            print('initial array loaded from: {}'.format(D.initial_simplex_filename))
        exec(read_data) 

    else: 
        # load the last Nvar+1 lines from the existing geometry map
        # 选取最新的12个尺寸列表，用来初始化optimiser
        D.initial_simplex = np.array(Geometry[:(M.Nvar+1)])
        if M.verbosity > 1:
            print('initial_simplex taken from existing Geometry')

    if M.verbosity > 1:
        print('initial_simplex loaded:')
        print(' D.initial_simplex =', D.initial_simplex)


    # load_data 
    DATA_LIST, Index, Status, Geometry, Post, Cost, Directory, Method  = load_data(M,C,D)

    # set inital vector
    x0 = Geometry[-1]   # Should this be replaced by last non evaluated function

    if M.verbosity >1:
        print('Optimisation starting ...')
    # run optimiser
    res =  minimize(Calculation,x0,args=(M,C,D),method='Nelder-Mead',options={'initial_simplex': D.initial_simplex, 'maxiter':M.maxiter})

    if M.verbosity > 1:
        print('Optimisation complete')
        print('Final Residual: {:f}'.format(res))


    # plot outcomes from optimisation process
    plot_outcomes(M,C,D,Geometry)

    return 0
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
longOptions = ["help", "job=", "verbosity="]
###
###
def printUsage():
    print("")
    print("Usage: Optimizer.py [--help] [--job=<jobFileName>] [--verbosity=<0,1,2>]")
    print("\n")
    print(" --help      Display help.")
    print("\n")
    print(" --job=      Use this to specify the job file.")
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


