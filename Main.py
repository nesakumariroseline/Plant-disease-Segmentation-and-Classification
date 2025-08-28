import numpy as np
import cv2 as cv
import os
from numpy import matlib
from EOAA import EOAA
from Global_Vars import Global_Vars
from Model_CNN import Model_CNN
from Model_DTMUNeT import Model_TransMobileUNetplusplus
from Model_DenseNet import Model_DenseNet
from Model_PROPOSED import Model_PROPOSED
from Model_RAN import Model_RAN
from Objective_Function import Obj_fun
from PROPOSED import PROPOSED
from Plot_Results import *
from RSA import RSA
from SCO import SCO
from SGO import SGO


def Read_Image(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    image = cv.resize(image, (512, 512))
    return image


def Read_Images(directory_name):
    Fold_Array = os.listdir(directory_name)
    Images = []
    Target = []
    iter = 1
    flag = 0
    for i in range(len(Fold_Array)):
        Img_Array = os.listdir(directory_name + Fold_Array[i])
        for j in range(len(Img_Array)):
            print(i, j)
            image = Read_Image(directory_name + Fold_Array[i] + '/' + Img_Array[j])
            Images.append(image)
            if Fold_Array[i][len(Fold_Array[i]) - 7:] == 'healthy':
                Target.append(0)
            else:
                flag = 1
                Target.append(iter)
        if flag == 1:
            iter = iter + 1
        uniq = np.unique(Target)
        Tar = np.zeros((len(Target), len(uniq)))
        for j in range(len(uniq)):
            Index = np.where(Target == uniq[j])
            Tar[Index[0], j] = 1
    return Images, Tar


no_of_class = 10

# Read Dataset
an = 0
if an == 1:
    Directory = './Dataset/'
    Dataset_List = os.listdir(Directory)
    for n in range(len(Dataset_List)):
        Images, Target = Read_Images(Directory + Dataset_List[n] + '/')
        np.save('Images_' + str(n + 1) + '.npy', Images)
        np.save('Target_' + str(n + 1) + '.npy', Target)

# Segmentation
an = 0
if an == 1:
    for n in range(no_of_class):
        Image = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        seg = Model_TransMobileUNetplusplus(Image, Target)
        np.save('Segmentation_' + str(n + 1) + '.npy', seg)

# OPTIMIZATION
an = 0
if an == 1:
    Bestsol = []
    for n in range(no_of_class):
        Feat = np.load('Segmentation_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Feat_1 = Feat
        Global_Vars.Target = Target
        Npop = 10
        Ch_len = 3
        xmin = matlib.repmat(([5, 5, 0]), Npop, 1)
        xmax = matlib.repmat(([255, 50, 4]), Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(Ch_len):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Obj_fun
        max_iter = 50

        print('RSA....')
        [bestfit1, fitness1, bestsol1, Time1] = RSA(initsol, fname, xmin, xmax, max_iter)

        print('SCO....')
        [bestfit2, fitness2, bestsol2, Time2] = SCO(initsol, fname, xmin, xmax, max_iter)

        print('SGO....')
        [bestfit3, fitness3, bestsol3, Time3] = SGO(initsol, fname, xmin, xmax, max_iter)

        print('EOAA....')
        [bestfit4, fitness4, bestsol4, Time4] = EOAA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

        Sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        Bestsol.append(Sol)
    np.save('BestSol_CLS.npy', np.asarray(Bestsol))


# Classification
an = 1
if an == 1:
    Eval_all = []
    for n in range(no_of_class):
        Feature = np.load('Segmentation_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)[n]
        K = 5
        Per = 1 / 5
        Perc = round(Feature.shape[0] * Per)
        eval = []
        for i in range(K):
            Eval = np.zeros((10, 25))
            for j in range(5):
                Feat = Feature
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
                Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
                test_index = np.arange(i * Perc, ((i + 1) * Perc))
                total_index = np.arange(Feat.shape[0])
                train_index = np.setdiff1d(total_index, test_index)
                Train_Data = Feat[train_index, :]
                Train_Target = Target[train_index, :]
                Eval[j, :] = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval[5, :] = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :] = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :] = Model_RAN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :] = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9, :] = Eval[4, :]
            eval.append(Eval)
        Eval_all.append(eval)
    np.save('Eval_Fold.npy', np.asarray(Eval_all))

Plot_Table()
plot_roc()
Plot_Batchsize()
plot_results_conv()
plot_results_seg()
Image_Results()