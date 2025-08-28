import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import metrics
import cv2 as cv


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_roc():
    lw = 2
    cls = ['CNN', 'DCNet', 'RIC-Net', 'MobileNet-V2', 'proposed ARMNet-IIEOA']
    colors = cycle(["m", "b", "r", "lime", "k"])
    Predicted = np.load('roc_score.npy', allow_pickle=True)
    Actual = np.load('roc_act.npy', allow_pickle=True)
    for i in range(len(Actual)):
        Dataset = ['Dataset 1', 'Dataset 2']
        for j, color in zip(range(5), colors):
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[i, 3, j], Predicted[i, 3, j])
            auc = metrics.roc_auc_score(Actual[i, 3, j], Predicted[i, 3, j])
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[j]
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/Roc.png"
        plt.savefig(path)
        plt.show()


def Plot_Table():
    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'pt', 'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Table_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    Algorithm = ['TERMS', 'RSA-ARMNet', 'SCO-ARMNet', 'SGO-ARMNet', 'EOAA-ARMNet', 'IIEOA-ARMNet']
    Activation = ['1', '2', '3', '4', '5']
    Classifier = ['TERMS', 'CNN', 'DCNet', 'RIC-Net', 'MobileNet-V2', 'proposed ARMNet-IIEOA']
    for n in range(1):
        for i in range(eval.shape[1]):
            value = eval[n, i, :, 4:]
            Table = PrettyTable()
            Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Terms)])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[j, Table_Terms])
            print('-------------------------------------------------- ', Activation[i],
                  ' FOLD ',
                  'Algorithm Comparison --------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Terms)])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Terms])
            print('-------------------------------------------------- ', Activation[i],
                  ' FOLD ',
                  'Classifier Comparison --------------------------------------------------')
            print(Table)


def Plot_Batchsize():
    eval = np.load('Eval_ALL.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'pt',
             'G means', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Term = [0, 1, 2, 3, 4, 7, 9, 11, 12, 16]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[n, k, l, Graph_Term[j] + 4]
            length = np.arange(6)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='>', markerfacecolor='red',
                    markersize=12,
                    label='RSA-ARMNet')
            ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='>', markerfacecolor='green',
                    markersize=12,
                    label='SCO-ARMNet')
            ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='>', markerfacecolor='cyan',
                    markersize=12,
                    label='SGO-ARMNet')
            ax.plot(length, Graph[:, 3], color='#f504c9', linewidth=3, marker='>', markerfacecolor='#fdff38',
                    markersize=12,
                    label='EOAA-ARMNet')
            ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='>', markerfacecolor='w', markersize=12,
                    label='IIEOA-ARMNet')

            ax.fill_between(length, Graph[:, 0], Graph[:, 3], color='#acc2d9', alpha=.5)
            ax.fill_between(length, Graph[:, 3], Graph[:, 2], color='#c48efd', alpha=.5)
            ax.fill_between(length, Graph[:, 2], Graph[:, 1], color='#be03fd', alpha=.5)
            ax.fill_between(length, Graph[:, 1], Graph[:, 4], color='#b2fba5', alpha=.5)
            plt.xticks(length, ('4', '8', '16', '32', '48', '64'))
            plt.xlabel('Batch Size', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path = "./Results/Epoch_%s_Alg.png" % (Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(6)
            ax.bar(X + 0.00, Graph[:, 5], edgecolor='k', hatch='//', color='b', width=0.10, label="CNN")
            ax.bar(X + 0.10, Graph[:, 6], edgecolor='k', hatch='-', color='#6dedfd', width=0.10, label="DCNet")
            ax.bar(X + 0.20, Graph[:, 7], edgecolor='k', hatch='//', color='lime', width=0.10, label="RIC-Net")
            ax.bar(X + 0.30, Graph[:, 8], edgecolor='k', hatch='-', color='#ed0dd9', width=0.10, label="MobileNet-V2")
            ax.bar(X + 0.40, Graph[:, 9], edgecolor='w', hatch='..', color='k', width=0.10, label="proposed ARMNet-IIEOA")
            plt.xticks(X + 0.25, ('4', '8', '16', '32', '48', '64'))
            plt.xlabel('Batch Size')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/Epoch_%s_Med.png" % (Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()


def plot_results_conv():
    Result = np.load('Fitness.npy', allow_pickle=True)
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'RSA-ARMNet', 'SCO-ARMNet', 'SGO-ARMNet', 'EOAA-ARMNet', 'IIEOA-ARMNet']
    for i in range(Result.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- ', 'Statistical Report ',
              '--------------------------------------------------')
        print(Table)
        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='.', markerfacecolor='red', markersize=12,
                 label='RSA-ARMNet')
        plt.plot(length, Conv_Graph[1, :], color='c', linewidth=3, marker='.', markerfacecolor='green',
                 markersize=12,
                 label='SCO-ARMNet')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='.', markerfacecolor='cyan',
                 markersize=12,
                 label='SGO-ARMNet')
        plt.plot(length, Conv_Graph[3, :], color='y', linewidth=3, marker='.', markerfacecolor='magenta',
                 markersize=12,
                 label='EOAA-ARMNet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='.', markerfacecolor='black',
                 markersize=12,
                 label='IIEOA-ARMNet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Convergence.png")
        plt.show()


def plot_results_seg():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
             'NPV',
             'FDR', 'F1-Score', 'MCC']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 5, :], color='#cb00f5', hatch='*', edgecolor='k', width=0.10, label="UNet")
            ax.bar(X + 0.10, stats[i, 6, :], color='lime', hatch='*', edgecolor='k', width=0.10, label="UNet3+")
            ax.bar(X + 0.20, stats[i, 7, :], color='r', hatch='*', edgecolor='k', width=0.10, label="TransUnet")
            ax.bar(X + 0.30, stats[i, 8, :], color='c', hatch='*', edgecolor='k', width=0.10, label="Trans-Unet++")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', hatch='\\', edgecolor='w', width=0.10, label="DTMUnet++")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/seg_%s_met.png" % (Terms[i - 4])
            plt.savefig(path1)
            plt.show()



def Image_Results():
    Images = np.load('Images_1.npy', allow_pickle=True)
    Unet = np.load('segment_unet.npy', allow_pickle=True)
    Unet3 = np.load('segment_Unet3.npy', allow_pickle=True)
    Trans = np.load('segment_Transunet.npy', allow_pickle=True)
    without = np.load('segment_Transunet++.npy', allow_pickle=True)
    Segemt = np.load('Segmentation_1.npy', allow_pickle=True)

    Image = [101, 102, 112, 109, 100]
    for i in range(len(Image)):
        plt.subplot(2, 3, 1)
        plt.title('Original')
        plt.imshow(Images[Image[i] - 1])
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.title('Unet')
        plt.imshow(Unet[Image[i] - 1])
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.title(' Unet3+ ')
        plt.imshow(Unet3[Image[i] - 1])
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.title('Trans Unet')
        plt.imshow(Trans[Image[i] - 1])
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.title('Transunet++')
        plt.imshow(without[Image[i] - 1])
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.title('Segmentation')
        plt.imshow(Segemt[Image[i] - 1])
        plt.axis('off')

        plt.show()
        cv.imwrite('./Results/Image Results/Dataset1_orig-' + str(i + 1) + '.png', Images[Image[i] - 1])
        cv.imwrite('./Results/Image Results/Dataset1_UNET-' + str(i + 1) + '.png', Unet[Image[i] - 1])
        cv.imwrite('./Results/Image Results/Dataset1_UNET3+-' + str(i + 1) + '.png', Unet3[Image[i] - 1])
        cv.imwrite('./Results/Image Results/Dataset1_Trans-' + str(i + 1) + '.png', Trans[Image[i] - 1])
        cv.imwrite('./Results/Image Results/Dataset1_Trans++-' + str(i + 1) + '.png', without[Image[i] - 1])
        cv.imwrite('./Results/Image Results/Dataset1_segmented-' + str(i + 1) + '.png', Segemt[Image[i] - 1])

if __name__ == '__main__':
    Plot_Table()
    plot_roc()
    Plot_Batchsize()
    plot_results_conv()
    plot_results_seg()
    Image_Results()
