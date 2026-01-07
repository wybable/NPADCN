import os
import numpy as np
import pandas as pd
import time

from sympy import public
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime
from torchvision.models.detection.roi_heads import keypointrcnn_loss
import train
from icecream import ic
from Utils import load_dataset, sampling, aa_and_each_accuracy, generate_iter, \
    generate_png, PredictMap_png, record_output, compute_imf_weights, ArcFace, generate_iter_Dateset
from torchsummary import summary
from Deform_net import Deform_net
from thop import profile

# torch.cuda.max_split_size_mb()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# print time




"""===============================================程序运行前准备工作==================================================="""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or CPU
day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')  # 当前具体时间，用于实验记录结果的命名
seeds = [1331, 1332, 1333,  1334, 1335, 1336, 1337, 1338, 1339, 1340]  # 固定随机种子以保证结果可重复 for Monte Carlo runs
# seeds = [1332]

""" ======================================================加载数据====================================================="""
print('-----Importing Dataset-----')
global Dataset  # UP, LK
dataset = input('Please input the name of Dataset(PU, LK):')
start = time.time()  # 开始时间
Dataset = dataset.upper()
data_hsi, gt_hsi, TOTAL_SIZE, VALIDATION_SPLIT = load_dataset(Dataset)
end = time.time()
print('Read data consuming time: %s Seconds' % (end - start))
# VALIDATION_SPLIT = 100
# VALIDATION_SPLIT用于样本划分；<=1时表示测试集所占的比例；>1时表示每个类别都采集VALIDATION_SPLIT个样本；训练数据和验证数据集数目相同
# for IN: data_hsi.shape: (145, 145, 200)  gt_hsi.shape: (145, 145)   TOTAL_SIZE: 10249   VALIDATION_SPLIT: 0.97

'''================================================图像和标签格式转化降维==============================================='''
image_x, image_y, BAND = data_hsi.shape  # 记录原始HSI的图像
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))  # 将原始HSI数据行和列展开到一维
# np.prod 连乘操作，所有元素相乘  data_hsi.shape[:2]: (145, 145) data_hsi.shape[2:]: (200,)
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]))  # 将原始GroundTruth数据行和列展开到一维,与HSI数据对应
# gt_hsi.shape[:2]: (145, 145)

'''===========================================逐波段归一化，reshape成三维数组==========================================='''
scaler = preprocessing.StandardScaler()
data = scaler.fit_transform(data)  # 对每个波段进行归一化处理  # 对每个波段进行归一化处理
whole_data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
# 得到与原始HSI格式相同（三维）的归一化后的数据

'''========================================================设置模型参数==============================================='''
print('-----Importing Setting Parameters-----')
ITER = 1  # 迭代次数
PATCH_LENGTH = 4  # 图像块扩充尺寸
FULL_MAP = False
EARLY_STOP = True  # 是否执行早停策略
EARLY_NUM = 200  # 最好根据学习率的衰减模式进行调节
CLASS_BALANCING = False  # 是否进行样本均衡
if FULL_MAP:
    PredictMap_indices = np.arange(np.prod(gt_hsi.shape[:2]))  # 对图片中所有像素进行索引，用于绘制整张分类图

lr, num_epochs, batch_size = 5e-4, 100, 64  # 学习率 0.0005  训练轮数 200  每批次样本数 32
# lr, num_epochs, batch_size = 5e-4, 200, 64  # con_trans
# lr, num_epochs, batch_size = 0.0003, 200, 32
# lr, num_epochs, batch_size = 0.00050, 200, 16
# lr, num_epochs, batch_size = 0.00050, 200, 64
loss = torch.nn.CrossEntropyLoss()
# loss = ArcFace()

'''==================================================三维图像数组，填充边界============================================='''
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)  # 三个维度，对应维度要填充的行数、列数、通道数等。实际应用中可以镜像复制边界

'''===================================================记录精度和训练时间==============================================='''
CLASSES_NUM = max(gt)  # 类别数，我们一般在GroundTruth中设置背景为0，其余类别从1开始依次递增
print('The class numbers of the HSI data is:', CLASSES_NUM)
KAPPA = []
OA = []
AA = []
CON_MAT = np.zeros((ITER, CLASSES_NUM, CLASSES_NUM))  # 存放每次迭代的混淆矩阵
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))  # 记录每次迭代每种类别的分类精度
TRAINING_TIME = []
TESTING_TIME = []

'''===================================================开始迭代ITER次训练模型==========================================='''


def iter_main(index_iter, BAND, PATCH_LENGTH, CLASSES_NUM, lr, seeds, VALIDATION_SPLIT, gt, CLASS_BALANCING,
              padded_data, batch_size, device, day_str, num_epochs, EARLY_STOP, EARLY_NUM, Dataset, gt_hsi,
              OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME, CON_MAT):
    print('*********************Iter：', index_iter, ' ********************')

    # ==============================每次迭代需要重新设置的参数==================================
    net = Deform_net(BAND, num_classes=CLASSES_NUM, cnn_channel=9,)

    # =========================================计算模型复杂度==================================================
    shap = 0  # b, C, H, W, B
    input_size = (batch_size, 1, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, BAND)

    # summary(net.cuda(), input_size=(input_size[1:]), batch_size=-1)
    # flops, params = profile(net.cuda(), inputs=(torch.randn(input_size).cuda(),))
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    # n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)

    optimizer = optim.AdamW(net.parameters(), lr=lr)  # , amsgrad=False, weight_decay=0.0001)
    np.random.seed(seeds[index_iter])  # 10次训练，设置每次训练的随机种子，保证结果相同

    # =========GT拉平成一维向量，划分训练和测试样本，输出样本标签数组的列表索引==========
    if Dataset == 'AX':
        min_num = 5  # 每类最少样本数
    else:
        min_num = 3
    train_indices, val_indices, test_indices = sampling(VALIDATION_SPLIT, gt, min_num)  # 依据之前的样本划分参数VALIDATION_SPLIT划分训练和测试集
    total_indices = np.concatenate((train_indices, val_indices, test_indices))
    # total_indices[0:5]: [6843, 232, 11754, 7649, 14817]

    # ==================================进行样本平衡=============================
    if CLASS_BALANCING:
        weights = compute_imf_weights(train_indices, gt, CLASSES_NUM)
        weights = torch.from_numpy(weights.astype(np.float32)).to(device)
        print("Class balancing weights are: ", weights)
        loss.weight = weights

    TRAIN_SIZE = len(train_indices)  # 训练集大小 Train size: 307
    VAL_SIZE = len(val_indices)  # 验证集大小 Validation size: 307
    TEST_SIZE = len(test_indices)  # 测试集大小 Test size: 9635  Total size: 10249
    print('Train size: ', TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)
    print('Test size: ', TEST_SIZE)

    # =================================裁剪立方体图像块，数据组织batch=========================================
    print('-----Selecting Small Pieces from the Original Cube Data-----')
    train_iter, valida_iter, test_iter, all_iter, = \
        generate_iter_Dateset(train_indices, test_indices, val_indices, whole_data, PATCH_LENGTH, padded_data, BAND,
                              batch_size, gt, SHAPE=shap)

    # =========================================输出网络结构并开始训练网络==================================================
    # with torch.no_grad():
    #     for input, _ in train_iter:
    #         break
    #     summary(net.to(device), input.size()[1:])
    # We would like to use device=hyperparams['device'] altough we have
    # to wait for torchsummary to be fixed first.
    tic1 = time.time()
    train.train(net, train_iter, valida_iter, loss, optimizer, device, index_iter, day_str, epochs=num_epochs,
                early_stopping=EARLY_STOP, early_num=EARLY_NUM, datasetname=Dataset)  # 输出训练精度曲线，打印训练精度，保存训练模型
    toc1 = time.time()

    # =========================================测试最终模型==================================================
    pred_test = torch.tensor([]).to(device)
    net.eval()
    tic2 = time.time()
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device).float()  # 评估模式, 这会关闭dropout
            y_hat = net(X)
            if isinstance(y_hat, list):
                pred_test = torch.cat([pred_test, (y_hat[0] + y_hat[1]).argmax(axis=1)])
            else:
                pred_test = torch.cat([pred_test, y_hat.argmax(axis=1)])
    toc2 = time.time()
    pred_test = np.array(pred_test.cpu())

    # ======================================测试集评价指标计算=================================================
    gt_test = gt[test_indices] - 1
    overall_acc = metrics.accuracy_score(gt_test, pred_test, )
    confusion_matrix = metrics.confusion_matrix(gt_test, pred_test, )
    CON_MAT[index_iter] = list(confusion_matrix.astype(int))
    each_acc, average_acc, con_mat = aa_and_each_accuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(gt_test, pred_test, )
    OA.append(overall_acc)
    AA.append(average_acc)
    ELEMENT_ACC[index_iter, :] = each_acc
    KAPPA.append(kappa)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)

    # =========================================保存混淆矩阵=================================================
    con_mat = pd.DataFrame(confusion_matrix)
    excel_writer = pd.ExcelWriter('record/' + day_str + '_' + net.name + '_' + Dataset + '_confusion_' +
                                  str(index_iter) + '_' + str(round(overall_acc, 3)) + '.xlsx')
    con_mat.to_excel(excel_writer, "page_1", float_format='%.5f')
    excel_writer.save()
    excel_writer.close()

    # ==========================================模型保存======================================================
    torch.save(net.state_dict(), "./net/" + day_str + '_' + net.name + '_' + Dataset + '_' + str(index_iter) +
               '_' + str(round(overall_acc, 3)) + '.pt')

    # =================================== 标签样本预测分类图绘制=================================================
    generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices, index_iter, day_str)

    # =================================== 所有像素预测分类图绘制=================================================
    torch.cuda.empty_cache()
    return net.name


for index_iter in range(ITER):
    net_name = iter_main(index_iter, BAND, PATCH_LENGTH, CLASSES_NUM, lr, seeds, VALIDATION_SPLIT, gt, CLASS_BALANCING,
                         padded_data, batch_size, device, day_str, num_epochs, EARLY_STOP, EARLY_NUM, Dataset, gt_hsi,
                         OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME, CON_MAT)

print("--------" + net_name + " Training Finished-----------")

'''===================================================记录模型的评价指标==============================================='''
record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME, CON_MAT,
              'record/' + day_str + '_' + net_name + '_' +
              Dataset + '_' + 'Val_split：' + str(VALIDATION_SPLIT) + '_' + 'lr：' + str(lr) + '.txt')
end = time.time()
print('Running time: %s Seconds' % (end - start))
print('Running time: %s Minutes' % ((end - start) / 60))
print('Running time: %s Hours' % ((end - start) / 3600))
