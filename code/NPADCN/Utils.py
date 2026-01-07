import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
import torch.utils.data as Data
from icecream import ic
from collections import Counter
from RasterRead import IMAGE
import gc
import timm


""" =================================================加载数据========================================================="""


def load_dataset(Dataset):
    global data_hsi, gt_hsi, TOTAL_SIZE, VALIDATION_SPLIT
    if Dataset == 'PU':
        uPavia = sio.loadmat('../datasets/PaviaU.mat')
        gt_uPavia = sio.loadmat('../datasets/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        # TOTAL_SIZE = 42776
        TOTAL_SIZE = np.count_nonzero(gt_hsi)
        VALIDATION_SPLIT = 0.995
        # TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'LK':
        data_proj, data_geotrans, data_hsi = IMAGE().read_img(r'../datasets/WHU-Hi-LongKou.tif')
        gt_proj, gt_geotrans, gt_hsi = IMAGE().read_img(r'../datasets/WHU-Hi-LongKou_gt.tif')
        data_hsi = np.rollaxis(data_hsi, 0, 3)
        TOTAL_SIZE = np.count_nonzero(gt_hsi)
        VALIDATION_SPLIT = 25  # 0.998
        # TRAIN_SIZE = math.ceil(TOTAL_SIZE * (1-VALIDATION_SPLIT))

    if Dataset == 'AX':
        data_proj, data_geotrans, data_hsi = IMAGE().read_img(r'../datasets/AnXin_Image.dat')
        gt_proj, gt_geotrans, gt_hsi = IMAGE().read_img(r'../datasets/AnXin_Label.dat')
        data_hsi = np.rollaxis(data_hsi, 0, 3)
        TOTAL_SIZE = np.count_nonzero(gt_hsi)
        VALIDATION_SPLIT = 0.99
        # TRAIN_SIZE = math.ceil(TOTAL_SIZE * (1-VALIDATION_SPLIT))

    return data_hsi, gt_hsi, TOTAL_SIZE, VALIDATION_SPLIT


""" =================================================样本划分========================================================="""


def sampling(proportion, ground_truth, min):
    train = {}
    val = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]  # ravel()拉成一维数组
        np.random.shuffle(indexes)
        # print(len(indexes))
        labels_loc[i] = indexes
        if proportion < 1:
            nb_val = max(int((1 - proportion) * len(indexes)), min)  # 每一类至少取3个像素作为样本
            ic(nb_val)  # 3  42  24  7  14  21  3  14  3  29  73  17  6  37  11  3
        elif proportion > 1:
            if 2*proportion+1 <= len(indexes):
                nb_val = int(proportion)
            else:
                nb_val = len(indexes)//3
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        val[i] = indexes[nb_val:2*nb_val]
        test[i] = indexes[2*nb_val:]
    train_indexes = []
    val_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
        val_indexes += val[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    np.random.shuffle(val_indexes)
    return train_indexes, val_indexes, test_indexes


""" ======================定义立方体裁剪函数===extract_samll_cubic.py=========================================="""


def index_assignment(index, row, col, pad_length):  # index[0:5]: [6843, 232, 11754, 7649, 14817]  列向量转变成行、列
    # ic(index[0:5])
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length  # 6843/145+4=47+4=51 取整
        assign_1 = value % col + pad_length  # 6843%145 +4 =28+4=32  取余
        new_assign[counter] = [assign_0, assign_1]
    # ic(len(new_assign))  # len(new_assign): 10249
    return new_assign  # new_assign: {0: [51, 32], 1: [5, 91],}


# def assignment_index(assign_0, assign_1, col):
#     new_index = assign_0 * col + assign_1
#     return new_index


def select_patch(matrix, pos_row, pos_col, ex_len):  # 根据行列号提取图像块
    selected_patch = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)][:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    # selected_patch.shape: (9, 9, 200)
    return selected_patch


# 提取所有已标记像素的所有图像
def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):  # data_size: 10249      data_indices[0:5]: [6843, 232, 11754, 7649, 14817]      whole_data.shape: (145, 145, 200)      patch_length: 4     padded_data.shape: (153, 153, 200)      dimension: 200
    # ic(data_size, data_indices[0:5], whole_data.shape, patch_length, padded_data.shape, dimension)
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))  # small_cubic_data.shape: (10249, 9, 9, 200)
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
    for i in range(len(data_assign)):  # len(data_assign): 10249
        small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data  # small_cubic_data.shape: (10249, 9, 9, 200)


# ===============================bandpatch处理（SpectralFormer中的group-wise embedding）==================================
def gain_neighborhood_band(x_train, band_patch, patch=5):
    band = x_train.shape[3]
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)  # 将第二维拉平
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band), dtype=float)
    # 中心区域
    x_train_band[:, nn*patch*patch:(nn+1)*patch*patch, :] = x_train_reshape
    # 左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, i*patch*patch:(i+1)*patch*patch, :nn-i] = x_train_reshape[:, :, band-nn+i:]
            x_train_band[:, i*patch*patch:(i+1)*patch*patch, nn-i:] = x_train_reshape[:, :, :band-nn+i]
        else:
            x_train_band[:, i:(i+1), :(nn-i)] = x_train_reshape[:, 0:1, (band-nn+i):]
            x_train_band[:, i:(i+1), (nn-i):] = x_train_reshape[:, 0:1, :(band-nn+i)]
    # 右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:, (nn+i+1)*patch*patch:(nn+i+2)*patch*patch, :band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:, (nn+i+1)*patch*patch:(nn+i+2)*patch*patch, band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:, (nn+1+i):(nn+2+i), (band-i-1):] = x_train_reshape[:, 0:1, :(i+1)]
            x_train_band[:, (nn+1+i):(nn+2+i), :(band-i-1)] = x_train_reshape[:, 0:1, (i+1):]
    return x_train_band


""" ==============================================样本迭代器生成======================================================="""


def generate_iter(train_indices,  test_indices,  val_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION,
                  batch_size, gt, SHAPE = 0):
    # ic | TRAIN_SIZE: 307      # train_indices[0:5]: [5396, 20334, 16590, 4525, 5975]
    # TEST_SIZE: 9942            # test_indices[0:5]: [15433, 14413, 11540, 2581, 14705]
    # TOTAL_SIZE: 10249       # total_indices[0:5]: [6843, 232, 11754, 7649, 14817]
    # VAL_SIZE: 307                # whole_data.shape: (145, 145, 200)
    # PATCH_LENGTH: 4       # padded_data.shape: (153, 153, 200)
    # INPUT_DIMENSION: 200         # batch_size: 16
    # gt.shape: (21025,)            # gt_hsi.shape: (145, 145)
    # ic(np.max(train_indices), np.max(test_indices), np.max(total_indices))
    TRAIN_SIZE = len(train_indices)
    VAL_SIZE = len(val_indices)
    TEST_SIZE = len(test_indices)

    y_train = gt[train_indices] - 1
    y_val = gt[val_indices] - 1
    y_test = gt[test_indices] - 1
    gt_all = np.concatenate((y_train, y_val, y_test))

    # all_data = select_small_cubic(TOTAL_SIZE, total_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)  # small_cubic_data.shape: (10249, 9, 9, 200)
    train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)  # small_cubic_data.shape: (307, 9, 9, 200)
    test_data = select_small_cubic(TEST_SIZE, test_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)  # small_cubic_data.shape: (9942, 9, 9, 200)
    val_data = select_small_cubic(VAL_SIZE, val_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    all_data = np.concatenate((train_data, val_data, test_data))

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)  # x_train.shape: (307, 9, 9, 200)
    x_val = val_data.reshape(val_data.shape[0], val_data.shape[1], val_data.shape[2], INPUT_DIMENSION)
    x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)  # x_test_all.shape: (9942, 9, 9, 200)

    if SHAPE == 1:
        x1_tensor_train = torch.from_numpy(x_train).permute(0, 3, 1, 2).type(torch.FloatTensor).unsqueeze(1)  # (307, 1, 200, 9, 9)
    elif SHAPE == 2:
        x_train = gain_neighborhood_band(x_train, 3, PATCH_LENGTH*2+1)
        x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    else:
        x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)  # (307, 1, 9, 9, 200)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)  # np.array(y1_tensor_train).shape: (307,)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)  # np.array(torch_dataset_train).shape: (307, 2)
    del x1_tensor_train
    gc.collect()

    if SHAPE == 1:
        x1_tensor_valida = torch.from_numpy(x_val).permute(0, 3, 1, 2).type(torch.FloatTensor).unsqueeze(1)  # (307, 1, 200, 9, 9)
    elif SHAPE == 2:
        x_val = gain_neighborhood_band(x_val, 3, PATCH_LENGTH*2+1)
        x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor)
    else:
        x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)  # (307, 1, 9, 9, 200)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)  # np.array(y1_tensor_valida).shape: (307,)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)  # np.array(torch_dataset_valida).shape: (307, 2)
    # ic(np.array(x1_tensor_valida).shape, np.array(y1_tensor_valida).shape, np.array(torch_dataset_valida).shape)
    del x1_tensor_valida
    gc.collect()

    if SHAPE == 1:
        x1_tensor_test = torch.from_numpy(x_test).permute(0, 3, 1, 2).type(torch.FloatTensor).unsqueeze(1)  # (9635, 1, 200, 9, 9)
    elif SHAPE == 2:
        x_test = gain_neighborhood_band(x_test, 3, PATCH_LENGTH*2+1)
        x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    else:
        x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)  # (9635, 1, 9, 9, 200)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)  # np.array(y1_tensor_test).shape: (9635,)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, y1_tensor_test)  # np.array(torch_dataset_test).shape: (9635, 2)
    # ic(np.array(x1_tensor_test).shape, np.array(y1_tensor_test).shape, np.array(torch_dataset_test).shape)
    del x1_tensor_test
    gc.collect()

    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)  # all_data.shape: (10249, 9, 9, 200)
    if SHAPE == 1:
        all_tensor_data = torch.from_numpy(all_data).permute(0, 3, 1, 2).type(torch.FloatTensor).unsqueeze(1)  # (10249, 200, 9, 9)
    elif SHAPE == 2:
        all_data = gain_neighborhood_band(all_data, 3, PATCH_LENGTH*2+1)
        all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor)
    else:
        all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)  # (10249, 1, 9, 9, 200)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)  # np.array(all_tensor_data_label).shape: (10249,)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)  # np.array(torch_dataset_all).shape: (10249, 2)
    # ic(all_data.shape, np.array(all_tensor_data).shape, np.array(all_tensor_data_label).shape, np.array(torch_dataset_all).shape)
    del all_tensor_data
    gc.collect()

    train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)  # torch TensorDataset format  # mini batch size  # 要不要打乱数据 (打乱比较好)  # 多线程来读数据  #  X.shape: torch.Size([16, 1, 9, 9, 200]), y.shape: torch.Size([16])
    valiada_iter = Data.DataLoader(dataset=torch_dataset_valida, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = Data.DataLoader(dataset=torch_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    all_iter = Data.DataLoader(dataset=torch_dataset_all, batch_size=batch_size, shuffle=False, num_workers=0)
    # ic(all_tensor_data.shape)
    # gc.collect()

    return train_iter, valiada_iter, test_iter, all_iter  # , y_test


""" ============================================根据混淆矩阵计算精度===================================================="""


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    con_mat = (confusion_matrix.T/list_raw_sum).T
    return each_acc, average_acc, con_mat


""" ==============================================分类结果图绘制======================================================="""


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 3:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([64, 0, 64]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == 19:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 20:
            y[index] = np.array([64, 64, 64]) / 255.
        if item == 21:
            y[index] = np.array([64, 128, 192]) / 255.
        if item == 22:
            y[index] = np.array([192, 128, 64]) / 255.
        if item == 23:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 24:
            y[index] = np.array([0, 139, 0]) / 255.
        if item == 25:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 26:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 27:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 28:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 29:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 30:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 31:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 32:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 33:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 34:
            y[index] = np.array([255, 127, 80]) / 255.
        if item == 35:
            y[index] = np.array([127, 255, 0]) / 255.
        if item == 36:
            y[index] = np.array([238, 130, 238]) / 255.
        if item == 37:
            y[index] = np.array([220, 20, 60]) / 255.
        if item == 38:
            y[index] = np.array([46, 139, 87]) / 255.
        if item == 39:
            y[index] = np.array([25, 25, 112]) / 255.
        if item == 40:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 41:
            y[index] = np.array([127, 255, 212]) / 255.
        if item == 42:
            y[index] = np.array([218, 112, 214]) / 255.
        if item == 43:
            y[index] = np.array([216, 191, 216]) / 255.
        if item == 44:
            y[index] = np.array([221, 160, 221]) / 255.
        if item == 45:
            y[index] = np.array([75, 0, 130]) / 255.
        if item == 46:
            y[index] = np.array([65, 105, 225]) / 255.
        if item == 47:
            y[index] = np.array([70, 130, 180]) / 255.
        if item == 48:
            y[index] = np.array([230, 230, 250]) / 255.
        if item == 49:
            y[index] = np.array([240, 255, 255]) / 255.
        if item == 50:
            y[index] = np.array([64, 224, 208]) / 255.
        if item == 51:
            y[index] = np.array([240, 255, 240]) / 255.
        if item == 52:
            y[index] = np.array([34, 139, 34]) / 255.
        if item == 53:
            y[index] = np.array([124, 252, 0]) / 255.
        if item == 54:
            y[index] = np.array([255, 255, 240]) / 255.
        if item == 55:
            y[index] = np.array([240, 230, 140]) / 255.
        if item == 56:
            y[index] = np.array([245, 222, 179]) / 255.
        if item == 57:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 58:
            y[index] = np.array([255, 248, 220]) / 255.
        if item == 59:
            y[index] = np.array([255, 228, 181]) / 255.
        if item == 60:
            y[index] = np.array([255, 218, 185]) / 255.
        if item == 61:
            y[index] = np.array([210, 105, 30]) / 255.
        if item == 62:
            y[index] = np.array([160, 82, 45]) / 255.
        if item == 63:
            y[index] = np.array([255, 228, 225]) / 255.
        if item == 64:
            y[index] = np.array([250, 128, 114]) / 255.
        if item == 65:
            y[index] = np.array([255, 99, 71]) / 255.
        if item == 66:
            y[index] = np.array([255, 192, 203]) / 255.
        if item == 67:
            y[index] = np.array([165, 42, 42]) / 255.
        if item == 68:
            y[index] = np.array([178, 34, 34]) / 255.
        if item == 69:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 70:
            y[index] = np.array([64, 64, 64]) / 255.
        if item == 71:
            y[index] = np.array([64, 128, 192]) / 255.
        if item == 72:
            y[index] = np.array([192, 128, 64]) / 255.
        if item == 73:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 74:
            y[index] = np.array([0, 139, 0]) / 255.
        if item == 75:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 76:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 77:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 78:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 79:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 80:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 81:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 82:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 83:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 84:
            y[index] = np.array([255, 127, 80]) / 255.
        if item == 85:
            y[index] = np.array([127, 255, 0]) / 255.
        if item == 86:
            y[index] = np.array([238, 130, 238]) / 255.
        if item == 87:
            y[index] = np.array([220, 20, 60]) / 255.
        if item == 88:
            y[index] = np.array([46, 139, 87]) / 255.
        if item == 89:
            y[index] = np.array([25, 25, 112]) / 255.
        if item == 90:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 91:
            y[index] = np.array([127, 255, 212]) / 255.
        if item == 92:
            y[index] = np.array([218, 112, 214]) / 255.
        if item == 93:
            y[index] = np.array([216, 191, 216]) / 255.
        if item == 94:
            y[index] = np.array([221, 160, 221]) / 255.
        if item == 95:
            y[index] = np.array([75, 0, 130]) / 255.
        if item == 96:
            y[index] = np.array([65, 105, 225]) / 255.
        if item == 97:
            y[index] = np.array([70, 130, 180]) / 255.
        if item == 98:
            y[index] = np.array([230, 230, 250]) / 255.
        if item == 99:
            y[index] = np.array([240, 255, 255]) / 255.
        if item == 100:
            y[index] = np.array([64, 224, 208]) / 255.
        if item == 101:
            y[index] = np.array([240, 255, 240]) / 255.
        if item == 102:
            y[index] = np.array([34, 139, 34]) / 255.
        if item == 103:
            y[index] = np.array([124, 252, 0]) / 255.
        if item == 104:
            y[index] = np.array([255, 255, 240]) / 255.
        if item == 105:
            y[index] = np.array([240, 230, 140]) / 255.
        if item == 106:
            y[index] = np.array([245, 222, 179]) / 255.
        if item == 107:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 108:
            y[index] = np.array([255, 248, 220]) / 255.
        if item == 109:
            y[index] = np.array([255, 228, 181]) / 255.
        if item == 110:
            y[index] = np.array([255, 218, 185]) / 255.
        if item == 111:
            y[index] = np.array([210, 105, 30]) / 255.
        if item == 112:
            y[index] = np.array([160, 82, 45]) / 255.
        if item == 113:
            y[index] = np.array([255, 228, 225]) / 255.
        if item == 114:
            y[index] = np.array([250, 128, 114]) / 255.
        if item == 115:
            y[index] = np.array([255, 99, 71]) / 255.
        if item == 116:
            y[index] = np.array([255, 192, 203]) / 255.
        if item == 117:
            y[index] = np.array([165, 42, 42]) / 255.
        if item == 118:
            y[index] = np.array([178, 34, 34]) / 255.
        if item == 119:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 120:
            y[index] = np.array([64, 64, 64]) / 255.
        if item == 121:
            y[index] = np.array([64, 128, 192]) / 255.
        if item == 122:
            y[index] = np.array([192, 128, 64]) / 255.
        if item == 123:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 124:
            y[index] = np.array([0, 139, 0]) / 255.
        if item == 125:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 126:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 127:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 128:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 129:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 130:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 131:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 132:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 133:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 134:
            y[index] = np.array([255, 127, 80]) / 255.
        if item == 135:
            y[index] = np.array([127, 255, 0]) / 255.
        if item == 136:
            y[index] = np.array([238, 130, 238]) / 255.
        if item == 137:
            y[index] = np.array([220, 20, 60]) / 255.
        if item == 138:
            y[index] = np.array([46, 139, 87]) / 255.
        if item == 139:
            y[index] = np.array([25, 25, 112]) / 255.
        if item == 140:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 141:
            y[index] = np.array([127, 255, 212]) / 255.
        if item == 142:
            y[index] = np.array([218, 112, 214]) / 255.
        if item == 143:
            y[index] = np.array([216, 191, 216]) / 255.
        if item == 144:
            y[index] = np.array([221, 160, 221]) / 255.
        if item == 145:
            y[index] = np.array([75, 0, 130]) / 255.
        if item == 146:
            y[index] = np.array([65, 105, 225]) / 255.
        if item == 147:
            y[index] = np.array([70, 130, 180]) / 255.
        if item == 148:
            y[index] = np.array([230, 230, 250]) / 255.
        if item == 149:
            y[index] = np.array([240, 255, 255]) / 255.
        if item == 150:
            y[index] = np.array([64, 224, 208]) / 255.
        if item == 151:
            y[index] = np.array([240, 255, 240]) / 255.
        if item == 152:
            y[index] = np.array([34, 139, 34]) / 255.
        if item == 153:
            y[index] = np.array([124, 252, 0]) / 255.
        if item == 154:
            y[index] = np.array([255, 255, 240]) / 255.
        if item == 155:
            y[index] = np.array([240, 230, 140]) / 255.
        if item == 156:
            y[index] = np.array([245, 222, 179]) / 255.
        if item == 157:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 158:
            y[index] = np.array([255, 248, 220]) / 255.
        if item == 159:
            y[index] = np.array([255, 228, 181]) / 255.
        if item == 160:
            y[index] = np.array([255, 218, 185]) / 255.
        if item == 161:
            y[index] = np.array([210, 105, 30]) / 255.
        if item == 162:
            y[index] = np.array([160, 82, 45]) / 255.
        if item == 163:
            y[index] = np.array([255, 228, 225]) / 255.
        if item == 164:
            y[index] = np.array([250, 128, 114]) / 255.
        if item == 165:
            y[index] = np.array([255, 99, 71]) / 255.
        if item == 166:
            y[index] = np.array([255, 192, 203]) / 255.
        if item == 167:
            y[index] = np.array([165, 42, 42]) / 255.
        if item == 168:
            y[index] = np.array([178, 34, 34]) / 255.
        if item == 169:
            y[index] = np.array([192, 192, 192]) / 255.

        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y


def generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices, iter_index, day_str):  # 预测分类图
    pred_test = []
    with torch.no_grad():
        for X, y in all_iter:  # X.shape: torch.Size([16, 1, 9, 9, 200])  y没用
            X = X.to(device).float()
            net.eval()  # 评估模式, 这会关闭dropout
            y_hat = net(X)
            if isinstance(y_hat, list):
                pred_test.extend(np.array((y_hat[0] + y_hat[1]).cpu().argmax(axis=1)))
            else:
                if len(y_hat.shape)==1:
                    y_hat = y_hat.unsqueeze(0)
                pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))  # 每一行是一类的16个样本的概率值，取最大值对应的类别

        gt = gt_hsi.flatten()
        x_label = np.zeros(gt.shape)-1
        # for i in range(len(gt)):
        #     if gt[i] == 0:
        #         gt[i] = 17
        #         x_label[i] = 16

        x_label[total_indices] = pred_test
        x = np.ravel(x_label)
        y_list = list_to_colormap(x)
        y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

        classification_map(y_re, gt_hsi, 300, './classification_map/' + day_str + '_' + Dataset + '_' + net.name + '_' +
                           str(iter_index) + '.png')
        if iter_index == 0:
            gt = gt[:] - 1
            y_gt = list_to_colormap(gt)
            gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
            classification_map(gt_re, gt_hsi, 300,
                               './classification_map/' + Dataset + '_gt.png')
    print('------Get classification maps successful-------')


def PredictMap_png(PredictMap_iter, net, gt_hsi, Dataset, device, total_indices, iter_index, day_str):  # 预测分类图
    pred_test = []
    for X, y in PredictMap_iter:  # X.shape: torch.Size([16, 1, 9, 9, 200])  y没用
        # X = torch.tensor(X)
        X = X.to(device)
        net.eval()  # 评估模式, 这会关闭dropout
        pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))  # 每一行是一类的16个样本的概率值，取最大值对应的类别

    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)-1
    # for i in range(len(gt)):
    #     if gt[i] == 0:
    #         gt[i] = 17
    #         x_label[i] = 16

    x_label[total_indices] = pred_test
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    # path = '../DBDA/'
    classification_map(y_re, gt_hsi, 300, './classification_map/' + day_str + '_' + Dataset + '_' + net.name +
                       str(iter_index) + '_all.png')
    print('------Get classification full maps successful-------')


""" ================================================定义分类精度记录函数===record.py===================================="""


def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae, con_mat, path):
    f = open(path, 'a')

    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)
    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)
    sentence6 = 'Total average Training time is: ' + str(np.sum(training_time_ae)/len(training_time_ae)) + '\n'
    f.write(sentence6)
    sentence7 = 'Total average Testing time is: ' + str(np.sum(testing_time_ae)/len(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)

    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements in confusion matrix: " + '\n' + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements in confusion matrix: " + '\n' + str(element_std) + '\n' + '\n'
    f.write(sentence9)
    sentence10 = "The whole confusion matrix of all iteration: " + '\n' + str(con_mat) + '\n' + '\n'
    f.write(sentence10)
    f.close()


""" ================================================样本平衡参数计算==================================================="""


def compute_imf_weights(train_indices, ground_truth, n_classes=None):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
    Returns:
        numpy array with the IMF coefficients
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)
    train_gt = ground_truth[train_indices]-1

    for c in range(0, n_classes):
        frequencies[c] = np.count_nonzero(train_gt == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights


""" ================================================ArcFace损失函数==================================================="""


class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        logits = logits.clamp(-1, 1)
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        loss = torch.nn.CrossEntropyLoss(logits, labels)
        return loss


""" ================================================重构Dateset==================================================="""


class HSI_Dataset(Data.Dataset):
    def __init__(self, indices, padded_data, gt, patch_length, SHAPE):
        self.indices = indices
        self.padded_data = padded_data
        self.patch_length = patch_length
        self.y_gt = gt-1
        self.SHAPE = SHAPE
        self.data_assign = index_assignment(indices, padded_data.shape[0]-2*patch_length,
                                            padded_data.shape[1]-2*patch_length, patch_length)

    def __getitem__(self, idx):  # 默认是item，但常改为idx，是index的缩写
        x = select_patch(self.padded_data, self.data_assign[idx][0], self.data_assign[idx][1], self.patch_length)

        y = self.y_gt[self.indices[idx]]
        if self.SHAPE == 1:
            x = x.transpose(2, 0, 1)
            x = np.expand_dims(x, axis=0)
        else:
            x = np.expand_dims(x, axis=0)
        return x, y

    def __len__(self):
        return len(self.data_assign)


def generate_iter_Dateset(train_indices, test_indices, val_indices, whole_data, PATCH_LENGTH, padded_data,
                          INPUT_DIMENSION, batch_size, gt, SHAPE = 0):
    total_indices = np.concatenate((train_indices, val_indices, test_indices))
    TRAIN_SIZE = len(train_indices)
    VAL_SIZE = len(val_indices)

    y_train = gt[train_indices] - 1
    y_val = gt[val_indices] - 1
    train_data = select_small_cubic(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, padded_data,
                                    INPUT_DIMENSION)  # small_cubic_data.shape: (307, 9, 9, 200)
    val_data = select_small_cubic(VAL_SIZE, val_indices, whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data  # x_train.shape: (307, 9, 9, 200)
    x_val = val_data
    if SHAPE == 1:
        x1_tensor_train = torch.from_numpy(x_train).permute(0, 3, 1, 2).type(torch.FloatTensor).unsqueeze(1)  # (307, 1, 200, 9, 9)
        x1_tensor_valida = torch.from_numpy(x_val).permute(0, 3, 1, 2).type(torch.FloatTensor).unsqueeze(1)  # (307, 1, 200, 9, 9)
    elif SHAPE == 2:
        x_train = gain_neighborhood_band(x_train, 3, PATCH_LENGTH*2+1)
        x_val = gain_neighborhood_band(x_val, 3, PATCH_LENGTH * 2 + 1)
        x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor)
        x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor)
    else:
        x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)  # (307, 1, 9, 9, 200)
        x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)  # (307, 1, 9, 9, 200)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)  # np.array(y1_tensor_train).shape: (307,)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)  # np.array(y1_tensor_valida).shape: (307,)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)  # np.array(torch_dataset_train).shape: (307, 2)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida,y1_tensor_valida)  # np.array(torch_dataset_valida).shape: (307, 2)
    del x1_tensor_train
    gc.collect()


    test_data = HSI_Dataset(test_indices, padded_data, gt, PATCH_LENGTH, SHAPE)  # small_cubic_data.shape: (9942, 9, 9, 200)
    all_data = HSI_Dataset(total_indices, padded_data, gt, PATCH_LENGTH, SHAPE)

    train_iter = Data.DataLoader(dataset=torch_dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)  # torch TensorDataset format  # mini batch size  # 要不要打乱数据 (打乱比较好)  # 多线程来读数据  #  X.shape: torch.Size([16, 1, 9, 9, 200]), y.shape: torch.Size([16])
    valiada_iter = Data.DataLoader(dataset=torch_dataset_valida, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    all_iter = Data.DataLoader(dataset=all_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_iter, valiada_iter, test_iter, all_iter
