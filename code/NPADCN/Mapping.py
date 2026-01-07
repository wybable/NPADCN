import numpy as np
from icecream import ic
import time
from datetime import datetime
import torch
from sklearn import metrics, preprocessing
from HsiConfermer_residual import Con_Trans
from RasterRead import IMAGE
import matplotlib.pyplot as plt
import network
from SpectralFormer import SpectralFormer
from torchsummary import summary
from HsiConfermer_residual import Con_Trans
from A2S2KResNet import A2S2KResNet
from BS2Tnetwork import BS2T
from Hy_VIT import ViT
from Deform_net import Deform_net
import scipy.io as sio
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'



'''===============================================定义主程序需要用到的函数============================================='''


def select_patch(matrix, pos_row, pos_col, ex_len):  # 根据行列号提取图像块
    selected_patch = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)][:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    # selected_patch.shape: (9, 9, 200)
    return selected_patch


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


def classification_map(map, x, y, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(y * 2.0 / dpi, x * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


'''====================================================图像数据读取==================================================='''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or CPU
day = datetime.now()
day_str = day.strftime('%m_%d_%H_%M')  # 当前具体时间，用于实验记录结果的命
# Dataset = 'GQ'  # IN, UP, BS, SV, PC, KSC, HLS or AXGF
# Dataset = Dataset.upper()
start = time.time()  # 开始时间
drv = IMAGE()

# mat_data = sio.loadmat('../datasets/Indian_pines_corrected.mat')
# data_hsi = mat_data['indian_pines_corrected']
# data_proj, data_geotrans, data_hsi = drv.read_img(r'../datasets/MHJYX.dat')
# data_proj, data_geotrans, data_hsi = drv.read_img(r'../datasets/HLS.dat')
data_proj, data_geotrans, data_hsi = drv.read_img(r'../datasets/GaoQiao.dat')
# data_proj, data_geotrans, data_hsi = drv.read_img(r'../datasets/WHU-Hi-LongKou.tif')
data_hsi = np.rollaxis(data_hsi, 0, 3)
image_x, image_y, bands = data_hsi.shape  # 记录原始HSI的图像
end = time.time()
print('Read data consuming time: %s Seconds' % (end - start))

'''========================================================设置模型参数==============================================='''
patch_length = 4  # 图像块扩充尺寸
batch_size = 320
CLASSES_NUM = 12
PATH = './net/03_17_12_20_Deform_net_GQ_2_0.953.pt'
BAND = bands
# net = network.DBDA_network_MISH(BAND, CLASSES_NUM)
# net = Con_Trans(BAND, patch_len=patch_length, num_classes=CLASSES_NUM, cnn_channel=96, embed_dim=16, depth=4,
#                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                 drop_path_rate=0)
net = Deform_net(BAND, num_classes=CLASSES_NUM, cnn_channel=9,)
# net = A2S2KResNet(BAND, CLASSES_NUM, 2)
# net = network.DBMA_network(BAND, CLASSES_NUM)
# net = network.CDCNN_network(BAND, CLASSES_NUM)
# net = network.FDSSC_network(BAND, CLASSES_NUM)
# net = network.SSRN_network(BAND, CLASSES_NUM)
# net = SSFTTnet(in_channels=1, band=BAND, num_classes=CLASSES_NUM, num_tokens=4, dim=64, depth=1, heads=8,
#                mlp_dim=8, dropout=0.1, emb_dropout=0.1)
# net = SpectralFormer(image_size=PATCH_LENGTH*2+1, near_band=3, num_patches=BAND, num_classes=CLASSES_NUM, dim=64,
#                      depth=5, heads=4, mlp_dim=16, dropout=0, emb_dropout=0, mode='CAF')
# net = BS2T(BAND, CLASSES_NUM,)
# net = ViT(bands=BAND, image_size=patch_length*2+1, patch_size=1, num_classes=16, dim=128, depth=6, heads=16,
#           mlp_dim=256,channels=BAND, dropout=0.1, emb_dropout=0.1)
net.load_state_dict(torch.load(PATH))
net.to(device)
net.eval()


'''===============================图像和标签格式转化降,维逐波段归一化，reshape成三维数组=================================='''
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))  # 将原始HSI数据行和列展开到一维
# np.prod 连乘操作，所有元素相乘  data_hsi.shape[:2]: (145, 145) data_hsi.shape[2:]: (200,)
# data = preprocessing.scale(data)  # 对每个波段进行归一化处理
scaler = preprocessing.StandardScaler()
data = scaler.fit_transform(data)
whole_data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
# 得到与原始HSI格式相同（三维）的归一化后的数据
padded_data = np.lib.pad(whole_data, ((patch_length, patch_length), (patch_length, patch_length), (0, 0)),
                         'constant', constant_values=0)  # 三个维度，对应维度要填充的行数、列数、通道数等。实际应用中可以镜像复制边界





'''===========================================利用训练好的模型进行分类制图==========================================='''
pred_test = torch.tensor([]).to(device)
patch_data = []
batch_num = 0

tic1 = time.time()
times = 0
print('--------------------Start---------------------')
for i in range(image_x):
    for j in range(image_y):
        patch_data.append(select_patch(padded_data, i+patch_length, j+patch_length, patch_length))
        batch_num += 1
        if batch_num % batch_size == 0 or (i == image_x-1 and j == image_y-1):
            patch_data = np.array(patch_data)
            if net.name == "Con_Trans" or net.name == "SSFTT" or net.name == 'Vit':
                batch_data = torch.from_numpy(patch_data).permute(0, 3, 1, 2).type(torch.FloatTensor).unsqueeze(1)
            else:
                batch_data = torch.from_numpy(patch_data).type(torch.FloatTensor).unsqueeze(1)
            tic = time.time()
            batch_data = batch_data.to(device)
            y_hat = net(batch_data)
            # if isinstance(y_hat, list):
            #     pred_test.extend(np.array((y_hat[0]+y_hat[1]).cpu().argmax(axis=1)))
            # else:
            pred_test = torch.cat([pred_test, y_hat.argmax(axis=1)])
            toc = time.time()
            times += toc-tic
            batch_num = 0
            patch_data = []
        if j == image_y-1:
            toc1 = time.time()
            print("Row{} finished, cost {} s, {} min, {} h".format(i+1, toc1-tic1, (toc1-tic1)/60, (toc1-tic1)/3600))


assert len(pred_test) == image_x * image_y
pred_test = np.array(pred_test.cpu())
im_label = np.reshape(pred_test, (image_x, image_y))
for i in range(image_x):
    for j in range(image_y):
        if data_hsi[i, j].all() == 0:
            im_label[i, j] = -1
pred_test = np.reshape(im_label, (image_x * image_y))
drv.write_img('./full_classification_maps/' + PATH.split("/")[2][:-3] + "__" + str(times) + '__fullcls.dat',
              data_proj, data_geotrans, im_label, im_type="ENVI")
y_list = list_to_colormap(pred_test)
y_re = np.reshape(y_list, (image_x, image_y, 3))
y_save = y_re.transpose((2, 0, 1))
drv.write_img('./full_classification_maps/' + PATH.split("/")[2][:-3] + "__" + str(times) + '__fullrgb.dat',
              data_proj, data_geotrans, y_save, im_type="ENVI")
classification_map(y_re, image_x, image_y, 300, './full_classification_maps/' + PATH.split("/")[2][:-3] + "__"
                   + str(times) + '__fullpng.png')
toc1 = time.time()
print('------Get classification full maps successful, cost {} h, model prediction cost {}s -------'.format(
     (toc1-tic1)/3600, times))



























