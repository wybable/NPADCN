import time
import torch
import numpy as np
from icecream import ic
from IPython import display
from matplotlib import pyplot as plt
from icecream import ic


def evaluate_accuracy(data_iter, net, loss, device):
    """计算并返回测试集的精度值和损失值"""
    acc_sum, test_l_sum = 0.0, 0
    n = data_iter.dataset.__len__()
    net.eval()  # 评估模式, 关闭参数的反向传播和dropout等
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device).float()
            y = y.to(device)
            y_hat = net(X)
            if isinstance(y_hat, list):
                loss_list = [loss(o, y.long()) / len(y_hat) for o in y_hat]
                l = sum(loss_list)
                acc_sum += ((y_hat[0] + y_hat[1]).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            else:
                l = loss(y_hat, y.long())
                acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            test_l_sum += l.item() * y.shape[0]
    net.train()  # 改回训练模式
    return [acc_sum / n, test_l_sum / n]


def train(net, train_iter, valida_iter, loss, optimizer, device,  index_iter, day_str, epochs=30, early_stopping=True,
          early_num=30, datasetname='IN'):
    """训练过程"""
    loss_list = [100]
    early_epoch = 0
    net = net.to(device)
    print("training on ", device)
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    n = train_iter.dataset.__len__()
    # 学习率衰减方式设置为余弦退火
    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=2e-6, last_epoch=-1)
    PATH = './net/temp_' + net.name + '_' + datasetname + '.pt'  # 用于临时存储当前验证集loss最低的net
    times = 0
    for epoch in range(epochs):
        train_acc_sum, train_l_sum = 0.0, 0
        time_epoch_start = time.time()
        # lr_adjust = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
        for X, y in train_iter:
            X = X.to(device).float()
            y = y.to(device)
            # ic(X.shape, y.shape)  # X.shape: torch.Size([16, 200, 9, 9]), y.shape: torch.Size([16])
            y_hat = net(X)
            if isinstance(y_hat, list):
                l_list = [loss(o, y.long()) / len(y_hat) for o in y_hat]
                l = sum(l_list)
            else:
                l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item() * y.shape[0]
            if isinstance(y_hat, list):
                train_acc_sum += ((y_hat[0]+y_hat[1]).argmax(dim=1) == y).sum().item()
            else:
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            torch.cuda.empty_cache()


        lr_adjust.step(epoch)
        # print(optimizer.param_groups[-1]['lr'])
        valida_acc, valida_loss = evaluate_accuracy(valida_iter, net, loss, device)
        loss_list.append(valida_loss)
        time_epoch_end = time.time()
        time_epoch = time_epoch_end - time_epoch_start
        times += time_epoch

        print('epoch %d, train loss %.6f, train acc %.4f, valida loss %.6f, valida acc %.4f, time %.4f sec' % (epoch + 1, train_l_sum / n, train_acc_sum / n, valida_loss, valida_acc, time_epoch))

        # 绘图部分
        train_loss_list.append(train_l_sum / n)  # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)

        # 早停策略
        if early_stopping and loss_list[-2] < loss_list[-1]:  # < 0.05) and (loss_list[-1] <= 0.05):
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                break
        else:
            early_epoch = 0
            if valida_acc > 0.5:
                torch.save(net.state_dict(), PATH)
    print('epoch %d, loss %.4f, train acc %.4f, time %.4f sec' % (
        epoch + 1, train_l_sum / n, train_acc_sum / n, times))

    net.load_state_dict(torch.load(PATH))
    # set_figsize()
    plt.rcParams['figure.figsize'] = (8, 8.5)
    plt.figure()
    train_accuracy = plt.subplot(221)
    train_accuracy.set_title('train_accuracy')
    plt.plot(np.linspace(1, epoch, len(train_acc_list)), train_acc_list, color='green')
    plt.xlabel('epoch')
    plt.ylabel('train_accuracy')
    # train_acc_plot = np.array(train_acc_plot)
    # for x, y in zip(num_epochs, train_acc_plot):
    #    plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=11)

    test_accuracy = plt.subplot(222)
    test_accuracy.set_title('valida_accuracy')
    plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue')
    plt.xlabel('epoch')
    plt.ylabel('test_accuracy')
    # test_acc_plot = np.array(test_acc_plot)
    # for x, y in zip(num_epochs, test_acc_plot):
    #   plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=11)

    loss_sum = plt.subplot(223)
    loss_sum.set_title('train_loss')
    plt.plot(np.linspace(1, epoch, len(train_loss_list)), train_loss_list, color='red')
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    # ls_plot = np.array(ls_plot)

    test_loss = plt.subplot(224)
    test_loss.set_title('valida_loss')
    plt.plot(np.linspace(1, epoch, len(valida_loss_list)), valida_loss_list, color='gold')
    plt.xlabel('epoch')
    plt.ylabel('valida loss')
    # ls_plot = np.array(ls_plot)

    # plt.show()
    plt.savefig('./record/' + day_str + '_' + net.name + '_' + datasetname + '_' + str(index_iter) + '_Tra_val_curve_Epoch_%d.png' % (epoch + 1))

