from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import torch
from GetModel import TextCNN
from GetPerformance import performance


import sys


dtype = torch.FloatTensor
device = torch.device("cpu")
soft = nn.Softmax(dim=0)

#训练深度学习的模型
def dl_train_model(args,i,Train_data,Valida_data,getplt):

    mRNA = args.mRNA[0]
    length = args.len[0]
    epoch = args.epoch[0]
    batch_size = args.batchsize[0]
    lr = args.lr[0]

    color_list = ['r', 'g', 'b', 'c', 'y']
    '''
    1.这里用于处理数据集不平衡的问题
    '''
    Train_seq = []
    Train_ml = []
    Train_label = []
    for data in Train_data:
        Train_seq.append(data.seq)
        Train_ml.append(data.mlfeature)
        Train_label.append(int(data.label))


    input_Train_batch,input_ml_Train_batch, target_Train_batch = \
        torch.tensor(Train_seq, dtype=torch.float32).long(), torch.tensor(Train_ml, dtype=torch.float32), torch.LongTensor(Train_label)
    Train_dataset = Data.TensorDataset(input_Train_batch,input_ml_Train_batch, target_Train_batch)
    train_loader = torch.utils.data.DataLoader(
        Train_dataset,
        batch_size=batch_size,
    )


    Valida_seq = []
    Valida_ml = []
    Valida_label = []
    for data in Valida_data:
        Valida_seq.append(data.seq)
        Valida_ml.append(data.mlfeature)
        Valida_label.append(int(data.label))

    valida_batch, valida_ml_batch, valida_label_batch = \
        torch.tensor(Valida_seq, dtype=torch.float32).long(),torch.tensor(Valida_ml, dtype=torch.float32), torch.LongTensor(Valida_label)
    Valida_dataset = Data.TensorDataset(valida_batch, valida_ml_batch ,valida_label_batch)
    valida_loader = torch.utils.data.DataLoader(
        Valida_dataset,
        batch_size=batch_size,
    )


    '''
    2.这里用于通过训练数据集来进行模型的训练
    model = svm_train(train_label_list, train_vector_list, svm_params)
    '''
    model = TextCNN(args).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # print('\n')
    # sys.stdout.flush()
    # exit()

    loss_plane_list = []
    epoch_plane_list = []
    print("Start Train:")

    for epoch_num in range(epoch):
        model = model.train()

        times_of_epoch = 0
        loss_sum = 0

        for batch_x, batch_ml, batch_y in train_loader:

            times_of_epoch = times_of_epoch + 1
            batch_x, batch_ml, batch_y = batch_x.to(device), batch_ml.to(device),batch_y.to(device)

            pred = model(batch_x,batch_ml)
            loss = criterion(pred, batch_y)
            loss_sum = loss_sum + loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_average = float(loss_sum/times_of_epoch)
        loss_plane_list.append(loss_average)
        epoch_plane_list.append(epoch_num)

        print('Epoch:', '%04d' % (epoch_num+1), 'loss =', '{:.8f}'.format(loss_average))
        sys.stdout.flush()

        '''
        3.这里进行对测试集进行验证
        '''
        model = model.eval()
        times_of_valida_epoch = 0
        valida_loss_sum = 0

        with torch.no_grad():
            for valida_x, valida_ml,valida_y in valida_loader:
                times_of_valida_epoch = times_of_valida_epoch+1
                valida_x = valida_x.cuda()
                valida_ml = valida_ml.cuda()
                valida_y = valida_y.cuda()

                valida_pred = model(valida_x,valida_ml)
                valida_loss = criterion(valida_pred,valida_y)
                valida_loss_sum = valida_loss_sum + valida_loss.item()

        valida_average_loss = float(valida_loss_sum/times_of_valida_epoch)
        scheduler.step(valida_average_loss)

    getplt.Draw(list_x=epoch_plane_list, list_y=loss_plane_list, filename='./plt/' + str(mRNA) + str(length) +'.png', color=color_list[i],label=mRNA)

    return model



def dl_performance(dl_model,Test_data,batch_size):

    test_vector_list = []
    test_ml_list = []
    test_label_list = []

    for data in Test_data:
        test_vector_list.append(data.seq)
        test_ml_list.append(data.mlfeature)
        test_label_list.append(int(data.label))

    '''
    3.这里根据训练好的model对测试集进行测试t
    '''
    model = dl_model.eval()
    test_batch, test_ml, test_label_batch = \
        torch.tensor(test_vector_list, dtype=torch.float32).long(),torch.tensor(test_ml_list, dtype=torch.float32), torch.LongTensor(test_label_list)

    test_dataset = Data.TensorDataset(test_batch, test_ml ,test_label_batch)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
    )

    test_val = []
    test_predict_label = []
    test_origin_label = []
    with torch.no_grad():
        for test_x, test_ml, test_y in test_loader:
            test_x = test_x.cuda()
            test_ml = test_ml.cuda()
            test_y = test_y.cuda()
            result = model(test_x,test_ml)

            test_val_batch = []
            test_label_batch = []

            for data in result.data:
                probability = soft(data)
                probability = probability.cpu().numpy()
                test_val_batch.append(probability[1])

            for i in result.data.max(1, keepdim=True)[1]:
                if i[0] == 0:
                    test_label_batch.append(0)
                elif i[0] == 1:
                    test_label_batch.append(1)

            test_val = test_val + test_val_batch
            test_predict_label = test_predict_label + test_label_batch
            test_origin_label = test_origin_label + test_y.tolist()

    # 五重交叉验证的结果存储
    AUC,ACC,MCC,PR = performance(test_origin_label, test_predict_label, test_val)

    return ACC