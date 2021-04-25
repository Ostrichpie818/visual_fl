import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import time
import threading
from PIL import Image
import random


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=5, help='number of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=5, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=10, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')

class Server:
    def __init__(self,num_of_clients,epoch,batch_size,num_of_com,is_iid,alChoice):
        self.accuList=[]
        self.lossList=[]
        self.total_ll=[]
        self.thisList=[]
        self.threads = []
        self.lock = threading.Lock()
        self.cl_List=[]
        self.alChoicelist=['FedAvg']
        self.paraList=[]

        #初始化重要参数
        if(num_of_clients):
            self.num_of_clients=num_of_clients
        else:
            self.num_of_clients=2
        
        if(epoch):
            self.epoch=epoch
        else:
            self.epoch=2
        
        if(batch_size):
            self.batch_size=batch_size
        else:
            self.batch_size=2
            
        if(num_of_com):
            self.num_of_com=num_of_com
        else:
            self.num_of_com=2

        if (is_iid=='Non_IID(非独立同分布)'):
            self.is_iid = 1
        else:
            self.is_iid = 0
        for i in range(len(self.alChoicelist)):
            if(alChoice==self.alChoicelist[i]):
                self.alChoice=i
                break

    def test_mkdir(self,path):
        if not os.path.isdir(path):
            os.mkdir(path)

    #实时更新训练loss曲线
    def showList(self,lossList,clientList):
        st.write("## 训练收敛状况")
        chart_data = pd.DataFrame([lossList[0]],columns=clientList)
        chart = st.line_chart(chart_data)
        progress_bar = st.progress(0)
        progress_bar.progress(0)

        for i in range(0,len(lossList)):
            chart_data=pd.DataFrame([lossList[i]],columns=clientList)
            chart.add_rows(chart_data)

            progress_bar.progress(i/(len(lossList)-1))


    def train(self,num_of_clients,num_in_comm,net,loss_func,opti,global_parameters,myClients,index):
        if(len(self.cl_List)<num_of_clients):
            self.cl_List.append('Client:{}'.format(index))
        clients_in_comm = ['client{}'.format(index)]

        sum_parameters = None
        for client in tqdm(clients_in_comm):

            self.lock.acquire()
            
            if (self.alChoice==0):
                local_parameters = myClients.clients_set[client].localUpdate(self.epoch, self.batch_size, net,
                                                                         loss_func, opti, global_parameters,
                                                                         self.lossList[index])


            #这里必须空一行
            for i in range(len(self.lossList[index])):
                if (len(self.total_ll) < len(self.lossList[index])):
                    self.total_ll.append([])
                if (len(self.thisList) < len(self.lossList[index])):
                    self.thisList.append([])
                self.total_ll[i].append(self.lossList[index][i])
                self.thisList[i].append(self.lossList[index][i])
            self.lock.release()

            if(alChoice==self.alChoicelist[0]):
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        sum_parameters[key] = var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] + local_parameters[var]

                for var in global_parameters:
                    global_parameters[var] = (sum_parameters[var] / num_in_comm)



    def main(self):
        args = parser.parse_args()
        args = args.__dict__

        self.test_mkdir(args['save_path'])

        os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        net = None
        if args['model_name'] == 'mnist_2nn':
            net = Mnist_2NN()
        elif args['model_name'] == 'mnist_cnn':
            net = Mnist_CNN()

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            net = torch.nn.DataParallel(net)
        net = net.to(dev)

        loss_func = F.cross_entropy
        opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

        myClients = ClientsGroup('mnist', self.is_iid, self.num_of_clients, dev)
        testDataLoader = myClients.test_data_loader
        self.test_data = myClients.test_data
        self.test_label = myClients.test_label
        self.test_data_size=myClients.test_data_size

        num_in_comm = int(max(self.num_of_clients * args['cfraction'], 1))

        self.paraList.extend([args['model_name'],self.num_of_com, self.epoch, self.batch_size,
                             args['learning_rate'],self.num_of_clients,args['cfraction']])
        paraTable = pd.DataFrame([self.paraList],
                                columns=['Model Name', 'Num of Com',
                                         'Epoch', 'Batch_size', 'Learning Rate',
                                         'Num of Clients', 'C fraction'])
        st.write("""## 当前训练参数选择""")
        st.table(paraTable)

        global_parameters = {}
        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()

        #开始训练
        for i in range(self.num_of_com):
            self.threads=[]
            for j in range(self.num_of_clients):
                self.threads.append(threading.Thread(target=self.train,
                                                     kwargs={"num_of_clients":num_of_clients,"num_in_comm": num_in_comm, "net": net,
                                                             "loss_func": loss_func, "opti": opti,
                                                             "global_parameters": global_parameters,
                                                             "myClients": myClients, "index": j}))
                st.report_thread.add_report_ctx(self.threads[j])
                self.lossList.append([])

            for t in self.threads:
                t.start()

            for t in self.threads:
                t.join()

            self.showList(self.thisList,self.cl_List)
            self.thisList=[]
            self.lossList=[]
            #开始测试
            with torch.no_grad():
                print((i + 1) % args['val_freq'])
                if (i + 1) % args['val_freq'] == 0:
                    net.load_state_dict(global_parameters, strict=True)
                    sum_accu = 0
                    num = 0
                    for data, label in testDataLoader:
                        data, label = data.to(dev), label.to(dev)
                        preds = net(data)
                        preds = torch.argmax(preds, dim=1)
                        sum_accu += (preds == label).float().mean()
                        num += 1
                    accuracy=sum_accu/num
                    self.accuList.append(format(accuracy))

            if (i + 1) % args['save_freq'] == 0:
                torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, self.epoch,
                                                                                                self.batch_size,
                                                                                                args['learning_rate'],
                                                                                                self.num_of_clients,
                                                                                                args['cfraction'])))

            
if __name__=='__main__':
    file_dir = "./data/his_data"#读出数据
    testImg_dir = "./img/test_image"#读测试集图像
    files = os.listdir(file_dir)
    imgs = os.listdir(testImg_dir)
    files.sort(key= lambda x: int(x[:-4]))
    imgs.sort(key=lambda x: int(x[:-4]))
    image_1=Image.open("./img/machine-learning.png")
    image_2=Image.open("./img/machine-learning-mini.png")

    option = []
    for i in range(len(files)):
        option.append('{}. '.format(i+1)+files[i][0:4]+'年'+files[i][4:6]+'月'+files[i][6:8]+'日'+files[i][8:10]+' : '+files[i][10:12]+'   训练结果')

    #st.sidebar.image(image_2)
    st.sidebar.markdown('# 选择模式')
    choice_1=st.sidebar.radio("选择页面",['主页','开始训练','复现'])

    st.sidebar.markdown('## 新训练参数设置')

    num_of_clients = st.sidebar.number_input("请输入边缘结点数量",1)
    num_of_com = st.sidebar.number_input("请输入训练轮数", 1)
    epoch = 2 #st.sidebar.slider("Epoch", min_value=0, max_value=20)
    alChoice = st.sidebar.selectbox("聚合算法选择",['FedAvg'], key="1")
    is_iid = st.sidebar.selectbox("数据划分方式", ['IID(独立同分布)','Non_IID(非独立同分布)'], key="1")
    if(is_iid=='IID(独立同分布)'):
        batch_size = st.sidebar.slider("IID时：batchsize设置", min_value=0, max_value=100)
    else:
        batch_size= 0
        st.sidebar.info("数据集默认乱序分配")

    st.sidebar.markdown('## 过往训练展示形式')
    hisForm = st.sidebar.multiselect("请选择展示形式", ['参数表格','训练图像'], key="2")

    if(choice_1=='主页'):
        #界面文字和选项的显示
        img,title=st.beta_columns([1,8])
        img.image(image_1)
        title.write("""# 联邦学习可视化平台""")
        st.write("""*题目：联邦学习下边缘设备训练实时监控系统设计与实现*""")
        st.write("""*Designed By:  CYD_DLUT*""")
        st.info("""**请打开侧边栏选择展示内容!**  
            您可以选择开始新一轮的联邦学习训练，或者查看之前的训练结果""")

    elif(choice_1=='开始训练'):
        st.write("""                  

                """)
        st.info("""***训练进行中......***""")
        st.warning("训练结束后在当前界面直接改变参数，会开始新一轮的训练")

        startTime = time.localtime()
        sv = Server(num_of_clients, epoch, batch_size, num_of_com,is_iid,alChoice)
        sv.main()

        #测试集部分展示
        st.write("## 部分测试集展示")
        fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, )
        ax = ax.flatten()
        start = random.randint(0, sv.test_data_size - 10)
        for i in range(start, start + 10):
            img = sv.test_data[i].reshape(28, 28)
            ax[i - start].imshow(img, cmap='Greys', interpolation='nearest')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.savefig('./img/test_image/{}.png'.format(time.strftime("%Y%m%d%H%M",startTime)))#保存图像
        st.pyplot(fig)


        # 精确度图像
        st.write("## 精确度图像")
        Acculine = pd.DataFrame(sv.accuList, columns=['Accuracy'])
        st.line_chart(Acculine)

        # 保存训练数据
        path = './data/his_data/{}.npy'.format(time.strftime("%Y%m%d%H%M", startTime))
        dataList = sv.paraList +[-1] + sv.total_ll + [-1] + sv.accuList + [sv.num_of_clients]
        dl = np.array(dataList)
        np.save(path,dl)
        st.success('''***保存路径：{}***  
                ***保存成功!***'''.format(path))
    else:
        if(len(hisForm)==0):
            st.sidebar.error("请选择展示形式")
            st.error("请下拉侧边栏选择展示形式")
        hisDatalist=[]

        for i in range(len(option)):
            dpath = './data/his_data/' + files[i]
            his_dl = np.load(dpath, allow_pickle=True)
            his_dl = his_dl.tolist()
            hisDatalist.append(his_dl)

        if len(hisDatalist)==0:
            st.warning("没有可以显示的数据")

        else:
            for i in range(len(hisForm)):
                if (hisForm[i] == '参数表格'):
                    index = 0
                    his_ltable = []
                    st.write('''## 参数列表''')
                    for j in range(len(option)):
                        #st.write([hisDatalist[j][index:7]])
                        list=[]
                        list.append('{}'.format(option[j][8:21]))
                        list.extend(hisDatalist[j][index:7])
                        his_ltable.append(list)

                    his_para = pd.DataFrame(his_ltable,
                                        columns=['时间','模型名称', 'Num of Com',
                                                 'Epoch', 'Batch_size', 'Learning Rate',
                                                 'Num of Clients', 'C fraction'])
                    st.table(his_para)

                if (hisForm[i] == '训练图像'):
                    selector = st.selectbox("请选择需要展示的图像数据", option, key="1")
                    episode = ''
                    for i in range(len(selector)):
                        episode = episode+selector[i]
                        if(selector[i+1]=='.'):
                            break
                    episode = int(episode)-1#选中第episode个历史数据

                    fidForm = st.multiselect("请选择图像类别", ['测试集部分展示', '边缘节点训练图像','精确度图像'], key="3")
                    for j in range(len(fidForm)):
                        if fidForm[j]=='测试集部分展示':
                            st.write('## 测试集部分展示')
                            st.image('./img/test_image/{}'.format(imgs[episode]))

                        elif fidForm[j]=='边缘节点训练图像':
                            st.write('''## 边缘节点训练图像''')
                            colList = []
                            for i in range(hisDatalist[episode][len(hisDatalist[episode]) - 1]):
                                colList.append('Client: {}'.format(i))
                            numOfclients = 0
                            for i in range(hisDatalist[episode][1]):
                                his_lline = []
                                index = 8
                                while True:
                                    thisLine = []
                                    for j in range(numOfclients, numOfclients + hisDatalist[episode][5]):
                                        thisLine.append(hisDatalist[episode][index][j])
                                    his_lline.append(thisLine)
                                    index = index + 1
                                    if (hisDatalist[episode][index] == -1):
                                        break
                                numOfclients = numOfclients + hisDatalist[episode][5]
                                st.line_chart(pd.DataFrame(his_lline, columns=colList))

                        elif fidForm[j]=='精确度图像':
                            st.write('''## 精确度图像''')
                            accuLocation=len(hisDatalist[episode])-1-hisDatalist[episode][1]
                            his_aline = pd.DataFrame(hisDatalist[episode][accuLocation: len(hisDatalist[episode]) - 1],
                                                     columns=['Accuracy'])
                            st.line_chart(his_aline)
