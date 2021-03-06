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

        #?????????????????????
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

        if (is_iid=='Non_IID(??????????????????)'):
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

    #??????????????????loss??????
    def showList(self,lossList,clientList):
        st.write("## ??????????????????")
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


            #?????????????????????
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
        st.write("""## ????????????????????????""")
        st.table(paraTable)

        global_parameters = {}
        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()

        #????????????
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
            #????????????
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
    file_dir = "./data/his_data"#????????????
    testImg_dir = "./img/test_image"#??????????????????
    files = os.listdir(file_dir)
    imgs = os.listdir(testImg_dir)
    files.sort(key= lambda x: int(x[:-4]))
    imgs.sort(key=lambda x: int(x[:-4]))
    image_1=Image.open("./img/machine-learning.png")
    image_2=Image.open("./img/machine-learning-mini.png")

    option = []
    for i in range(len(files)):
        option.append('{}. '.format(i+1)+files[i][0:4]+'???'+files[i][4:6]+'???'+files[i][6:8]+'???'+files[i][8:10]+' : '+files[i][10:12]+'   ????????????')

    #st.sidebar.image(image_2)
    st.sidebar.markdown('# ????????????')
    choice_1=st.sidebar.radio("????????????",['??????','????????????','??????'])

    st.sidebar.markdown('## ?????????????????????')

    num_of_clients = st.sidebar.number_input("???????????????????????????",1)
    num_of_com = st.sidebar.number_input("?????????????????????", 1)
    epoch = 2 #st.sidebar.slider("Epoch", min_value=0, max_value=20)
    alChoice = st.sidebar.selectbox("??????????????????",['FedAvg'], key="1")
    is_iid = st.sidebar.selectbox("??????????????????", ['IID(???????????????)','Non_IID(??????????????????)'], key="1")
    if(is_iid=='IID(???????????????)'):
        batch_size = st.sidebar.slider("IID??????batchsize??????", min_value=0, max_value=100)
    else:
        batch_size= 0
        st.sidebar.info("???????????????????????????")

    st.sidebar.markdown('## ????????????????????????')
    hisForm = st.sidebar.multiselect("?????????????????????", ['????????????','????????????'], key="2")

    if(choice_1=='??????'):
        #??????????????????????????????
        img,title=st.beta_columns([1,8])
        img.image(image_1)
        title.write("""# ???????????????????????????""")
        st.write("""*???????????????????????????????????????????????????????????????????????????*""")
        st.write("""*Designed By:  CYD_DLUT*""")
        st.info("""**????????????????????????????????????!**  
            ???????????????????????????????????????????????????????????????????????????????????????""")

    elif(choice_1=='????????????'):
        st.write("""                  

                """)
        st.info("""***???????????????......***""")
        st.warning("??????????????????????????????????????????????????????????????????????????????")

        startTime = time.localtime()
        sv = Server(num_of_clients, epoch, batch_size, num_of_com,is_iid,alChoice)
        sv.main()

        #?????????????????????
        st.write("## ?????????????????????")
        fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, )
        ax = ax.flatten()
        start = random.randint(0, sv.test_data_size - 10)
        for i in range(start, start + 10):
            img = sv.test_data[i].reshape(28, 28)
            ax[i - start].imshow(img, cmap='Greys', interpolation='nearest')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.savefig('./img/test_image/{}.png'.format(time.strftime("%Y%m%d%H%M",startTime)))#????????????
        st.pyplot(fig)


        # ???????????????
        st.write("## ???????????????")
        Acculine = pd.DataFrame(sv.accuList, columns=['Accuracy'])
        st.line_chart(Acculine)

        # ??????????????????
        path = './data/his_data/{}.npy'.format(time.strftime("%Y%m%d%H%M", startTime))
        dataList = sv.paraList +[-1] + sv.total_ll + [-1] + sv.accuList + [sv.num_of_clients]
        dl = np.array(dataList)
        np.save(path,dl)
        st.success('''***???????????????{}***  
                ***????????????!***'''.format(path))
    else:
        if(len(hisForm)==0):
            st.sidebar.error("?????????????????????")
            st.error("????????????????????????????????????")
        hisDatalist=[]

        for i in range(len(option)):
            dpath = './data/his_data/' + files[i]
            his_dl = np.load(dpath, allow_pickle=True)
            his_dl = his_dl.tolist()
            hisDatalist.append(his_dl)

        if len(hisDatalist)==0:
            st.warning("???????????????????????????")

        else:
            for i in range(len(hisForm)):
                if (hisForm[i] == '????????????'):
                    index = 0
                    his_ltable = []
                    st.write('''## ????????????''')
                    for j in range(len(option)):
                        #st.write([hisDatalist[j][index:7]])
                        list=[]
                        list.append('{}'.format(option[j][8:21]))
                        list.extend(hisDatalist[j][index:7])
                        his_ltable.append(list)

                    his_para = pd.DataFrame(his_ltable,
                                        columns=['??????','????????????', 'Num of Com',
                                                 'Epoch', 'Batch_size', 'Learning Rate',
                                                 'Num of Clients', 'C fraction'])
                    st.table(his_para)

                if (hisForm[i] == '????????????'):
                    selector = st.selectbox("????????????????????????????????????", option, key="1")
                    episode = ''
                    for i in range(len(selector)):
                        episode = episode+selector[i]
                        if(selector[i+1]=='.'):
                            break
                    episode = int(episode)-1#?????????episode???????????????

                    fidForm = st.multiselect("?????????????????????", ['?????????????????????', '????????????????????????','???????????????'], key="3")
                    for j in range(len(fidForm)):
                        if fidForm[j]=='?????????????????????':
                            st.write('## ?????????????????????')
                            st.image('./img/test_image/{}'.format(imgs[episode]))

                        elif fidForm[j]=='????????????????????????':
                            st.write('''## ????????????????????????''')
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

                        elif fidForm[j]=='???????????????':
                            st.write('''## ???????????????''')
                            accuLocation=len(hisDatalist[episode])-1-hisDatalist[episode][1]
                            his_aline = pd.DataFrame(hisDatalist[episode][accuLocation: len(hisDatalist[episode]) - 1],
                                                     columns=['Accuracy'])
                            st.line_chart(his_aline)
