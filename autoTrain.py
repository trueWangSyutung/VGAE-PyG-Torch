import os

# 执行训练
# /usr/bin/python3.11 /home/wangsyutung/Desktop/VGAE/train.py
# 分别以VGAE和GAE对Cora、Citeseer、Pubmed进行训练

def CoraTrain():
    # Cora
    print("Cora+VGAE")
    os.system("/usr/bin/python3.11 ./train.py -dataset Cora -model VGAE -inputDim 1433")

    print("Cora+GAE")
    os.system("/usr/bin/python3.11 ./train.py -dataset Cora -model GAE -inputDim 1433")



def CiteseerTrain():
    # Citeseer
    print("Citeseer+VGAE")
    os.system("/usr/bin/python3.11 ./train.py -dataset Citeseer -model VGAE -inputDim 3703")

    print("Citeseer+GAE")
    os.system("/usr/bin/python3.11 ./train.py -dataset Citeseer -model GAE -inputDim 3703")


def PubmedTrain():
    # Pubmed
    print("Pubmed+VGAE")
    os.system("/usr/bin/python3.11 ./train.py -dataset Pubmed -model VGAE -inputDim 500")

    print("Pubmed+GAE")
    os.system("/usr/bin/python3.11 ./train.py -dataset Pubmed -model GAE -inputDim 500")



if __name__ == '__main__':
    CoraTrain()
    CiteseerTrain()
    PubmedTrain()

