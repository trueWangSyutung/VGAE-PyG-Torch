import os

# 执行训练
# /usr/bin/python3.11 /home/wangsyutung/Desktop/VGAE/train.py
# 分别以VGAE和GAE对Cora、Citeseer、Pubmed进行训练

def CoraTrain():
    # Cora
    print("Cora+VGAE")
    os.system("/usr/bin/python3.11 ./train.py -dataset Cora -model VGAE -inputDim 1433")
    # rocList mean 0.9108033370659889 = 91.08% = 91.1%
    # apList mean 0.9216454449848184 = 92.16% = 92.2%
    print("Cora+GAE")
    os.system("/usr/bin/python3.11 ./train.py -dataset Cora -model GAE -inputDim 1433")
    # rocList mean 0.9092147381080119 = 90.92% = 90.9%
    # apList mean 0.9208139334139614 = 92.08% = 92.1%


def CiteseerTrain():
    # Citeseer
    #print("Citeseer+VGAE")
    os.system("/usr/bin/python3.11 ./train.py -dataset Citeseer -model VGAE -inputDim 3703")
    # rocList mean 0.8694814635913538 = 86.95% = 86.9%
    # apList mean 0.8930089423134533 = 89.30% = 89.3%
    #print("Citeseer+GAE")
    #os.system("/usr/bin/python3.11 ./train.py -dataset Citeseer -model GAE -inputDim 3703")
    # rocList mean 0.8714126313247192 = 87.14% = 87.1%
    # apList mean 0.8895926274046977 = 88.96% = 89.0%

def PubmedTrain():
    # Pubmed
    print("Pubmed+VGAE")
    os.system("/usr/bin/python3.11 ./train.py -dataset Pubmed -model VGAE -inputDim 500")
    # rocList mean 0.9603508064910269 = 96.04% = 96.0%
    # apList mean 0.9604580939861425 = 96.05% = 96.1%
    #print("Pubmed+GAE")
    #os.system("/usr/bin/python3.11 ./train.py -dataset Pubmed -model GAE -inputDim 500")
    # ap mean 0.965735
    # roc mean 0.964935


if __name__ == '__main__':
    CiteseerTrain()
    #PubmedTrain()

