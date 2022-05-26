import time
from tqdm import tqdm
import torch

def train_per_epoch(model,optimizer,loss,lr_schedule,epoch,dataloader,device,printf,batch):
    start=time.time()
    all_loss=0
    all_accNum=0
    model.train()
    for idx,(img,labels) in enumerate(tqdm(dataloader)):  # 耗时
        img=img.to(device)
        # print("img size: ", img.size())
        labels=labels.to(device) # 实际标签
        out=model(img)           # 输出：([bacth, classNum])
        los=loss(out,labels)

        
        los.backward()         # 反向传播计算每个参数梯度
        optimizer.step()       # 通过梯度下降执行一次参数更新
        optimizer.zero_grad()  # 梯度置零
        all_loss+=los.item()

        # out.data.max(dim=1)[1] 以最大值对应的索引为分类输出类别.
        # 如果该结果和实际标签类别labels相同，及为分类正确
        cur_acc=(out.data.max(dim=1)[1]==labels).sum()  
        all_accNum+=cur_acc
        # 每处理printf个批量输出一次结果
        if (idx%printf)==0:
            print('epoch:{} training:[{}/{}] loss:{:.6f} accuracy:{:.6f}% lr:{}'.format(epoch,idx,len(dataloader),los.item(),cur_acc*100/len(labels),optimizer.param_groups[0]['lr']))

        lr_schedule.step(los.item())

    end=time.time()
    #训练完一个epoch算一次平均损失以及平均准确率
    all_loss/=len(dataloader)
    acc=all_accNum*100/(len(dataloader)*batch)
    print('epoch:{} time:{:.2f} seconds training_loss:{:.6f} training_accuracy:{:.6f}%'.format(epoch,end-start,all_loss,acc))
    return all_loss

@torch.no_grad()
def val(dataloader,model,device,epoch):
    start=time.time()
    model.eval()
    all_acc=0
    for idx,(img,labels) in enumerate(dataloader):
        img=img.to(device)
        labels=labels.to(device)
        out=model(img)

        cur_accNum=(out.data.max(dim=1)[1]==labels).sum()/len(labels)
        all_acc+=cur_accNum
    end=time.time()
    print('epoch:{} val_time:{:.2f} seconds val_accuracy:{:.6f}%'.format(epoch,end-start,all_acc*100/len(dataloader)))
    return all_acc/len(dataloader)