import torch
from PIL import Image
import torchvision.transforms as transforms
from model import MobileNet_v2
import argparse

def test(opt):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=MobileNet_v2().to(device)
    models=torch.load(opt.weights)
    net.load_state_dict(models['model'])

    img=Image.open(opt.picpath)
    img=transform(img)
    img=img[None].to(device)

    with torch.no_grad():
        net.eval()
        out=net(img).data.max(dim=1)[1]
    print('test result: {}'.format(opt.classes[out.item()]))
if __name__ == '__main__':
    parse=argparse.ArgumentParser()
    classes=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    parse.add_argument('--weights',type=str,default='weights/mobilenet_v2_15.pth',help='weight path')
    parse.add_argument('--picpath', type=str, default='test/test.jpeg', help='weight path')
    parse.add_argument('--classes', type=list, default=classes, help='weight path')
    opt=parse.parse_args()

    test(opt)