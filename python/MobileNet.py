import torch
import torch.utils
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchvision
from PIL import Image



def save_model():

    model=torchvision.models.mobilenet_v2(weights='DEFAULT') # try on google colab, if does not work locally
    model.eval() # it is important to get accurate output
    exmp=torch.randn(1,3,224,224)
    m=torch.jit.trace(model,exmp)
    torch.jit.save(m,'mnet1.pt')

    # s=optimize_for_mobile(m)
    # s._save_for_lite_interpreter('mnet.ptl') # this prediction is not correct

def test_model():
    images= ["burger.jpg","car.jpeg","cat.jpg","plane.jpg","dog.jpg"]
    for image in images:
        # load saved model
        model=torch.jit.load('mnet.pt')
        model.eval()
        # count=sum(p.numel() for p in model.parameters())
        # print(f"prams: {count}")
        img=Image.open(image).resize((224,224))
        img_tns=torchvision.transforms.ToTensor()(img).unsqueeze(0)
        result=model(img_tns) # total output class 1000
        # class_index=torch.argmax(result)
        # print(f'index: {class_index}')

        probabilities=torch.softmax(result[0],dim=0)
        # class_index=torch.argmax(result)
        max_val,max_indx=torch.max(probabilities,dim=0)
        
        print(f'{image} --> {max_val*100:0.2f}% : index {max_indx}')

test_model()
# save_model()