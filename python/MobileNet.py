import torch
import torchvision
from PIL import Image



# model=torchvision.models.mobilenet_v2(weights='DEFAULT') # try on google colab, if does not work locally
# model.eval()
# exmp=torch.randn(1,3,224,224)
# m=torch.jit.trace(model,exmp)
# torch.jit.save(m,'mnet.pt')

# s=optimize_for_mobile(m)
# s._save_for_lite_interpreter('mmm.ptl') # this prediction is not correct


# load saved model
model=torch.jit.load('mnet.pt')
model.eval()
count=sum(p.numel() for p in model.parameters())
print(f"prams: {count}")
img=Image.open('hen.jpeg').resize((224,224))
img_tns=torchvision.transforms.ToTensor()(img).unsqueeze(0)
result=model(img_tns) # total output class 1000
class_index=torch.argmax(result)
print(f'index: {class_index}')