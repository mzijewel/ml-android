import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

# pick always same random values
torch.manual_seed(41)

data = torch.randn(10,3)

a=data[:,0:1] # 1st column
b=data[:,1:2] # 2nd column
c=data[:,2:3] # 3rd column

y=2*a+3*b+4*c

# create model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,1) # input 3, output 1


    def forward(self,x):
        return self.linear(x)
    


model=Model()

loss_fn=nn.MSELoss()
optmizer=torch.optim.SGD(model.parameters(),lr=0.01)

# Train the model
epochs = 600
losses = []
for epoch in range(epochs):

    y_pred = model(data)

    # Compute loss
    loss = loss_fn(y_pred, y)

    # Backward pass and optimization
    optmizer.zero_grad()
    loss.backward()
    optmizer.step()

    # Track loss
    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# evaluate flag on
model.eval()
input=torch.tensor([1,2,3],dtype=torch.float32) 
with torch.no_grad():
    p=model(input) # 2a+3b+4c
    print(p) # tensor([19.6101])


input_shape=torch.randn(1,3) # input=3
script=torch.jit.trace(model,input_shape)
torch.jit.save(script,'m_eqn.pt') # lower size

# optimized_script=optimize_for_mobile(script)
# optimized_script._save_for_lite_interpreter("m_eqn2.ptl") # little bit larger size

