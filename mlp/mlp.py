"""
We need training dataset: MNIST
You can load MNIST from Pytorch Dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from torchmetrics import Accuracy

import tqdm

transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ])

train_mnist = torchvision.datasets.MNIST('./mnist', train=True, download=True, transform=transform)


train_loader = torch.utils.data.DataLoader(train_mnist, batch_size=4, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./mnist', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ])),
  batch_size=4, shuffle=False)

"""
Because MNIST dataset is 2D (not 1D)
"""

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        """
        Inheritance
        """
        self.model = nn.Sequential(
                        nn.Linear(784, 100),
                        nn.ReLU(),
                        nn.Linear(100, 50),
                        nn.ReLU(),
                        nn.Linear(50, 10),
                        nn.Sigmoid()
                      )

    def forward(self, image):
        input_vector = image.view(-1, 28 * 28)
        logits = self.model(input_vector)
        return logits

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_fn = nn.CrossEntropyLoss()
train_loss = 0.0

print("Training Started............................")
num_epochs = 1
for epoch in range(num_epochs):
    idx = 1
    for images, labels in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels) 
        loss.backward()
        optimizer.step()
        idx += 1
        train_loss += loss
    train_loss = train_loss / idx
    print("Training Loss: ", train_loss.item())

print("Training Finished............................")


print("Evaluation Started............................")

model.eval() #turn-off the gradients

#1. training phase --> update model weights (parameters)
#2. testing phase --> do not update model weights (parameters) 
#                 --> turn-off the gradients

preds_list = []
labels_list = []

metrics = Accuracy(task='multiclass', num_classes=10) #define object 

for images, labels in tqdm.tqdm(test_loader):
    outputs = model(images)
    outputs = F.softmax(outputs, dim=1) #[0.1, 0.2, 0.005, 0.003, 0.4, ...,0.03]
    preds = torch.argmax(outputs, 1) #return the position has the greatest value
                  #argmax --> argument maximum -> position of the maximum value
    preds_list += preds.cpu().numpy().tolist() #Python integer
    labels_list += labels.cpu().numpy().tolist()

preds_list = torch.tensor(preds_list)
labels_list = torch.tensor(labels_list)

accs = metrics(preds_list, labels_list)
print("Accuracy: ", accs.item())

print("Evaluation Finished............................")

"""
Homework:
 - What is AUC?
   - AUC is only for binary classification
   - AUC range from 0 to 1
   - AUC is the area under the ROC curve
     - higher AUC --> better model
       * To calculate AUC:
           - 1. True positive rate (TPR)
           - 2. False positive rate (FPR)
     - Measure performance of the model across all classification thresholds???
       - Thresholds: decision boundary
           - if the probability is greater than the threshold --> positive
    - When you have imbalanced dataset?
      - 100 samples: 80 negative, 20 positive samples
      What happens if your model predicts all samples as negative?
      What happens if your model predicts all samples as positive?
      
     
 - AUC and ROC Curve
 - Confusion Matrix
 - Sensitive, Specificity, Precision, Recall, F1-Score
 - True Prediction Rate and False Prediction Rate
 - True Positive and False Negative, True Negative and False Positive
 - Fair and Unfair Model
     -> Toss a fair coin: 50% head, 50% tail
 - Deep Learning mdoels are not unfair (has bias)
     -> if your data has bias --> your model has bias
  - Dataset for Covid-19 --> only for White people
  -     if you use the model for Black people --> the model is unfair
  - Unfair AI --> polistic
"""