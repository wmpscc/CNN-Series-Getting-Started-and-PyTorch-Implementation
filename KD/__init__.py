import torch
from torch import nn, optim
from torch.nn import functional as F
from utils.load_data_Fnt10 import load_data_Fnt10
from utils import evaluate_accuracy
from tqdm import tqdm
from KD.Net import TeacherNet, StudentNet, Student2Net

if __name__ == '__main__':
    INPUT_SIZE = 112
    BATCH_SIZE = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacherNet = TeacherNet(10)
    teacherNet.load_state_dict(torch.load("./teacherNet.pth"))
    teacherNet.eval()
    teacherNet.train(mode=False)
    teacherNet = teacherNet.to(device)


    studentNet = Student2Net(classes=10)
    studentNet.load_state_dict(torch.load("./studentNet-ST.pth"))
    studentNet = studentNet.to(device)
    optimizer = optim.Adam(studentNet.parameters(), lr=1e-3)
    lossCE = nn.CrossEntropyLoss()
    lossKD = nn.KLDivLoss()

    trainDL, valDL = load_data_Fnt10(INPUT_SIZE, BATCH_SIZE)

    num_epochs = 30
    T, lambda_stu = 5.0, 0.05
    for epoch in range(num_epochs):
        sum_loss = 0
        sum_acc = 0
        batch_count = 0
        n = 0
        for X, y in tqdm(trainDL):
            X = X.to(device)
            y = y.to(device)
            y_student = studentNet(X)

            loss_student = lossCE(y_student, y)
            y_teacher = teacherNet(X)
            loss_teacher = lossKD(F.log_softmax(y_student / T, dim=1),
                                  F.softmax(y_teacher / T, dim=1))
            loss = lambda_stu* loss_student + (1 - lambda_stu) * T * T * loss_teacher
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            sum_acc += (y_student.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(valDL, studentNet)
        print("epoch %d: loss=%.4f \t acc=%.4f \t test acc=%.4f" % (epoch + 1, sum_loss / n, sum_acc / n, test_acc))
    torch.save(studentNet.state_dict(), './studentNet-ST.pth')