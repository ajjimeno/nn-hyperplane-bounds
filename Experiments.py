import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from Losses import MultiHuberLoss

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32

def train(model, criteria, training_set, testing_set, optim_wd=0.0, lr=0.001, epochs=100):
    optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-08)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, verbose=True)

    training_set_size = len(training_set) * batch_size
    owd_factor = optim_wd

    print("owd_factor:", owd_factor)

    for e in range(epochs):
        model.train()

        total_loss = 0
        count = 0
        print ("Epoch :", e, flush=True)

        total_accuracy = 0
        count_labels = 0

        dl2 = 0

        dl2_sv = 0
        svs = 0


        for step, batch in enumerate(training_set):
            model.zero_grad()
            #print (e, " ", count, flush=True)
            data = batch[0].to(device)
            labels = batch[1].to(device)

            output, z = model(data)

            loss = criteria(output, labels)

            if owd_factor > 0:
                wl2 = torch.dot(model.fc2.weight.flatten(), model.fc2.weight.flatten())
                loss += owd_factor * wl2
        
                norm = torch.dot(z.flatten(),z.flatten())
                #norm = torch.dot(z.flatten(),z.flatten())
                loss += owd_factor * norm

                if isinstance(criteria, MultiHuberLoss):
                    # Find support vectors
                    for i in torch.logical_or(torch.isclose(output, torch.tensor(1.0)),
                                     torch.isclose(output, torch.tensor(-1.0))
                                     ).nonzero():
                        ind = i[0].item()
                        mnorm = torch.dot(z[ind].flatten(), z[ind].flatten())

                        svs += 1

                        if mnorm > dl2_sv:
                            dl2_sv = mnorm

                if norm > dl2:
                    dl2 = norm

            total_loss += loss.item()
            total_accuracy += flat_accuracy(output.clone().detach(), labels)
            count_labels += len(labels)
            count += 1

            loss.backward()

            optimizer.step()

        print ("loss: ", total_loss/count, flush=True)
        print ("training acc: ", total_accuracy/count_labels, flush=True)

        if owd_factor > 0:
            print("dl2: ", dl2)
            print("wl2: ", torch.dot(model.fc2.weight.flatten(), model.fc2.weight.flatten()))

            if isinstance(criteria, MultiHuberLoss):
                print("dl2 sv: ", dl2_sv)
                print("svs: ", svs)

        if e > 0 and e % 25 == 0: 
            print ("Testing acc: ", predict(model, testing_set))

        scheduler.step()

def flat_accuracy(preds, labels):
    pred_flat = torch.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return torch.sum(pred_flat == labels_flat).item() 

def predict(model, testing_set):

    model.eval()

    total_accuracy = 0
    count = 0

    for step, batch in enumerate(testing_set):
        data = batch[0].to(device)
        labels = batch[1].to(device)

        with torch.no_grad():
            output, _ = model(data)

            total_accuracy += flat_accuracy(output.clone().detach(), labels)
            count += len(labels)

    return total_accuracy/count

import MNIST
import CIFAR

if __name__ == '__main__':
    criteria = [nn.CrossEntropyLoss().to(device), MultiHuberLoss().to(device)]

    dropout_values = [False, True]

    training_set_percentages = [ 1, 5, 10, 20, 40, 60, 80, 100 ]

    aug_values = [ False, True ]

    #sets = [ MNIST, CIFAR ]
    sets = [ MNIST ]

    for s in sets:

        testing_set = s.testloader()

        for tsp in training_set_percentages:
            for c in criteria:
                for aug in aug_values:
                    for optim_wd in s.owd_weights:
                        for dropout in dropout_values:
                            for i in range(4):
                                training_set =  s.trainloader(tsp, aug)

                                nmodel = s.model(dropout).to(device)
                                train(nmodel, c, training_set, testing_set, optim_wd=optim_wd, lr=s.lr, epochs=s.epochs)

                                train_result = predict(nmodel, training_set) 
                                test_result = predict(nmodel, testing_set)

                                print ("Set: ", s.name)
                                print ("Training set size %: ", tsp)
                                print ("Criteria: ", c)
                                print ("Aug: ", aug)
                                print ("optim wd: ", optim_wd)
                                print ("drop out: ", dropout)
                                print ("Training acc: ", train_result)
                                print ("Testing acc: ", test_result) 

                                print (s.name, \
                                    "|Net|tsp|",tsp, \
                                    "|crit|", c, \
                                    "|aug|", aug, \
                                    "|owd|", optim_wd, \
                                    "|do|", dropout, \
                                    "|training|", train_result, \
                                    "|testing|", test_result)
