#Pytorch imports

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import tqdm_notebook

torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#Miscellaneous Utilities 

import numpy as np
import scipy.linalg as la
import pandas as pd
import random
from matplotlib import pyplot as plt
import math



def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
	
def columnize(x):
    return x.reshape((len(x),1))
	
class Basic_Classification_Dataset(Dataset):
    def __init__(self,data, mode = 'train'):
        self.mode = mode
        

        if self.mode == 'train':

            self.inp = data[:,:-1]
            self.oup = data[:,-1]
        else:
            self.inp = data
    def __len__(self):
        return len(self.inp)
    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt  = torch.Tensor(self.inp[idx])
            oupt  = torch.Tensor([self.oup[idx]]).long()
            return { 'inp': inpt,
                     'oup': oupt,
            }
        else:
            inpt = torch.Tensor(self.inp[idx])
            return { 'inp': inpt
            }

			
class MIMO_Dataset(Dataset):
    def __init__(self,seperated_set,portion_same_class = 0.5, portion_different_class = 0.5):
        self.seperated_set = seperated_set   
        self.class_number = len(seperated_set)
        self.class_sizes = np.array([i.shape[0] for i in self.seperated_set])
        self.item_shape = self.seperated_set[0].shape[1]
        self.output_shape = 2*self.item_shape
        self.portion_same_class = portion_same_class
        self.portion_different_class = portion_different_class
        self.duplication_percent = 1-(self.portion_same_class+self.portion_different_class)
    def __len__(self):
        return np.sum(np.array([i.shape[0] for i in self.seperated_set]))
    def __getitem__(self, idx):
        inpt = np.empty(self.output_shape)
        oupt = 0
        random_choice = np.random.rand()
        if random_choice < self.portion_same_class:#draw from same class
            my_class = np.random.randint(self.class_number) 
            locations = np.random.randint(self.class_sizes[my_class],size = 2)
            inpt[:self.item_shape]=self.seperated_set[my_class][locations[0]]
            inpt[self.item_shape:]=self.seperated_set[my_class][locations[1]]
        elif random_choice < (self.portion_same_class+self.portion_different_class):   # draw from differnt classes        
            my_classes = np.random.choice(self.class_number, size = 2, replace=False) 
            location0 = np.random.randint(self.class_sizes[my_classes[0]],size = 1)
            location1 = np.random.randint(self.class_sizes[my_classes[1]],size = 1)
            inpt[:self.item_shape]=self.seperated_set[my_classes[0]][location0]
            inpt[self.item_shape:]=self.seperated_set[my_classes[1]][location1]
            oupt = 1
        else:  #duplicate element          
            my_class = np.random.randint(self.class_number)
            location = np.random.randint(self.class_sizes[my_class],size = 1)
            inpt[:self.item_shape]=self.seperated_set[my_class][location]
            inpt[self.item_shape:]=self.seperated_set[my_class][location]
            
        inpt = torch.tensor(inpt).float()       

            
        oupt = torch.Tensor(np.array([oupt])).long()
                
        return { 'inp': inpt,'oup': oupt}
    
    def get_specific(self, class1, class2, loc1, loc2):
        inpt = torch.empty(self.output_shape)
        inpt[:self.channels]=self.seperated_set[class1,loc1]
        inpt[self.channels:]=self.seperated_set[class2,loc2]

        return inpt
		
def check_verifier(verifier,classifier,test_set, batch_size):
    
    full_verifier_predictions =np.array([])
    full_classifier_predictions =np.array([])
    
    verifier_accuracy = 0
    classifier_accuracy = 0
    
    #ones = 0
    #zeros = 0
    
    data_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = False)
    
    classifier.eval()
    verifier.eval()
    
    for bidx, batch in tqdm(enumerate((data_loader)),total = len(data_loader)):
        
        x_val, y_val = batch['inp'], batch['oup']
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        y_val = y_val.flatten()
        half = x_val.shape[1]//2
        #print(half)
        #print(x_val.shape)
        #print(y_val)
        with torch.no_grad():
            verifier_predictions = torch.nn.functional.softmax(verifier(x_val),dim = 1)[:,0]
            raw_classifier_predictions1 = torch.nn.functional.softmax(classifier(x_val[:,:half]),dim = 1)
            raw_classifier_predictions2 = torch.nn.functional.softmax(classifier(x_val[:,half:]),dim = 1)
        classifier_predictions = torch.sum(raw_classifier_predictions1 * raw_classifier_predictions2,dim = 1)
        
        for i in range(len(verifier_predictions)): 
            if y_val[i] == 1:
                #ones+=1
                if verifier_predictions[i] < 0.5:
                   verifier_accuracy += 1 
                if classifier_predictions[i] < 0.5:
                   classifier_accuracy += 1 
            else:
                #zeros+=1
                if verifier_predictions[i] >= 0.5:
                   verifier_accuracy += 1 
                if classifier_predictions[i] >= 0.5:
                   classifier_accuracy += 1 
        
        full_verifier_predictions = np.append(full_verifier_predictions,verifier_predictions.detach().cpu().numpy())
        full_classifier_predictions = np.append(full_classifier_predictions,classifier_predictions.detach().cpu().numpy())
    
    verifier_accuracy = verifier_accuracy / len(test_set)
    classifier_accuracy = classifier_accuracy / len(test_set)
    
    #print(ones)
    #print(zeros)
    #print((ones+zeros)/len(test_set))
    
    plt.plot([0,1],[0,1])
    plt.scatter(full_verifier_predictions,full_classifier_predictions)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.show()
    
    return verifier_accuracy, classifier_accuracy, full_verifier_predictions, full_classifier_predictions
			
class softmaxed_network(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.softener = nn.Softmax(dim = -1)
    
    def forward(self,x):
        logits = self.classifier(x)
        probs = self.softener(logits)
        return probs
		
class Basic_fully_connected_Network(nn.Module):
    def __init__(self, input_size, class_number):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 60)
        self.layer2 = nn.Linear(60, 60)
        self.layer3 = nn.Linear(60, 60)
        self.layer4 = nn.Linear(60, class_number)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        logits = self.layer4(x)
        
        return logits.float()
		
class Enhanced_fully_connected_Network(nn.Module):
    def __init__(self, input_size, class_number):
        super().__init__()
        self.layer0 = nn.Linear(input_size,120)
        self.layer1 = nn.Linear(120, 120)
        self.layer2 = nn.Linear(120, 120)

        self.layer3 = nn.Linear(120, class_number)
        
        self.dropout0 = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        
    def forward(self, x):

        x = F.relu(self.layer0(x))
        x = self.dropout0(x)
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)

        logits = self.layer3(x)

        
        return logits.float()
		
def even_triple_split(data,training_percent=0.8,testing_percent=0.1):
    attribute_count = data.shape[1]
    class_labels,class_sizes = np.unique(data[:,-1],return_counts=True)
    
    label_count = class_labels.shape[-1]
    
    training_data_sizes = (class_sizes*training_percent).astype(int)
    training_cut_offs = np.zeros(label_count+1,dtype = int)
    training_cut_offs[1:] = np.cumsum(training_data_sizes)
    
    testing_data_sizes = (class_sizes*testing_percent).astype(int)
    testing_cut_offs = np.zeros(label_count+1,dtype = int)
    testing_cut_offs[1:] = np.cumsum(testing_data_sizes)
    
    validation_data_sizes = class_sizes-(training_data_sizes+testing_data_sizes)
    validation_cut_offs = np.zeros(label_count+1,dtype = int)
    validation_cut_offs[1:] = np.cumsum(validation_data_sizes)
    
    seperated_data = []
    
    for i in class_labels:
        seperated_data.append(data[data[:,-1]==i])


    training_data = np.empty((np.sum(training_data_sizes),attribute_count))
    testing_data = np.empty((np.sum(testing_data_sizes),attribute_count))
    validation_data = np.empty((np.sum(validation_data_sizes),attribute_count))
    
    for i in range(label_count):
        training_data[training_cut_offs[i]:training_cut_offs[i+1]]=seperated_data[i][0:training_data_sizes[i]]
        testing_data[testing_cut_offs[i]:testing_cut_offs[i+1]]=seperated_data[i][training_data_sizes[i]:training_data_sizes[i]+testing_data_sizes[i]]
        validation_data[validation_cut_offs[i]:validation_cut_offs[i+1]]=seperated_data[i][training_data_sizes[i]+testing_data_sizes[i]:]
        

    
    return training_data,testing_data,validation_data
	
def upsample(data_sets):
    lengths = []
    for i in data_sets:
        lengths.append(i.shape[0])
    lengths = np.array(lengths)
    
    class_number = len(lengths)
    
    target = np.max(lengths)
    max_location = np.argmax(lengths)    
    
    output = np.empty((class_number*target,*data_sets[0][0].shape))
    
    for i in range(class_number):
        copies = target//lengths[i]
        extras = target%lengths[i]
        
        for j in range(copies):
            output[i*target+j*lengths[i]:i*target+(j+1)*lengths[i]] = data_sets[i]
            
        output[i*target +copies*lengths[i]:(i+1)*target] = data_sets[i][:extras]
        
    return output
    
    
def get_datasets_from_dataframe(dataframe):
    array = dataframe.to_numpy()
    training_array, testing_array, validation_array = even_triple_split(array)
    
    sorted_training_data = [training_array[np.where(training_array[:,-1]==1)],training_array[np.where(training_array[:,-1]==0)]]
    sorted_testing_data = [testing_array[np.where(testing_array[:,-1]==1)],testing_array[np.where(testing_array[:,-1]==0)]]
    sorted_validation_data = [validation_array[np.where(validation_array[:,-1]==1)],validation_array[np.where(validation_array[:,-1]==0)]]
    
    upsampled_training_data = upsample(sorted_training_data)
    upsampled_testing_data = upsample(sorted_testing_data)
    upsampled_validation_data = upsample(sorted_validation_data)
    
    training_dataset = Basic_Classification_Dataset(upsampled_training_data)
    testing_dataset = Basic_Classification_Dataset(upsampled_testing_data)
    validation_dataset = Basic_Classification_Dataset(upsampled_validation_data)
    
    return training_dataset, testing_dataset, validation_dataset
    
def get_difference_datasets_from_dataframe(dataframe):
    array = dataframe.to_numpy()
    training_array, testing_array, validation_array = even_triple_split(array)
    
    sorted_training_data = [training_array[np.where(training_array[:,-1]==1)],training_array[np.where(training_array[:,-1]==0)]]
    sorted_testing_data = [testing_array[np.where(testing_array[:,-1]==1)],testing_array[np.where(testing_array[:,-1]==0)]]
    sorted_validation_data = [validation_array[np.where(validation_array[:,-1]==1)],validation_array[np.where(validation_array[:,-1]==0)]]
    

    
    difference_training_data = MIMO_Dataset([sorted_training_data[0][:,:-1],sorted_training_data[1][:,:-1]],portion_same_class = 0.5, portion_different_class = 0.5)
    difference_testing_data = MIMO_Dataset([sorted_testing_data[0][:,:-1],sorted_testing_data[1][:,:-1]],portion_same_class = 0.5, portion_different_class = 0.5)
    difference_validation_data = MIMO_Dataset([sorted_validation_data[0][:,:-1],sorted_validation_data[1][:,:-1]],portion_same_class = 0.5, portion_different_class = 0.5)

    
    return difference_training_data, difference_testing_data, difference_validation_data
	
def check_net_accuracy(NET, testing_data_set, batch_size, verbose = False):
    
    
    accuracy = 0
    data_test = DataLoader(dataset = testing_data_set, batch_size = batch_size, shuffle = False)
    NET.eval()
    for bidx, batch in tqdm(enumerate((data_test)),total = len(data_test)):
        x_val, y_val = batch['inp'], batch['oup']
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        y_val = y_val.flatten()
        with torch.no_grad():
            predictions = NET(x_val)
            
        if batch_size == 1:
            if predictions.argmax() == y_val.item():

                accuracy += 1
        else:
            for idx, i in enumerate(predictions):


                if i.argmax() == y_val[idx].item():

                    accuracy += 1
                
    accuracy = (accuracy/len(testing_data_set))

    if verbose:
        print('Model Accuracy : '+str(accuracy*100))
        
    return accuracy
	
def check_classwise_net_accuracy(NET, testing_data_set, class_number, batch_size, verbose = False):
    
    
    guesses = np.zeros((class_number,class_number))
    true_counts = np.zeros(class_number)
    
    data_test = DataLoader(dataset = testing_data_set, batch_size = batch_size, shuffle = False)
    NET.eval()
    for bidx, batch in tqdm(enumerate((data_test)),total = len(data_test)):
        x_val, y_val = batch['inp'], batch['oup']
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        y_val = y_val.flatten()
        with torch.no_grad():
            predictions = NET(x_val)
        
        for idx, i in enumerate(predictions):
            true_class = y_val[idx].item()
            guessed_class = i.argmax()
            true_counts[true_class] += 1
            guesses[true_class,guessed_class] += 1
                
    guesses = (guesses.T/true_counts).T
    accuracy = np.copy(np.diag(guesses))
    np.fill_diagonal(guesses,0)

    if verbose:
        print('Overall Model Accuracy: '+str(100*np.mean(accuracy)))
        print('Model Accuracy by Class:')
        print(accuracy)
        print('False Postive Rates by Class:')
        print(np.sum(guesses,axis = 1))
        print('False Postive Rates by Type:')
        print(guesses)
        
    return accuracy,guesses
	
def check_net_calibaration(NET, testing_data_set, batch_size, temperature, bins = 15, verbose = False):
    
    bin_totals = np.zeros(bins)
    bin_corrects = np.zeros(bins)
    all_guess_values = np.zeros(len(testing_data_set))
    
    total_inputs = len(testing_data_set)
    
    accuracy = 0
    data_test = DataLoader(dataset = testing_data_set, batch_size = batch_size, shuffle = False)
    NET.eval()
    for bidx, batch in tqdm(enumerate((data_test)),total = len(data_test)):
        x_val, y_val = batch['inp'], batch['oup']
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        y_val = y_val.flatten()
        with torch.no_grad():
            predictions = torch.nn.functional.softmax(NET(x_val)/temperature,dim = -1)
        #print(predictions)
        all_guess_values[bidx*batch_size:(bidx+1)*batch_size] = np.max(predictions.detach().cpu().numpy(),axis = 1)
        

        for idx, i in enumerate(predictions):
            place = torch.max(i)*bins

            place = int(place.item())

            if place == bins:
                place -= 1

            bin_totals[place] += 1

            if i.argmax() == y_val[idx].item():
                bin_corrects[place] += 1
                accuracy += 1
    

    divisible_totals = np.copy(bin_totals)
    
    divisible_totals[np.where(divisible_totals == 0)] += 1
        
    accuracies = bin_corrects/(divisible_totals)
    accuracy /= total_inputs
    
    average_confidence = np.mean(all_guess_values)
    
    confidences = np.linspace(0.5/bins,1-0.5/bins,bins)

    
    ECE = np.sum(np.abs(accuracies-confidences)*bin_totals)/total_inputs

    if verbose:
        plt.hist(all_guess_values,bins = bins,weights=np.ones(total_inputs) / total_inputs)
        plt.plot([accuracy,accuracy],[0,1],label = 'Accuracy')
        plt.plot([average_confidence,average_confidence],[0,1],label = 'Average Confidence')
        plt.xlim(0,1)
        plt.ylabel('Percent of Samples')
        plt.legend()
        plt.show()


        plt.bar(np.linspace(0,1,bins+1)[:-1]+0.5/bins, accuracies,width = 1/bins)
        plt.scatter(np.linspace(0,1,bins+1)[:-1]+0.5/bins, accuracies)
        plt.plot([0+0.5/bins,1-0.5/bins],[0+0.5/bins,1-0.5/bins],color = 'r')
        plt.title(temperature)
        plt.ylabel('Accuracy')
        plt.show()
        
    return ECE
	
	
def train_neural_net(NET, parameter_file_name, convergence_param, max_epochs, batch_size , training_data_set,validation_data_set,testing_data_set = None, new_net = False):
    
    def train(model, x, y, optimizer, criterion):
        model.train()
        model.zero_grad()
        output = model(x)
        loss = criterion(output,y)
        loss.backward()
        optimizer.step()
        return loss, output
    
    def test(model, x, y, criterion):
        model.eval()
        with torch.no_grad():
            output = model(x)
            loss = criterion(output,y)
        return loss, output
    

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(NET.parameters(), lr = 0.0001)
    
    Training_Accuracy_History = np.zeros(max_epochs)
    Validation_Accuracy_History = np.zeros(max_epochs)
    Training_Loss_History = np.zeros(max_epochs)
    Validation_Loss_History = np.zeros(max_epochs)


    best_model_state = NET.state_dict()

    train_acc = 0
    test_acc = 0

    best_epoch = -1


    best_loss = 0
    best_accuracy = 0
    
    epochs_since_improvement = 0
    epoch_count = 0

    data_train = DataLoader(dataset = training_data_set, batch_size = batch_size, shuffle = True)
    data_validate = DataLoader(dataset = validation_data_set, batch_size = batch_size, shuffle = False)
    data_test = DataLoader(dataset = testing_data_set, batch_size = batch_size, shuffle = False)
    
    if not new_net:
        
        NET.load_state_dict(torch.load(parameter_file_name))    
        
        for bidx, batch in tqdm(enumerate(data_validate),total = len(data_validate)):
            x_val, y_val = batch['inp'], batch['oup']
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_val = y_val.flatten()
            loss, predictions = test(NET,x_val,y_val, criterion)
            best_loss+=loss
        
            for idx, i in enumerate(predictions):
                if i.argmax() == y_val[idx].item():
                    best_accuracy += 1
                
        best_accuracy = (best_accuracy/len(validation_data_set))
        
        print('Initial Validation Accuracy : '+str(best_accuracy*100))
        print('Initial Validation Loss : '+str(best_loss))
    
              
    for epoch in range(max_epochs):
        epoch_test_loss = 0
        epoch_train_loss = 0
        train_acc = 0
        test_acc = 0
        for bidx, batch in tqdm(enumerate(data_train),total = len(data_train)):
            x_train, y_train = batch['inp'], batch['oup']
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            y_train = y_train.flatten()
            loss, predictions = train(NET,x_train,y_train, optimizer, criterion)
            epoch_train_loss+=loss
        
        
        
            for idx, i in enumerate(predictions):
                if i.argmax() == y_train[idx].item():
                    train_acc += 1
        train_acc = (train_acc/len(training_data_set))

    
        for bidx, batch in tqdm(enumerate(data_validate),total = len(data_validate)):
            x_val, y_val = batch['inp'], batch['oup']
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_val = y_val.flatten()
            loss, predictions = test(NET,x_val,y_val, criterion)
            epoch_test_loss+=loss
        
            for idx, i in enumerate(predictions):
                if i.argmax() == y_val[idx].item():
                    test_acc += 1
            
        test_acc = (test_acc/len(validation_data_set))
    
    
        Training_Accuracy_History[epoch] = train_acc
        Validation_Accuracy_History[epoch] = test_acc
        Training_Loss_History[epoch] = epoch_train_loss
        Validation_Loss_History[epoch] = epoch_test_loss
        epoch_count += 1
    
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch
            print ("New best found")
            torch.save(NET.state_dict(), parameter_file_name)
            epochs_since_improvement = 0
        elif (test_acc >= best_accuracy) and (epoch_test_loss <= best_loss):
            best_accuracy = test_acc
            best_epoch = epoch
            print ("New best found")
            torch.save(NET.state_dict(), parameter_file_name)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        
        
        
        print('Epoch {} Training Accuracy : {}'.format(epoch+1, train_acc*100))
 
        print('Epoch {} Validation Accuracy : {}'.format(epoch+1, test_acc*100))
 
        
        if epochs_since_improvement >= convergence_param:
            print('Network converged')
            break
            
    if epochs_since_improvement < convergence_param:
        print('Maximum epochs exceeded')
    
    NET.load_state_dict(torch.load(parameter_file_name))    
              
    if testing_data_set != None:
        final_loss = 0
        final_accuracy = 0
        for bidx, batch in tqdm(enumerate(data_test),total = len(data_test)):
            x_test, y_test = batch['inp'], batch['oup']
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            y_test = y_test.flatten()
            loss, predictions = test(NET,x_test,y_test, criterion)
            final_loss+=loss
        
            for idx, i in enumerate(predictions):
                if i.argmax() == y_test[idx].item():
                    final_accuracy += 1
                
        final_accuracy = (final_accuracy/len(testing_data_set))

        print('The best model accuracy was '+str(best_accuracy) +' found in epoch '+str(best_epoch+1))  
        print('After {} epochs the cross validation accuray of the best model was: {}'.format(epoch+1, final_accuracy*100))
        print('After {} epochs the cross validation loss of the best model was: {}'.format(epoch+1, final_loss))
    else:
        print('The best model accuracy was '+str(best_accuracy) +' found in epoch '+str(best_epoch+1))

    Training_Accuracy_History = Training_Accuracy_History[:epoch_count]
    Validation_Accuracy_History = Validation_Accuracy_History[:epoch_count]
    Training_Loss_History = Training_Loss_History[:epoch_count]
    Validation_Loss_History = Validation_Accuracy_History[:epoch_count]
              
    plt.title('Accuracy')
    plt.plot(Training_Accuracy_History,label = "Training Accuracy")
    plt.plot(Validation_Accuracy_History,label = "Validation Accuracy")
    plt.legend()
    plt.show()

    plt.title('Loss')
    plt.plot(Training_Loss_History/len(training_data_set),label = "Training Loss")
    plt.plot(Validation_Loss_History/len(validation_data_set),label = "Validation Loss")
    plt.legend()
    plt.show()
    
    return Training_Accuracy_History, Validation_Accuracy_History, Training_Loss_History, Validation_Loss_History
 

 
