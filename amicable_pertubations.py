import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import tqdm_notebook

#Package for creating Actionable Counterfactuals

import dice_ml as dml
from dice_ml.explainer_interfaces.dice_pytorch import DicePyTorch
from dice_ml.utils.exception import UserConfigValidationException

# Package for creating Adversarial Examples

import art

#Miscellaneous Utilities 

import numpy as np
import scipy.linalg as la
import pandas as pd
import random

from matplotlib import pyplot as plt

import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def columnize(x):
    return x.reshape((len(x),1))

def atanh(x, eps=1e-6):
    """
    The inverse hyperbolic tangent function, missing in pytorch.
    :param x: a tensor or a Variable
    :param eps: used to enhance numeric stability
    :return: :math:`\\tanh^{-1}{x}`, of the same type as ``x``
    """
    x = x * (1 - eps)
    return 0.5 * torch.log((1.0 + x) / (1.0 - x))

def to_tanh_space(x, box):
    # type: (Union[Variable, torch.FloatTensor], Tuple[float, float]) -> Union[Variable, torch.FloatTensor]
    """
    Convert a batch of tensors to tanh-space. This method complements the
    implementation of the change-of-variable trick in terms of tanh.
    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in tanh-space, of the same dimension;
             the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1,:] - box[0,:]) * 0.5
    _box_plus = (box[1,:] + box[0,:]) * 0.5
    return atanh((x - _box_plus) / _box_mul)

def from_tanh_space(x, box):
    # type: (Union[Variable, torch.FloatTensor], Tuple[float, float]) -> Union[Variable, torch.FloatTensor]
    """
    Convert a batch of tensors from tanh-space to oridinary image space.
    This method complements the implementation of the change-of-variable trick
    in terms of tanh.
    :param x: the batch of tensors, of dimension [B x C x H x W]
    :param box: a tuple of lower bound and upper bound of the box constraint
    :return: the batch of tensors in ordinary image space, of the same
             dimension; the returned tensor is on the same device as ``x``
    """
    _box_mul = (box[1,:] - box[0,:]) * 0.5
    _box_plus = (box[1,:] + box[0,:]) * 0.5
    return torch.tanh(x) * _box_mul + _box_plus

class AP_generator_Continuous():

    def __init__(
    self,
    classifier,
    verifier,
    dist_measure,
    input_size,
    output_size,
    p,
    q,
    wanted_indices,
    unwanted_indices,
    bounds,
    immutable_indices,
    conditioner,
    categorical_feature_indexes,
    bound_penalty = 100,
    categorical_penalty = 1,
    initial_margin = 0.9,
    initial_balance_param = 1,
    max_steps = 6,
    learning_rate = 0.001,
    max_iters = 10000,
    confidence = 0.5
    ):
        self.classifier = classifier
        self.classifier.eval()
        self.verifier = verifier
        self.verifier.eval()
        self.dist_measure = dist_measure
        self.categorical_feature_indexes = categorical_feature_indexes
        self.categorical_penalty = categorical_penalty
        self.bound_penalty = bound_penalty
        self.margin = initial_margin
        if p is not None:
            self.p = p
            print(q)
            if q is not None:
                self.q = q
                self.wanted = wanted_indices
                self.unwanted = unwanted_indices
                self.neutral = np.array([i for i in range(output_size) if (i not in wanted_indices) and  (i not in unwanted_indices)])
                self.target_type = 'full'
            else:
                self.wanted = wanted_indices
                self.neutral = np.array([i for i in range(output_size) if (i not in wanted_indices)])
                self.target_type = 'wanted'
        else:
            self.q = q
            self.unwanted = unwanted_indices
            self.neutral = np.array([i for i in range(output_size) if (i not in unwanted_indices)])
            self.target_type = 'unwanted'
        
        self.bounds = bounds
        self.immutable = torch.tensor(immutable_indices)
        self.mutable = torch.tensor([i for i in range(input_size) if (i not in immutable_indices)])
        self.conditioner = conditioner
        self.default_max_iters = max_iters
        self.default_max_steps = max_steps
        self.lr = learning_rate
        self.output_size = output_size
        self.input_size = input_size
        self.lbda = initial_balance_param
        self.confidence = confidence
        
    def calc_full_PD(self,y,shrinkage=None):
        desirability = torch.sum(y[self.wanted])
        undesirability = torch.sum(y[self.unwanted])
        neutrality = torch.sum(y[self.neutral])
        
        if shrinkage == None:
            shrinkage = self.margin
            
        p = 1-shrinkage*(1-self.p)
        q = shrinkage*self.q
        
        
        q_inv = 1-q
        p_inv = 1-p
        pq_inv = 1 - (p + q)

        distance = torch.tensor(0)
        
        slant_wanted_bound = (1-undesirability)*p/q_inv
        slant_unwanted_bound = (1-desirability)*q/p_inv

        if (desirability >= p) and (undesirability <= q):
            pass
        elif (undesirability > q) and (desirability>=slant_wanted_bound):
            distance = undesirability*torch.log(undesirability/q) + (desirability + neutrality)*torch.log((desirability + neutrality)/q_inv)
        elif (desirability < p) and (undesirability<=slant_unwanted_bound):
            distance = desirability*torch.log(desirability/p) + (undesirability + neutrality)*torch.log((undesirability + neutrality)/p_inv)
        else:
            distance = desirability*torch.log(desirability/p) + undesirability*torch.log(undesirability/q) + neutrality*torch.log(neutrality/pq_inv)
        return distance
    
    def calc_wanted_PD(self,y,shrinkage=None):
        desirability = torch.sum(y[self.wanted])
        neutrality = torch.sum(y[self.neutral])
        
        if shrinkage == None:
            shrinkage = self.margin
            
        p = 1-shrinkage*(1-self.p)
        
        p_inv = 1-p

        distance = torch.tensor(0)

        if desirability >= p:
            pass
        else:
            distance = desirability*torch.log(desirability/p) + neutrality*torch.log(neutrality/p_inv)
        return distance
    
    def calc_unwanted_PD(self,y,shrinkage=None):
        undesirability = torch.sum(y[self.unwanted])
        neutrality = torch.sum(y[self.neutral])
        
        if shrinkage == None:
            shrinkage = self.margin      
                        
        q = shrinkage*self.q
        
        q_inv = 1-q

        distance = torch.tensor(0)

        if undesirability <= q:
            pass
        else:
            distance = undesirability*torch.log(undesirability/q) + neutrality*torch.log(neutrality/q_inv)
        return distance
    
    def calculate_PD(self,input,shrinkage = 1):
        prediction = torch.nn.functional.softmax(self.classifier(torch.tensor(input)),dim = 0)
        output = 0
        if self.target_type == 'full':
            output += self.calc_full_PD(prediction,shrinkage=shrinkage)
        elif self.target_type == 'wanted':
            output += self.calc_wanted_PD(prediction,shrinkage=shrinkage)
        elif self.target_type == 'unwanted':
            output += self.calc_unwanted_PD(prediction,shrinkage=shrinkage)
        else:
            print('Error: target type undefined')
        return output
        
    def suggest_AP(self,x,lbda,shrinkage,verbose = False):
    
        original = torch.tensor(x,dtype= float)
        x_tilde = np.copy(x)
        final_prediction =np.zeros(self.output_size)
        

        suggested_xtilde = to_tanh_space(original,self.bounds)
        suggested_xtilde.requires_grad = True

        
        opt = torch.optim.Adam([suggested_xtilde], lr=self.lr)
            
        for i in range(self.default_max_iters):
            opt.zero_grad()
            loss = torch.tensor(0.)
            prediction = torch.nn.functional.softmax(self.classifier(from_tanh_space(suggested_xtilde, self.bounds)),dim = 0)
            if self.target_type == 'full':
                loss += self.calc_full_PD(prediction,shrinkage=shrinkage)
            elif self.target_type == 'wanted':
                loss += self.calc_wanted_PD(prediction,shrinkage=shrinkage)
            elif self.target_type == 'unwanted':
                loss += self.calc_unwanted_PD(prediction,shrinkage=shrinkage)
            else:
                print('Error: target type undefined')
                
            loss += lbda * self.dist_measure(original,from_tanh_space(suggested_xtilde, self.bounds))
            
            for indices in self.categorical_feature_indexes:
                loss += self.categorical_penalty * torch.pow((torch.sum(from_tanh_space(suggested_xtilde, self.bounds)[indices[0]:indices[1]+1])-1.0),2)
            
            loss.backward(retain_graph=True)

            suggested_xtilde.grad[self.immutable] *= 0
            opt.step()
            
        final_x_tilde = self.conditioner(from_tanh_space(suggested_xtilde, self.bounds))
            
        prediction = torch.nn.functional.softmax(self.classifier(final_x_tilde),dim = 0)

        prob_distance = 0
            
        if self.target_type == 'full':
            prob_distance += self.calc_full_PD(prediction,shrinkage=1)
        elif self.target_type == 'wanted':
            prob_distance += self.calc_wanted_PD(prediction,shrinkage=1)
        elif self.target_type == 'unwanted':
            prob_distance += self.calc_unwanted_PD(prediction,shrinkage=1)
        else:
            print('Error: target type undefined')
            
        inpt_distance = self.dist_measure(original,final_x_tilde)
        
        if verbose:
            print('Final suggestion:')
            print(final_x_tilde)
            print('Changes made:')
            print(final_x_tilde-original)
            print('Estimated proablilities:')
            print(prediction)
            print('Distance from target:')
            print(prob_distance.item())
            print('Cost:')
            print(inpt_distance.item())
            print('')
            unconditioned = from_tanh_space(suggested_xtilde, self.bounds)
            print('Unconditioned suggestion:')
            print(unconditioned)
            print('Changes made:')
            print(unconditioned-original)
            unc_pred = torch.nn.functional.softmax(self.classifier(unconditioned),dim = 0)
            print('Estimated proablilities:')
            print(unc_pred)
            unc_dist = 0
            if self.target_type == 'full':
                unc_dist += self.calc_full_PD(unc_pred,shrinkage=1)
            elif self.target_type == 'wanted':
                unc_dist += self.calc_wanted_PD(unc_pred,shrinkage=1)
            elif self.target_type == 'unwanted':
                unc_dist += self.calc_unwanted_PD(unc_pred,shrinkage=1)
            else:
                print('Error: target type undefined')
            print('Distance from target:')
            print(unc_dist.item())
            print('Cost:')
            print(self.dist_measure(unconditioned,final_x_tilde).item())
        
            
        return final_x_tilde, prediction, prob_distance, inpt_distance
    
    

        
    
    def suggest_AP2(self,x,lbda,shrinkage,verbose = False):
    
        original = torch.tensor(x,dtype= float)
        x_tilde = np.copy(x)
        final_prediction =np.zeros(self.output_size)
        

        suggested_xtilde = original.clone().detach()
        suggested_xtilde.requires_grad = True

        
        opt = torch.optim.Adam([suggested_xtilde], lr=self.lr)

            
        for i in range(self.default_max_iters):
            opt.zero_grad()
            loss = torch.tensor(0.)
            prediction = torch.nn.functional.softmax(self.classifier(suggested_xtilde),dim = 0)
            if self.target_type == 'full':
                loss += self.calc_full_PD(prediction,shrinkage=shrinkage)
            elif self.target_type == 'wanted':
                loss += self.calc_wanted_PD(prediction,shrinkage=shrinkage)
            elif self.target_type == 'unwanted':
                loss += self.calc_unwanted_PD(prediction,shrinkage=shrinkage)
            else:
                print('Error: target type undefined')
                
            loss += lbda * self.dist_measure(original,suggested_xtilde)
            
            for indices in self.categorical_feature_indexes:
                loss += self.categorical_penalty * torch.pow(torch.sum(suggested_xtilde[indices[0]:indices[1]+1])-1.0,2)
                
            for feature in range(self.input_size):
                loss += self.bound_penalty * torch.nn.functional.relu(suggested_xtilde[feature]-self.bounds[1,feature])
                loss += self.bound_penalty * torch.nn.functional.relu(self.bounds[0,feature]-suggested_xtilde[feature])
            
            loss.backward(retain_graph=True)

            suggested_xtilde.grad[self.immutable] *= 0
            opt.step()

            
        final_x_tilde = self.conditioner(suggested_xtilde)
            
        prediction = torch.nn.functional.softmax(self.classifier(final_x_tilde),dim = 0)

        prob_distance = 0
            
        if self.target_type == 'full':
            prob_distance += self.calc_full_PD(prediction,shrinkage=1)
        elif self.target_type == 'wanted':
            prob_distance += self.calc_wanted_PD(prediction,shrinkage=1)
        elif self.target_type == 'unwanted':
            prob_distance += self.calc_unwanted_PD(prediction,shrinkage=1)
        else:
            print('Error: target type undefined')
            
        inpt_distance = self.dist_measure(original,final_x_tilde)
        
        if verbose:
            print('Final suggestion:')
            print(final_x_tilde)
            print('Changes made:')
            print(final_x_tilde-original)
            print('Estimated proablilities:')
            print(prediction)
            print('Distance from target:')
            print(prob_distance.item())
            print('Cost:')
            print(inpt_distance.item())
            print('')
            unconditioned = suggested_xtilde
            print('Unconditioned suggestion:')
            print(unconditioned)
            print('Changes made:')
            print(unconditioned-original)
            unc_pred = torch.nn.functional.softmax(self.classifier(unconditioned),dim = 0)
            print('Estimated proablilities:')
            print(unc_pred)
            unc_dist = 0
            if self.target_type == 'full':
                unc_dist += self.calc_full_PD(unc_pred,shrinkage=1)
            elif self.target_type == 'wanted':
                unc_dist += self.calc_wanted_PD(unc_pred,shrinkage=1)
            elif self.target_type == 'unwanted':
                unc_dist += self.calc_unwanted_PD(unc_pred,shrinkage=1)
            else:
                print('Error: target type undefined')
            print('Distance from target:')
            print(unc_dist.item())
            print('Cost:')
            print(self.dist_measure(unconditioned,final_x_tilde).item())
            
            print('One hot variance from summing to one:')
            for indices in self.categorical_feature_indexes:
                print(torch.sum(suggested_xtilde[indices[0]:indices[1]+1])-1.0)
        
            
        return final_x_tilde, prediction, prob_distance, inpt_distance
    
    def suggest_AP_path(self,x,lbda,shrinkage,iterations,output_steps):
        
        
        path_length = iterations//output_steps
        original = torch.tensor(x,dtype= float)
        x_tilde = np.copy(x)
        ap_path = np.zeros((path_length ,self.input_size))
        prediction_probs = np.zeros((path_length ,self.output_size))
        deltas = np.zeros(path_length)
        epsilons = np.zeros(path_length)
        

        suggested_xtilde = original.clone().detach()
        suggested_xtilde.requires_grad = True

        
        opt = torch.optim.Adam([suggested_xtilde], lr=self.lr)

            
        for i in range(iterations+1):
            opt.zero_grad()
            loss = torch.tensor(0.)
            prediction = torch.nn.functional.softmax(self.classifier(suggested_xtilde),dim = 0)
            if self.target_type == 'full':
                loss += self.calc_full_PD(prediction,shrinkage=shrinkage)
            elif self.target_type == 'wanted':
                loss += self.calc_wanted_PD(prediction,shrinkage=shrinkage)
            elif self.target_type == 'unwanted':
                loss += self.calc_unwanted_PD(prediction,shrinkage=shrinkage)
            else:
                print('Error: target type undefined')
                
            loss += lbda * self.dist_measure(original,suggested_xtilde)
            
            for indices in self.categorical_feature_indexes:
                loss += self.categorical_penalty * torch.pow(torch.sum(suggested_xtilde[indices[0]:indices[1]+1])-1.0,2)
                
            for feature in range(self.input_size):
                loss += self.bound_penalty * torch.nn.functional.relu(suggested_xtilde[feature]-self.bounds[1,feature])
                loss += self.bound_penalty * torch.nn.functional.relu(self.bounds[0,feature]-suggested_xtilde[feature])
            
            loss.backward(retain_graph=True)

            suggested_xtilde.grad[self.immutable] *= 0
            opt.step()
            
            if (i%output_steps == 0) and (1 != 0):
                this_x_tilde = self.conditioner(suggested_xtilde)
                this_prediction = torch.nn.functional.softmax(self.classifier(this_x_tilde),dim = 0)
                this_prob_distance = 0
                if self.target_type == 'full':
                    this_prob_distance += self.calc_full_PD(this_prediction,shrinkage=1)
                elif self.target_type == 'wanted':
                    this_prob_distance += self.calc_wanted_PD(this_prediction,shrinkage=1)
                elif self.target_type == 'unwanted':
                    this_prob_distance += self.calc_unwanted_PD(this_prediction,shrinkage=1)
                else:
                    print('Error: target type undefined')
                place = (i//output_steps)-1
                this_input_distance = self.dist_measure(original,this_x_tilde)
                ap_path[place] = this_x_tilde.detach().numpy()
                prediction_probs[place] = this_prediction.detach().numpy()
                deltas[place] = this_prob_distance
                epsilons[place] = this_input_distance
            

            
        
        
        
        
            
        return ap_path, prediction_probs, deltas, epsilons
    
    def create_budget_optimal_AP(self,x,budget,confidence = None, lbda = None):
        
        if confidence == None:
            confidence = self.confidence
            
        if lbda == None:
            lbda = self.lbda
            
        
        
        for i in range(self.default_max_steps):
            
            print('Confidence step: '+str(i)+' with margin '+str(confidence))
            
            has_grown = False
            valid_found = False
            
            current_lbda = lbda

            
            for j in range(self.default_max_steps):
                print('Budget step: '+str(j)+' with balance parameter: '+str(current_lbda))
                suggested_AP, estimated_probs, prob_distance, inpt_distance = self.suggest_AP(x,current_lbda,confidence)
                print('Suggested Perturbation')
                print(suggested_AP)
                print('Estimated Class Probabilities')
                print(estimated_probs)
                if inpt_distance > budget:
                    print('Exceeded Budget')
                    if not valid_found:
                        current_lbda *= 2
                        has_grown = True
                    else:
                        break
                else:
                    valid_found = True
                    print('Within Budget')
                    final_x_tilde, final_probs, final_distance, final_cost = suggested_AP, estimated_probs, prob_distance, inpt_distance
                    if has_grown:
                        break 
                    else:
                        current_lbda *= 0.5
            
            estimated_confidence = self.verifier(torch.cat((final_x_tilde,torch.tensor(x))))[1]
            
            if estimated_confidence >= confidence:
                           
                return final_x_tilde, final_probs, final_distance, final_cost
            
            else:
                confidence = confidence/2
                
        print('Valid Amicable Perturbation NOT found')
        
        return final_x_tilde, final_probs, final_distance, final_cost
    
    def create_tolerance_optimal_AP(self,x,tolerance,confidence = None, lbda = None):
        
        if confidence == None:
            confidence = self.confidence
            
        if lbda == None:
            lbda = self.lbda
            
        
        
        for i in range(self.default_max_steps):
            
            print('Confidence step: '+str(i)+' with margin '+str(confidence))
            
            has_shrunk = False
            valid_found = False
            
            current_lbda = lbda

            
            for j in range(self.default_max_steps):
                print('Budget step: '+str(j)+' with balance parameter: '+str(current_lbda))
                suggested_AP, estimated_probs, prob_distance, inpt_distance = self.suggest_AP(x,current_lbda,confidence)
                print('Suggested Perturbation')
                print(suggested_AP)
                print('Estimated Class Probabilities')
                print(estimated_probs)
                if prob_distance > tolerance:
                    print('Outside Tolerance')
                    if not valid_found:
                        current_lbda *= 0.5
                        has_shrunk = True
                    else:
                        break
                else:
                    valid_found = True
                    print('Within Tolerance')
                    final_x_tilde, final_probs, final_distance, final_cost = suggested_AP, estimated_probs, prob_distance, inpt_distance
                    if has_shrunk:
                        break 
                    else:
                        current_lbda *= 2
            
            estimated_confidence = self.verifier(torch.cat((final_x_tilde,torch.tensor(x))))[1]
            
            if estimated_confidence >= confidence:
                           
                return final_x_tilde, final_probs, final_distance, final_cost
            
            else:
                confidence = confidence/2
                
        print('Valid Amicable Perturbation NOT found')
        
        return final_x_tilde, final_probs, final_distance, final_cost
                
                
def create_pareto_graph(creator, classifier, verifier, x, begin_lmbd, end_lmbd, lmbd_number, iterations, print_steps, discrepancy_cutoffs, discrepancy_names, intersections):
    lambdas = np.linspace(begin_lmbd, end_lmbd, lmbd_number)
    
    original_probs = torch.nn.functional.softmax(classifier(torch.tensor(x))).detach().cpu().numpy()
    
    aps = []
    probs = []
    deltas = []
    epsilons = []
    
    for i in lambdas:
        print('Using lambda '+str(i))
        amicable,probability,delta,epsilon = creator.suggest_AP_path(torch.tensor(x),i,1,iterations,print_steps)
        aps.append(amicable)
        probs.append(probability)
        deltas.append(delta)
        epsilons.append(epsilon)
        
    aps = np.concatenate(aps) 
    probs = np.concatenate(probs) 
    deltas = np.concatenate(deltas) 
    epsilons = np.concatenate(epsilons) 
    
    aps,locations = np.unique(aps,axis = 0,return_index = True)
    probs = probs[locations]
    deltas = deltas[locations]
    epsilons = epsilons[locations]
    #print(aps.shape)
    #print(probs.shape)
    
    verified_aps = np.zeros(aps.shape[0])
    discrepancies = np.zeros(aps.shape[0])
    #print(discrepancies.shape)

    for idx in range(aps.shape[0]):
        verification = torch.nn.functional.softmax(verifier(torch.tensor(np.concatenate((x,aps[idx]))).to(device).double()))
        verified_aps[idx] += verification[1]
        discrepancies[idx] += np.abs(np.sum(probs[idx]*original_probs)-verification[0].cpu().detach().numpy())
        
    original_delta = creator.calc_wanted_PD(torch.nn.functional.softmax(classifier(torch.tensor(x))),shrinkage=1).item()
    
    
    for j in intersections: 
        verified = []
        rejected = []
        for i,delta in enumerate(deltas):
            if verified_aps[i] >= j * (1 -(delta/original_delta)):
                verified.append(i)
            else:
                rejected.append(i)
        
        plt.scatter(epsilons[verified],deltas[verified],label = 'verified')
        plt.scatter(epsilons[rejected],deltas[rejected],label = 'rejected')
        plt.legend()
        plt.title('Using Max Cutoff '+str(j))
        plt.xlabel('epsilon')
        plt.ylabel('delta')
        plt.show()
    
    for j in range(len(discrepancy_cutoffs)):
        verified = []
        rejected = []
        for i in range(len(deltas)):
            if discrepancies[i] >= discrepancy_cutoffs[j] :
                verified.append(i)
            else:
                rejected.append(i)
        
        plt.scatter(epsilons[verified],deltas[verified],label = 'verified')
        plt.scatter(epsilons[rejected],deltas[rejected],label = 'rejected')
        plt.legend()
        plt.title('Discrepancy Cutoff '+discrepancy_names[j])
        plt.xlabel('epsilon')
        plt.ylabel('delta')
        plt.show()
        
        
    return aps,probs,deltas,epsilons,verified_aps,discrepancies
	
def test_individual(individual,classifier, soft_classifier, verifier,AP_creator,counterfactual_creator,names,variable_names,immutables,begin_lambda,end_lambda,lambda_count,iterations,conditioner,distance,interpretor,change_interprator,counterfactual_count,intersection = 0.51,verbose = False):
    aps,probs,deltas,epsilons,ver,discrep = create_pareto_graph(AP_creator, classifier, verifier, individual, begin_lambda, end_lambda, lambda_count, iterations, 1, np.array([]), np.array([]), np.array([]))

    
    delta_original = AP_creator.calculate_PD(individual).detach().numpy()
    tense_ind = torch.tensor(individual).float()
    panda_ind = pd.DataFrame(np.array([individual]),columns = names)
    
    original_probs = soft_classifier(tense_ind).detach().cpu().numpy()
    
    single_found = True
    diverse_found = True
    
    try:
        single_CF = counterfactual_creator.generate_counterfactuals(panda_ind, total_CFs=1, desired_class="opposite",features_to_vary=variable_names)
        single_CF = single_CF.cf_examples_list[0].final_cfs_df.to_numpy()[0,:-1]
        single_CF[immutables] = individual[immutables]
        single_CF = conditioner(torch.tensor(single_CF)).float()
        single_epsilon = distance(tense_ind.double(),single_CF.double())
        single_delta = AP_creator.calculate_PD(single_CF.double()).detach().numpy()
        single_probs = soft_classifier(single_CF).detach().cpu().numpy()
        single_ver = torch.nn.functional.softmax(verifier(torch.cat((tense_ind,single_CF),dim = 0).to(device).double()),dim = 0).detach().cpu().numpy()
        single_discrep = np.abs(np.sum(single_probs*original_probs)-single_ver[0])
        single_ver = single_ver[1]
    except UserConfigValidationException:
        print('Unable to find single counterfactual.')
        single_found = False
    
    
    try:
        diverse_CF = counterfactual_creator.generate_counterfactuals(panda_ind, total_CFs=counterfactual_count, desired_class="opposite",features_to_vary=variable_names)
        diverse_CF = diverse_CF.cf_examples_list[0].final_cfs_df.to_numpy()[:,:-1]
        total_cfs = diverse_CF.shape[0]
    
        diverse = torch.empty(diverse_CF.shape)
        for i,cf in enumerate(diverse_CF):
            diverse[i] = conditioner(torch.tensor(cf)).float()
            diverse[i][immutables] = torch.tensor(individual[immutables]).float()
            
        diverse_epsilons = np.empty(total_cfs)
        for i in range(total_cfs):
            diverse_epsilons[i]  = distance(tense_ind.double(),diverse[i].double())
            
        diverse_deltas = np.empty(total_cfs)
        for i in range(total_cfs):
            diverse_deltas[i]  = AP_creator.calculate_PD(diverse[i].double()).detach().numpy()
            
        diverse_probs = soft_classifier(diverse).detach().cpu().numpy()        
        diverse_ver = torch.nn.functional.softmax(verifier(torch.cat((tense_ind.repeat(total_cfs,1),diverse),dim = 1).to(device).double()),dim = 1).detach().cpu().numpy()

        diverse_discrep = np.abs(np.sum(diverse_probs*np.tile(original_probs,[total_cfs,1]),axis = 1)-diverse_ver[:,0])
        diverse_ver = diverse_ver[:,1] 
        
    except UserConfigValidationException:
        print('Unable to find Diverse counterfactuals.')
        diverse_found = False

    
    con = ver-intersection*(1-(deltas/delta_original))
    if single_found:
        single_con = single_ver-intersection*((1-single_delta/delta_original))
    if diverse_found:
        diverse_con = diverse_ver-intersection*((1-diverse_deltas/delta_original))

        
    if single_found:
        if diverse_found:
            all_ver = np.concatenate((ver,np.array([single_ver]),diverse_ver))
            all_discrep = np.concatenate((discrep,np.array([single_discrep]),diverse_discrep))
            all_con = np.concatenate((con,np.array([single_con]),diverse_con))
        else:
            all_ver = np.concatenate((ver,np.array([single_ver])))
            all_discrep = np.concatenate((discrep,np.array([single_discrep])))
            all_con = np.concatenate((con,np.array([single_con])))
    else:
        if diverse_found:
            all_ver = np.concatenate((ver,diverse_ver))
            all_discrep = np.concatenate((discrep,diverse_discrep))
            all_con = np.concatenate((con,diverse_con))
        else:
            all_ver = ver
            all_discrep = discrep
            all_con = con

    ver_max = all_ver.max()
    ver_min = all_ver.min()
    
    discrep_max = all_discrep.max()
    discrep_min = all_discrep.min()

    con_max = all_con.max()
    con_min = all_con.min()
    
    if verbose:
        if interpretor is not None:
            interpretor(individual)
        
    plt.scatter(epsilons,deltas,marker = 'o',label = 'Amicable Perturbations')
    if single_found:
        plt.scatter(single_epsilon,single_delta,marker = '*',label = 'Original Counterfactual')
    if diverse_found:
        plt.scatter(diverse_epsilons,diverse_deltas,marker = 's',label = 'DICE')
    plt.title('All Solutions')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\delta$')
    plt.legend()
    plt.show()
    
    plt.scatter(epsilons,deltas,c=discrep,marker = 'o',label = 'Amicable Perturbations',cmap = 'cividis')
    plt.clim(discrep_min,discrep_max)
    if single_found:
        plt.scatter(single_epsilon,single_delta,c=single_discrep,marker = '*',label = 'Original Counterfactual',cmap = 'cividis')
        plt.clim(discrep_min,discrep_max)
    if diverse_found:
        plt.scatter(diverse_epsilons,diverse_deltas,c=diverse_discrep,marker = 's',label = 'DICE',cmap = 'cividis')
        plt.clim(discrep_min,discrep_max)
    plt.title('Darker Means Smaller Discrepancy Value')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\delta$')
    plt.legend()
    plt.colorbar()
    plt.show()
    
    plt.scatter(epsilons,deltas,c=con,marker = 'o',label = 'Amicable Perturbations',cmap = 'cividis_r')
    plt.clim(con_min,con_max)
    if single_found:
        plt.scatter(single_epsilon,single_delta,c=single_con,marker = '*',label = 'Original Counterfactual',cmap = 'cividis_r')
        plt.clim(con_min,con_max)
    if diverse_found:
        plt.scatter(diverse_epsilons,diverse_deltas,c=diverse_con,marker = 's',label = 'DICE',cmap = 'cividis_r')
        plt.clim(con_min,con_max)
    plt.title('Darker Means  Higher Verifier Confidence')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\delta$')
    plt.legend()
    plt.colorbar()
    plt.show()
    
    
    plt.scatter(epsilons,deltas,c=ver,marker = 'o',label = 'Amicable Perturbations',cmap = 'cividis_r')
    plt.clim(ver_min,ver_max)
    if single_found:
        plt.scatter(single_epsilon,single_delta,c=single_ver,marker = '*',label = 'Original Counterfactual',cmap = 'cividis_r')
        plt.clim(ver_min,ver_max)
    if diverse_found:
        plt.scatter(diverse_epsilons,diverse_deltas,c=diverse_ver,marker = 's',label = 'DICE',cmap = 'cividis_r')
        plt.clim(ver_min,ver_max)
    plt.title('Darker Means Higher Verifier Value')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel(r'$\delta$')
    plt.legend()
    plt.colorbar()
    plt.show()
    
    if verbose:

        ordered_discrepancies = np.argsort(discrep)
        print('')
        print('Amicable Perturbations ')
        print('')
        print('')
        for i in ordered_discrepancies:
            print('Distance to target: '+str(deltas[i]))
            print('Cost :'+str(epsilons[i]))
            print('Dicrepancy value : '+str(discrep[i]))
            print('Verification Confidence: '+str(con[i]))
            print('Straight verification value: '+str(ver[i]))           
            change_interprator(individual,aps[i])
            print('')
                  
        if single_found:
            print('')
            print('Single Counterfactual')
            print('')
            print('')
            print('Distance to target: '+str(single_delta))
            print('Cost :'+str(single_epsilon))
            print('Dicrepancy value : '+str(single_discrep))
            print('Verification Confidence: '+str(single_con))
            print('Straight verification value: '+str(single_ver))           
            change_interprator(individual,single_CF.numpy())
            print('')
                  
        if diverse_found:
            print('')
            print('Diverse Counterfactuals')
            print('')
            print('')
            ordered_discrepancies = np.argsort(diverse_discrep)
            for i in ordered_discrepancies:
                print('Distance to target: '+str(diverse_deltas[i]))
                print('Cost :'+str(diverse_epsilons[i]))
                print('Dicrepancy value : '+str(diverse_discrep[i]))
                print('Verification Confidence: '+str(diverse_con[i]))
                print('Straight verification value: '+str(diverse_ver[i]))           
                change_interprator(individual,diverse[i].numpy())
                print('')
                
    #print(aps)
    #print(probs)
    #print(columnize(deltas))
    #print(columnize(epsilons))
    #print(columnize(ver))
    #print(columnize(discrep))
    
    if single_found:
        if diverse_found:
            return np.concatenate((aps,probs,columnize(deltas),columnize(epsilons),columnize(ver),columnize(discrep)),axis = 1),np.concatenate((single_CF.numpy(),single_probs,np.array([single_delta]),np.array([single_epsilon]),np.array([single_ver]),np.array([single_discrep]))),np.concatenate((diverse.numpy(),diverse_probs,columnize(diverse_deltas),columnize(diverse_epsilons),columnize(diverse_ver),columnize(diverse_discrep)),axis = 1)
        else:
            return np.concatenate((aps,probs,columnize(deltas),columnize(epsilons),columnize(ver),columnize(discrep)),axis = 1),np.concatenate((single_CF.numpy(),single_probs,single_delta,single_epsilon,single_ver,single_discrep)),None
    else:
        if diverse_found:
            return np.concatenate((aps,probs,columnize(deltas),columnize(epsilons),columnize(ver),columnize(discrep)),axis = 1),None,np.concatenate((diverse.numpy(),diverse_probs,columnize(diverse_deltas),columnize(diverse_epsilons),columnize(diverse_ver),columnize(diverse_discrep)),axis = 1)
        else:
            return np.concatenate((aps,probs,columnize(deltas),columnize(epsilons),columnize(ver),columnize(discrep)),axis = 1),None,None
			
def save_results(inputs,class_number,attribute_names,outfile,classifier,soft_classifier,verifier,AP_creator,counterfactual_creator,variable_names,immutables,begin_lambda,end_lambda,lambda_count,iterations,conditioner,distance,interpretor,change_interprator,counterfactual_count,intersection = 0.51,verbose = False):
    names = ['individual']
    names += list(attribute_names)
    names += ['class '+str(i) for i in range(class_number)]
    names += ['delta','epsilon','verification','discrepancy']
    #print(names)
    #print(list(attribute_names))
    
    
    aps = np.zeros((1,len(names)))
    cfs = np.zeros((1,len(names)))
    dice = np.zeros((1,len(names)))
    count = 0
    for i,individual in enumerate(inputs):
        print(count)
        count += 1
        amicable, counterfactual, diverse = test_individual(individual,classifier, soft_classifier, verifier,AP_creator,counterfactual_creator,attribute_names,variable_names,immutables,begin_lambda,end_lambda,lambda_count,iterations,conditioner,distance,interpretor,change_interprator,counterfactual_count)
        
        amicable = np.concatenate((i*np.ones((len(amicable),1)),amicable),axis = 1)
        aps = np.concatenate((aps,amicable),axis = 0)
        
        if counterfactual is not None:
            counterfactual = np.concatenate((np.array([i]),counterfactual)).reshape((1,len(names)))
            cfs = np.concatenate((cfs,counterfactual),axis = 0)
        if diverse is not None:
            diverse = np.concatenate((i*np.ones((len(diverse),1)),diverse),axis = 1)
            dice = np.concatenate((dice,diverse),axis=0)
            
        aps_df = panda_ind = pd.DataFrame(aps[1:],columns = names)
        cfs_df = panda_ind = pd.DataFrame(cfs[1:],columns = names)
        dice_df = panda_ind = pd.DataFrame(dice[1:],columns = names)
            
            
        aps_df.to_csv('amicable_pertubations_'+outfile, index=False)
        cfs_df.to_csv('counterfactuals_'+outfile, index=False)
        dice_df.to_csv('Diverse_counterfactuals_'+outfile, index=False)
        
def save_original_deltas(individuals,ap_creator, save_file):
    output = np.empty((individuals.shape[0],2))
    output[:,0] = np.arange(individuals.shape[0])
    for index,individual in enumerate(individuals):
        #print(ap_creator.calculate_PD(individual).detach().cpu().numpy())
        output[index,1] = ap_creator.calculate_PD(individual).detach().cpu().numpy()
    my_dataframe =  pd.DataFrame(output,columns = ['individual','delta'])
    my_dataframe.to_csv(save_file, index=False)      
    
def save_cwl2_attacks(individuals,targets,attacker,confidences,ap_generator,classifier,soft_classifier,verifier,distance_measure,attribute_names,outfile_name):
    
    individual_number = individuals.shape[0]
    confidence_number = len(confidences)
    attribute_number = individuals.shape[1]
    #print(attribute_number)
    
    outfile= np.zeros((individual_number * confidence_number,attribute_number+7))
    for i in range(confidence_number):
        outfile[i*individual_number:(i+1)*individual_number,0] = np.arange(individual_number)
    
    for i,con in enumerate(confidences):
        classifier.to(device)
        attacker.confidence = con
        attacked = attacker.generate(individuals.astype(np.float32),labels)
        outfile[i*individual_number:(i+1)*individual_number, 1:1+attribute_number] = attacked
        
        original_probs = soft_classifier(torch.tensor(individuals).float()).detach().cpu().numpy()
        probs = soft_classifier(torch.tensor(attacked)).detach().cpu().numpy()
        outfile[i*individual_number:(i+1)*individual_number, 1+attribute_number:3+attribute_number] = probs
        
        classifier.cpu()
        for j,person in enumerate(attacked):
            outfile[i*individual_number+j, 3+attribute_number] = ap_generator.calculate_PD(person).detach().cpu().numpy()
            outfile[i*individual_number+j, 4+attribute_number] = distance_measure(torch.tensor(individuals[j]).double(),torch.tensor(attacked[j]).double())
        
        verification = torch.nn.functional.softmax(verifier(torch.cat((torch.tensor(individuals),torch.tensor(attacked)),dim = 1).to(device).double()),dim = 0).detach().cpu().numpy()
        
        dicrepancy = np.abs(np.sum(probs*original_probs,axis = 1) -  verification[:,0])
        
        outfile[i*individual_number:(i+1)*individual_number, 5+attribute_number] = verification[:,1]
        
        outfile[i*individual_number:(i+1)*individual_number, 6+attribute_number] = dicrepancy
        
        
    names = ['individual']
    names += list(attribute_names)
    names += ['class '+str(i) for i in range(2)]
    names += ['delta','epsilon','verification','discrepancy']
        
        
    my_dataframe = pd.DataFrame(outfile,columns = names)
            
    #print(outfile)       
    my_dataframe.to_csv(outfile_name, index=False)    
        
def get_success_data(ap_dataframe, cf_dataframe, dice_dataframe, cw_dataframe, original_deltas, cost_cut_offs, distance_cut_offs):
    ap_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    cf_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    dice_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    cw_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    
    ap_verified_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    cf_verified_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    dice_verified_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    cw_verified_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    
    total = int(ap_dataframe['individual'].max())
    
    original_deltas_array = original_deltas.to_numpy()[:,1]
    

    ap_array = ap_dataframe[['individual','delta','epsilon']].to_numpy()
    cf_array = cf_dataframe[['individual','delta','epsilon']].to_numpy()
    dice_array = dice_dataframe[['individual','delta','epsilon']].to_numpy()
    cw_array = cw_dataframe[['individual','delta','epsilon']].to_numpy()
    
    for j,cost_cut in enumerate(cost_cut_offs):
        #print(j)
        for i,distance_cut in enumerate(distance_cut_offs):
            #print(i)
            for individual in range(total):
                
                current_locations = np.where(ap_array[:,0]==individual)[0]
                #print(current_locations)
                current_delta = ap_array[current_locations[0],1]
                #print(current_delta)
                
                ap_in_bounds = (ap_array[:,0]==individual)*(ap_array[:,1]<=distance_cut)*(ap_array[:,2]<=cost_cut)
                if np.any(ap_in_bounds):
                    ap_success[i,j] += 1
                          
                cf_in_bounds = (cf_array[:,0]==individual)*(cf_array[:,1]<=distance_cut)*(cf_array[:,2]<=cost_cut)
                if np.any(cf_in_bounds):
                    cf_success[i,j] += 1

                          
                dice_in_bounds = (dice_array[:,0]==individual)*(dice_array[:,1]<=distance_cut)*(dice_array[:,2]<=cost_cut)
                if np.any(dice_in_bounds):
                    dice_success[i,j] += 1

                        
                cw_in_bounds = (cw_array[:,0]==individual)*(cw_array[:,1]<=distance_cut)*(cw_array[:,2]<=cost_cut)
                if np.any(cw_in_bounds):
                    cw_success[i,j] += 1

                

                    
    ap_success/=  total
    cf_success/=  total
    dice_success/=  total
    cw_success/=  int(cw_dataframe['individual'].max())
                          

                          
    barWidth = 0.2
    
    return ap_success, cf_success, dice_success, cw_success


def get_verified_success_data(ap_dataframe, cf_dataframe, dice_dataframe, cw_dataframe, original_deltas, cost_cut_offs, distance_cut_offs,verification_constant):
    ap_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    cf_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    dice_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    cw_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    
    ap_verified_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    cf_verified_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    dice_verified_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    cw_verified_success = np.zeros((len(distance_cut_offs),len(cost_cut_offs)))
    
    total = int(ap_dataframe['individual'].max())
    
    original_deltas_array = original_deltas.to_numpy()[:,1]
    

    ap_array = ap_dataframe[['individual','delta','epsilon','discrepancy']].to_numpy()
    cf_array = cf_dataframe[['individual','delta','epsilon','discrepancy']].to_numpy()
    dice_array = dice_dataframe[['individual','delta','epsilon','discrepancy']].to_numpy()
    cw_array = cw_dataframe[['individual','delta','epsilon','discrepancy']].to_numpy()
    
    for j,cost_cut in enumerate(cost_cut_offs):
        #print(j)
        for i,distance_cut in enumerate(distance_cut_offs):
            #print(i)
            for individual in range(total):
                
                current_locations = np.where(ap_array[:,0]==individual)[0]
                #print(current_locations)
                current_delta = ap_array[current_locations[0],1]
                #print(current_delta)
                
                ap_in_bounds = (ap_array[:,0]==individual)*(ap_array[:,1]<=distance_cut)*(ap_array[:,2]<=cost_cut)
                if np.any(ap_in_bounds*(ap_array[:,3]<=verification_constant)):
                    ap_success[i,j] += 1
                          
                cf_in_bounds = (cf_array[:,0]==individual)*(cf_array[:,1]<=distance_cut)*(cf_array[:,2]<=cost_cut)
                if np.any(cf_in_bounds*(cf_array[:,3]<=verification_constant)):
                    cf_success[i,j] += 1

                          
                dice_in_bounds = (dice_array[:,0]==individual)*(dice_array[:,1]<=distance_cut)*(dice_array[:,2]<=cost_cut)
                if np.any(dice_in_bounds*(dice_array[:,3]<=verification_constant)):
                    dice_success[i,j] += 1

                        
                cw_in_bounds = (cw_array[:,0]==individual)*(cw_array[:,1]<=distance_cut)*(cw_array[:,2]<=cost_cut)
                if np.any(cw_in_bounds*(cw_array[:,3]<=verification_constant)):
                    cw_success[i,j] += 1

              

                    
    ap_success/=  total
    cf_success/=  total
    dice_success/=  total
    cw_success/=  int(cw_dataframe['individual'].max())
                          

                          
    barWidth = 0.2
    
    return ap_success, cf_success, dice_success, cw_success
    
        
    