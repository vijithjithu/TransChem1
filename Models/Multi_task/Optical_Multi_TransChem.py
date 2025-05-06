import pandas as pd
import numpy as np
import yaml
import torch
import json
import torch.nn as nn
from copy import deepcopy
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torchmetrics import R2Score
from transformers import RobertaModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from newFeats_multi_dataset import MultiTask_Dataset, MultiTask_DataAugmentation

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

"""Adapted from RobertaTokenizer"""
from PolymerSmilesTokenization import PolymerSmilesTokenizer

np.random.seed(seed=1)


def roberta_base_AdamW_LLRD(model, lr, weight_decay):

    # To be passed to the optimizer (only parameters of the layers you want to update).
    opt_parameters = []
    named_parameters = list(model.named_parameters())
    print("number of named parameters =", len(named_parameters))

    # According to AAAMLP book by A. Thakur, we generally do not use any decay
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # === Pooler ======================================================

    params_0 = [p for n, p in named_parameters if "pooler" in n 
                and any(nd in n for nd in no_decay)]
    print("params in pooler without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if "pooler" in n 
                and not any(nd in n for nd in no_decay)]
    print("params in pooler with decay =", len(params_1))

    head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(head_params)

    head_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(head_params)

    print("pooler lr =", lr)

    # === Regressors ======================================================
    for head_name in ["EPS_FCNN", "Nc_FCNN"]:
        params_0 = [p for n, p in named_parameters if head_name in n and any(nd in n for nd in no_decay)]
        print(f"params in {head_name} without decay =", len(params_0))
        params_1 = [p for n, p in named_parameters if head_name in n and not any(nd in n for nd in no_decay)]
        print(f"params in {head_name} with decay =", len(params_1))

        head_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(head_params)

        head_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(head_params)

        print(f"{head_name} lr =", lr)

    # === Hidden layers ==========================================================

    for layer in range(5, -1, -1):
        params_0 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} without decay =", len(params_0))
        params_1 = [p for n, p in named_parameters if f"encoder.layer.{layer}." in n
                    and not any(nd in n for nd in no_decay)]
        print(f"params in hidden layer {layer} with decay =", len(params_1))

        layer_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
        opt_parameters.append(layer_params)

        layer_params = {"params": params_1,
                        "lr": lr, "weight_decay": weight_decay}
        opt_parameters.append(layer_params)

        print("hidden layer", layer, "lr =", lr)

        lr *= 0.9

    # === Embeddings layer ==========================================================

    params_0 = [p for n, p in named_parameters if "embeddings" in n
                and any(nd in n for nd in no_decay)]
    print("params in embeddings layer without decay =", len(params_0))
    params_1 = [p for n, p in named_parameters if "embeddings" in n
                and not any(nd in n for nd in no_decay)]
    print("params in embeddings layer with decay =", len(params_1))

    embed_params = {"params": params_0, "lr": lr, "weight_decay": 0.0}
    opt_parameters.append(embed_params)

    embed_params = {"params": params_1, "lr": lr, "weight_decay": weight_decay}
    opt_parameters.append(embed_params)
    print("embedding layer lr =", lr)

    return AdamW(opt_parameters, lr=lr, no_deprecation_warning=True)

class MaskedLoss(nn.Module):
    def __init__(self):
        super(MaskedLoss, self).__init__()
        
    def forward(self, outputs, props, loss_mask, properties):
        loss = {}
        for i,property in enumerate(properties):
            loss[property] = torch.sum(((outputs[:,i]-props[:,i])*loss_mask[:,i])**2.0)  / (1 if torch.sum(loss_mask[:,i])==0 else 
                                                                                            torch.sum(loss_mask[:,i]))
        return loss
        
"""MT Model"""
class MultiTask(nn.Module):
    def __init__(self, drop_rate=0.1):
        super(MultiTask, self).__init__()
        self.PretrainedModel = deepcopy(PretrainedModel)
        self.PretrainedModel.resize_token_embeddings(len(tokenizer))
        self.embedding_transform = nn.Linear(self.PretrainedModel.config.hidden_size, 32)

        self.EPS_FCNN = nn.Sequential(
            nn.Linear(32+40, 32),
            nn.Dropout(0.3),
            nn.SiLU(),
            nn.Linear(32,1)
        )
        self.Nc_FCNN = nn.Sequential(
            nn.Linear(32+40, 32),
            nn.Dropout(0.3),
            nn.SiLU(),
            nn.Linear(32,1)
        )


    def forward(self, input_ids, attention_mask,new_feats):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_size]

        transformed_logits = self.embedding_transform(logits)  # Shape: [batch_size, 32]

        combined = torch.cat([transformed_logits, new_feats], dim=1)  # Shape: [batch_size, 32 + 73]
        EPS_FCNN_output = self.EPS_FCNN(combined)
        Nc_FCNN_output = self.Nc_FCNN(combined)

        
        final_output = torch.cat([EPS_FCNN_output,Nc_FCNN_output], dim=1)

        return final_output 

def r2score(y_pred, y_actual, mask):
    bool_mask = mask.bool()
    masked_y_pred = y_pred[bool_mask]
    masked_y_actual = y_actual[bool_mask]

    ssr = torch.sum((masked_y_pred - masked_y_actual)**2)
    sst = torch.sum((masked_y_actual - torch.mean(masked_y_actual))**2)
    r2 = 1 - (ssr / sst)

    return r2

def train(model, optimizer, scheduler, Loss, train_dataloader, device, properties):

    model.train()

    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        props = batch["props"].to(device)
        new_feats = batch["new_feats"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids,attention_mask,new_feats).float()
        total_loss = 0.0
        loss = Loss(outputs, props.float(), loss_mask, properties)
        for i,property in enumerate(properties):
            total_loss += loss[property]
        total_loss.backward()
        optimizer.step()
        scheduler.step()

    return None

def test(model, Loss, train_dataloader, test_dataloader, device, scaler, optimizer, scheduler, epoch, properties):

    train_loss = {}
    test_loss = {}
    r2_train = {}
    r2_test = {}

    for i,property in enumerate(properties):
        train_loss[property] = 0.0
        test_loss[property] = 0.0
        r2_train[property] = 0.0
        r2_test[property] = 0.0
    
    model.eval()

    with torch.no_grad():
        train_pred = {property: [] for property in properties}
        train_true = {property: [] for property in properties}
        test_pred = {property: [] for property in properties}
        test_true = {property: [] for property in properties}
        train_loss_mask = {property: [] for property in properties}
        test_loss_mask = {property: [] for property in properties}

        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)  
            new_feats = batch["new_feats"].to(device)          
            props = batch["props"].to(device)
            outputs = model(input_ids,attention_mask,new_feats).float()

            for i,property in enumerate(properties):
                outputs[:,i] = torch.from_numpy(scaler[i].inverse_transform(outputs[:, i].reshape(-1, 1).cpu().numpy())).view(-1)
                props[:,i] = torch.from_numpy(scaler[i].inverse_transform(props[:, i].reshape(-1, 1).cpu().numpy())).view(-1)

            loss = Loss(outputs, props.float(), loss_mask, properties)

            for i,property in enumerate(properties):
                train_loss[property] += loss[property].item() * torch.sum(loss_mask[:,i])
                train_pred[property].append(outputs[:, i].unsqueeze(1).double().to(device))
                train_true[property].append(props[:, i].unsqueeze(1).double().to(device)) 
                train_loss_mask[property].append(loss_mask[:, i].unsqueeze(1).double().to(device))

        for i,property in enumerate(properties):
            train_pred[property] = torch.cat(train_pred[property], dim=0)
            train_true[property] = torch.cat(train_true[property], dim=0)
            train_loss_mask[property] = torch.cat(train_loss_mask[property], dim=0)

        for i,property in enumerate(properties):      
            train_loss[property] = train_loss[property] / torch.sum(train_loss_mask[property])
            r2_train[property] = r2score(train_pred[property].flatten().to("cpu"), train_true[property].flatten().to("cpu"), train_loss_mask[property].flatten().to("cpu"))
            print(f'train {property} RMSE = ', torch.sqrt(train_loss[property]).item())
            print(f'train {property} r^2 = ', r2_train[property].item())


        for step, batch in enumerate(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)        
            new_feats = batch["new_feats"].to(device)    
            props = batch["props"].to(device)
            outputs = model(input_ids,attention_mask,new_feats).float()

            for i,property in enumerate(properties):
                outputs[:,i] = torch.from_numpy(scaler[i].inverse_transform(outputs[:, i].reshape(-1, 1).cpu().numpy())).view(-1)
                props[:,i] = torch.from_numpy(scaler[i].inverse_transform(props[:, i].reshape(-1, 1).cpu().numpy())).view(-1)
                
            loss = Loss(outputs, props.float(), loss_mask, properties)

            for i,property in enumerate(properties):
                test_loss[property] += loss[property].item() * torch.sum(loss_mask[:,i])
                test_pred[property].append(outputs[:, i].unsqueeze(1).double().to(device))
                test_true[property].append(props[:, i].unsqueeze(1).double().to(device)) 
                test_loss_mask[property].append(loss_mask[:, i].unsqueeze(1).double().to(device))

        for i,property in enumerate(properties):
            test_pred[property] = torch.cat(test_pred[property], dim=0)
            test_true[property] = torch.cat(test_true[property], dim=0)
            test_loss_mask[property] = torch.cat(test_loss_mask[property], dim=0)

        for i,property in enumerate(properties):      
            test_loss[property] = test_loss[property] / torch.sum(test_loss_mask[property])
            r2_test[property] = r2score(test_pred[property].flatten().to("cpu"), test_true[property].flatten().to("cpu"), test_loss_mask[property].flatten().to("cpu"))
            print(f'test {property} RMSE = ', torch.sqrt(test_loss[property]).item())
            print(f'test {property} r^2 = ', r2_test[property].item())


    for i,property in enumerate(properties):  
        writer.add_scalar(f"{property}_Loss/train", train_loss[property], epoch)
        writer.add_scalar(f"{property}_r^2/train", r2_train[property], epoch)
        writer.add_scalar(f"{property}_Loss/test", test_loss[property], epoch)
        writer.add_scalar(f"{property}_r^2/test", r2_test[property], epoch)

    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'epoch': epoch}
    torch.save(state, multitask_config['save_path'])

    return train_loss, test_loss, r2_train, r2_test


def main(multitask_config):

    f = open('multi_2.json')
    feats_to_norm = json.load(f)

    total_mean_best_r2 = 0.0        

    """Data"""
    if multitask_config['CV_flag']:
        print("Start Cross Validation")
        data = pd.read_csv(multitask_config['train_file'])
        print("No. of features:",len(data.columns[1:-2]))
        print()
        """K-fold"""
        splits = KFold(n_splits=multitask_config['k'], shuffle=True,
                       random_state=1)  # k=1 for train-test split and k=5 for cross validation
        
        properties = ['EPS','Nc']

        train_loss_avg, test_loss_avg, train_r2_avg, test_r2_avg = {}, {}, {}, {} # monitor the best metrics in each fold

        for i, property in enumerate(properties):
            train_loss_avg[property] = []
            test_loss_avg[property] = []
            train_r2_avg[property] = []
            test_r2_avg[property] = []
    
        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(data.shape[0]))):
            print('-------------------------------------------------------------------------------------------------------------------------')
            print('Fold {}'.format(fold + 1))

            train_data = data.loc[train_idx, :].reset_index(drop=True)
            test_data = data.loc[val_idx, :].reset_index(drop=True)

            if multitask_config['aug_flag']:
                print("Data Augamentation")
                DataAug = MultiTask_DataAugmentation(multitask_config['aug_indicator'])
                train_data = DataAug.smiles_augmentation(train_data)

            if multitask_config['normalize']:
                print("Some features are normalized")
                scaler_feats = StandardScaler()
                train_data.loc[:, feats_to_norm] = scaler_feats.fit_transform(train_data.loc[:, feats_to_norm])
                test_data.loc[:, feats_to_norm] = scaler_feats.transform(test_data.loc[:, feats_to_norm])
            else:
                print("All features are normalized")
                scaler_feats = StandardScaler()
                train_data.iloc[:, 1:-2] = scaler_feats.fit_transform(train_data.iloc[:, 1:-2])
                test_data.iloc[:, 1:-2] = scaler_feats.transform(test_data.iloc[:, 1:-2])

            scaler = []
            
            for i,property in enumerate(properties):
                scaled = StandardScaler()

                train_data.loc[train_data[property] != -99, property] = scaled.fit_transform(
                    train_data.loc[train_data[property] != -99, [property]]).flatten()

                test_data.loc[test_data[property] != -99, property] = scaled.transform(
                    test_data.loc[test_data[property] != -99, [property]]).flatten()
                
                scaler.append(scaled)

            train_dataset = MultiTask_Dataset(train_data,tokenizer, multitask_config['blocksize'])
            test_dataset = MultiTask_Dataset(test_data,tokenizer, multitask_config['blocksize'])

            train_dataloader = DataLoader(train_dataset, multitask_config['batch_size'], shuffle=True, num_workers=multitask_config["num_workers"])

            test_dataloader = DataLoader(test_dataset, multitask_config['batch_size'], shuffle=False, num_workers=multitask_config["num_workers"])

            """Parameters for scheduler"""
            steps_per_epoch = train_data.shape[0] // multitask_config['batch_size']
            training_steps = steps_per_epoch * multitask_config['num_epochs']
            warmup_steps = int(training_steps * multitask_config['warmup_ratio'])

            """Train the model"""
            model = MultiTask(drop_rate=multitask_config['drop_rate']).to(device)
            model = model.double()
            print(model)
            Loss = MaskedLoss()


            if multitask_config['LLRD_flag']:
                optimizer = roberta_base_AdamW_LLRD(model, multitask_config['lr_rate'], multitask_config['weight_decay'])
            else:
                optimizer = AdamW(
                    [
                        {"params": model.PretrainedModel.parameters(), "lr": multitask_config['lr_rate'],
                         "weight_decay": 0.0},
                        {'params': model.EPS_FCNN.parameters(), 'lr': multitask_config['lr_rate_reg'], 'weight_decay':multitask_config['weight_decay']},
                        {'params': model.Nc_FCNN.parameters(), 'lr': multitask_config['lr_rate_reg'], 'weight_decay': multitask_config['weight_decay']},
                       
                    ],
                    no_deprecation_warning=True
                )


            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)
            torch.cuda.empty_cache()
            fold_train_loss_best, fold_test_loss_best, fold_best_train_r2, fold_best_test_r2 = {}, {}, {}, {}  # Keep track of the best test r^2 in one fold. If cross-validation is not used, that will be the same as best_r2.
            fold_mean_best_test_r2 = 0.0
            count = 0     # Keep track of how many successive non-improvement epochs
            for epoch in range(multitask_config['num_epochs']):
                print("epoch: %s/%s" % (epoch+1, multitask_config['num_epochs']))
                print('properties',properties)
                train(model, optimizer, scheduler, Loss, train_dataloader, device, properties)
                train_loss, test_loss, r2_train, r2_test = test(model, Loss, train_dataloader,test_dataloader, device, scaler, optimizer, scheduler, epoch, properties)

                if (sum(r2_test[property] for property in r2_test) / len(properties)) > fold_mean_best_test_r2:
                    for i,property in enumerate(properties):
                        fold_best_train_r2[property] = r2_train[property]
                        fold_best_test_r2[property] = r2_test[property]
                        fold_train_loss_best[property] = train_loss[property]
                    
                        fold_test_loss_best[property] = test_loss[property]
                    count = 0
                    fold_mean_best_test_r2 = sum(r2_test[property] for property in r2_test) / len(properties)
                    print(f'Updated fold_train_loss_best[{property}]:', fold_train_loss_best[property])
                else:
                    count += 1

                if sum(r2_test[property] for property in r2_test) / len(properties) >  total_mean_best_r2:
                    total_mean_best_r2 = sum(r2_test[property] for property in r2_test) / len(properties)  
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'fold:': fold}
                    torch.save(state, multitask_config['best_model_path'])   # save the best model

                if count >= multitask_config['tolerance']:
                    print("Early stop")
                    if fold_mean_best_test_r2 == 0:
                        print("Poor performance with negative r^2")
                    break
            
            for i,property in enumerate(properties):
                train_loss_avg[property].append(np.sqrt(fold_train_loss_best[property].cpu()))
                test_loss_avg[property].append(torch.sqrt(fold_test_loss_best[property].cpu()))
                train_r2_avg[property].append(fold_best_train_r2[property].cpu())
                test_r2_avg[property].append(fold_best_test_r2[property].cpu())
            writer.flush()

        """Average of metrics over all folds"""
        for i,property in enumerate(properties):
            train_rmse = np.mean(np.array(train_loss_avg[property]))
            test_rmse = np.mean(np.array(test_loss_avg[property]))
            train_r2 = np.mean(np.array(train_r2_avg[property]))
            test_r2 = np.mean(np.array(test_r2_avg[property]))
            std_test_rmse = np.std(np.array(test_loss_avg[property]))
            std_test_r2 = np.std(np.array(test_r2_avg[property]))

            print(f"Train RMSE for {property} = {train_rmse}")
            print(f"Train R^2 {property} = {train_r2}")
            print(f"Test RMSE for {property} = {test_rmse}")
            print(f"Test R^2 {property} = {test_r2}")
            print(f"Standard Deviation of Test RMSE {property} = {std_test_rmse}")
            print(f"Standard Deviation of Test R^2 {property} = {std_test_r2}")

            writer.flush()

        """Average of metrics over all folds"""
        for i,property in enumerate(properties):
            train_rmse = np.mean(np.array(train_loss_avg[property]))
            test_rmse = np.mean(np.array(test_loss_avg[property]))
            train_r2 = np.mean(np.array(train_r2_avg[property]))
            test_r2 = np.mean(np.array(test_r2_avg[property]))
            std_test_rmse = np.std(np.array(test_loss_avg[property]))
            std_test_r2 = np.std(np.array(test_r2_avg[property]))

            print(f"Train RMSE for {property} = {train_rmse}")
            print(f"Train R^2 {property} = {train_r2}")
            print(f"Test RMSE for {property} = {test_rmse}")
            print(f"Test R^2 {property} = {test_r2}")
            print(f"Standard Deviation of Test RMSE {property} = {std_test_rmse}")
            print(f"Standard Deviation of Test R^2 {property} = {std_test_r2}")

        writer.flush()    

    else:
        print("Train Test Split")
        train_data = pd.read_csv(multitask_config['train_file'])
        test_data = pd.read_csv(multitask_config['test_file'])

        if multitask_config['aug_flag']:
            print("Data Augamentation")
            DataAug = MultiTask_DataAugmentation(multitask_config['aug_indicator'])
            train_data = DataAug.smiles_augmentation(train_data)

        properties = ['EPS','Nc']

        scaler = []
            
        for i,property in enumerate(properties):
            scaled = StandardScaler()

            train_data.loc[train_data[property] != -99, property] = scaled.fit_transform(
                train_data.loc[train_data[property] != -99, [property]]).flatten()
                
            test_data.loc[test_data[property] != -99, property] = scaled.transform(
                test_data.loc[test_data[property] != -99, [property]]).flatten()
                 
            scaler.append(scaled)

        train_dataset = MultiTask_Dataset(train_data,tokenizer, multitask_config['blocksize'])
        test_dataset = MultiTask_Dataset(test_data,tokenizer, multitask_config['blocksize'])

        train_dataloader = DataLoader(train_data, multitask_config['batch_size'], shuffle=True, num_workers=multitask_config["num_workers"])
        test_dataloader = DataLoader(test_data, multitask_config['batch_size'], shuffle=False, num_workers=multitask_config["num_workers"])

        """Parameters for scheduler"""
        steps_per_epoch = train_data.shape[0] // multitask_config['batch_size']
        training_steps = steps_per_epoch * multitask_config['num_epochs']
        warmup_steps = int(training_steps * multitask_config['warmup_ratio'])

        """Train the model"""
        model = MultiTask(drop_rate=multitask_config['drop_rate']).to(device)
        model = model.double()
        Loss = MaskedLoss()

        if multitask_config['LLRD_flag']:
                optimizer = roberta_base_AdamW_LLRD(model, multitask_config['lr_rate'], multitask_config['weight_decay'])
        else:
            optimizer = AdamW(
                [
                    {"params": model.PretrainedModel.parameters(), "lr": multitask_config['lr_rate'],
                     "weight_decay": 0.0},
                    {'params': model.Eea_FCNN.parameters(), 'lr': multitask_config['lr_rate_reg'], 'weight_decay':multitask_config['weight_decay']},
                    {'params': model.Egb_FCNN.parameters(), 'lr': multitask_config['lr_rate_reg'], 'weight_decay': multitask_config['weight_decay']},
                    {'params': model.Egc_FCNN.parameters(), 'lr': multitask_config['lr_rate_reg'], 'weight_decay': multitask_config['weight_decay']},
                    {'params': model.Ei_FCNN.parameters(), 'lr': multitask_config['lr_rate_reg'], 'weight_decay': multitask_config['weight_decay']}                     
                ],
                no_deprecation_warning=True
            )

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=training_steps)
        torch.cuda.empty_cache()
        fold_train_loss_best, fold_test_loss_best, fold_best_train_r2, fold_best_test_r2 = {}, {}, {}, {}  # Keep track of the best test r^2 in one fold. If cross-validation is not used, that will be the same as best_r2.
        fold_mean_best_test_r2 = 0.0
        count = 0     # Keep track of how many successive non-improvement epochs
        for epoch in range(multitask_config['num_epochs']):
            print("epoch: %s/%s" % (epoch+1, multitask_config['num_epochs']))
            train(model, optimizer, scheduler, Loss, train_dataloader, device, properties)
            train_loss, test_loss, r2_train, r2_test = test(model, Loss, train_dataloader,test_dataloader, device, scaler, optimizer, scheduler, epoch, properties)

            if sum(r2_test[property] for property in r2_test) / len(properties) > fold_mean_best_test_r2:
                for i,property in enumerate(properties):
                    fold_best_train_r2[property] = r2_train[property]
                    fold_best_test_r2[property] = r2_test[property]
                    fold_mean_best_test_r2 += fold_best_test_r2[property]
                    fold_train_loss_best[property] = train_loss[property]
                    fold_test_loss_best[property] = test_loss[property]
                count = 0
                fold_mean_best_test_r2 = sum(r2_test[property] for property in r2_test) / len(properties)
            else:
                count += 1

            if sum(r2_test[property] for property in r2_test) / len(properties) >  total_mean_best_r2:
                total_mean_best_r2 = sum(r2_test[property] for property in r2_test) / len(properties)  
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}
                torch.save(state, multitask_config['best_model_path'])   # save the best model

            if count >= multitask_config['tolerance']:
                print("Early stop")
                if fold_mean_best_test_r2 == 0:
                    print("Poor performance with negative r^2")
                break
        
        writer.flush()



if __name__ == "__main__":

    multitask_config = yaml.load(open("config_Multi_optical_TransChem_finetune_params.yaml", "r"), Loader=yaml.FullLoader)
    print(multitask_config)

    """Device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PretrainedModel = RobertaModel.from_pretrained(multitask_config['model_path'])
    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=multitask_config['blocksize'])
    PretrainedModel.config.hidden_dropout_prob = multitask_config['hidden_dropout_prob']
    PretrainedModel.config.attention_probs_dropout_prob = multitask_config['attention_probs_dropout_prob']


    """Run the main function"""
    main(multitask_config)