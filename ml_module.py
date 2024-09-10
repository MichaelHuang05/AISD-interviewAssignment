import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(13, 128)
        self.dropout = nn.Dropout(p=0.5)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)          
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x




class DataPreprocess():
    def __init__(self, data_df):
        self.data = data_df
    
    def preprocess(self, inference = False, y_col = ['Exited'], non_feature_col = ['id', 'CustomerId', 'Surname'], cat_col = ['Geography', 'Gender']):
        ''' Preprocess data '''
        X = self.data.drop(columns = non_feature_col)
        X['Age'] = X['Age'].astype(int)
        # One hot encoding
        X = pd.get_dummies(X, columns = cat_col)
        booling_to_int_col = []
        for i in cat_col:
            booling_to_int_col.extend(X.filter(regex = i).columns.tolist())
        # Convert booling to integer 
        X[booling_to_int_col] = X[booling_to_int_col].astype(int)

        if not inference:
            X = X.drop(columns = y_col)
            y = self.data[y_col]
            return X, y
        else:
            return X
    

def Train_model(X_train, y_train, model_name = 'xgboost', n_estimators = 1000, num_epochs = 200):
    if model_name == 'xgboost':
        # Train xgb model
        model = xgb.XGBClassifier(eval_metric='logloss',        
                                  n_estimators=n_estimators,             
                                  max_depth=4,                  
                                  learning_rate=0.1)
        
        model.fit(X_train, y_train)

    elif model_name == 'DNN':
        # Train DNN model
        # To tensor
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)

        # Load to dataloader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device : {}'.format(device))
        model = DNN().to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = num_epochs
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    else:
        model =  None

    return model 


def make_prediction(model, model_name, X_test, DNN_threshold = .5):
    # Make prediction on test dataset
    if model_name == 'xgboost':
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

    elif model_name == 'DNN':
        ## Convert to tensor        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        ## Predict 
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X_test_tensor)
            y_pred = (y_pred_prob >= DNN_threshold).to(int)
            
        ## Convert tensor back to numpy
        y_pred = y_pred.numpy()
        y_pred_prob = y_pred_prob.squeeze().numpy()

    return y_pred, y_pred_prob






def plot_roc_curve(roc_dict, model_list):
    plt.figure(figsize=(8, 4))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    for model_name in model_list:
        y_test = roc_dict[model_name][0]
        y_pred_prob = roc_dict[model_name][1]

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (area = {auc_score:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    return None




