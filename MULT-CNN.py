import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import numpy as np
import os
import time
from PIL import Image
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class MultiHeadCNN(nn.Module):
    def __init__(self):
        super(MultiHeadCNN, self).__init__()
        
        self.branches = nn.ModuleList()
        for i in range(5):
            branch = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 128 -> 64
                
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 64 -> 32
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
                
                nn.Flatten(),
                
                nn.Linear(64 * 16 * 16, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5)
            )
            self.branches.append(branch)
        
        self.fusion = nn.Sequential(
            nn.Linear(256 * 5, 512),  # 5 concat
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [batch_size, 5, 128, 128]
        
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            channel_data = x[:, i:i+1, :, :]
            
            branch_out = branch(channel_data)  # [batch_size, 256]
            branch_outputs.append(branch_out)
        
        concat_output = torch.cat(branch_outputs, dim=1)  # [batch_size, 256*5]
        
        output = self.fusion(concat_output)  # [batch_size, 1]
        
        return output

def split_data(input_data, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    num_samples = 2000
    
    indices = np.random.permutation(num_samples)
    
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    X_train = input_data[train_indices]
    X_val = input_data[val_indices]
    X_test = input_data[test_indices]
    
    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            train_indices, val_indices, test_indices)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=2000):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                val_loss += criterion(outputs, batch_labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

def test_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    mse = np.mean((predictions - true_labels) ** 2)
    mae = np.mean(np.abs(predictions - true_labels))
    rmse = np.sqrt(mse)
    
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return predictions, true_labels, mse, mae, rmse

def cross_validation(input_data, labels, k_folds=5, epochs=50):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(input_data)):
        
        X_train_fold = input_data[train_idx]
        y_train_fold = labels[train_idx]
        X_val_fold = input_data[val_idx]
        y_val_fold = labels[val_idx]
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_fold),
            torch.FloatTensor(y_train_fold)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_fold),
            torch.FloatTensor(y_val_fold)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiHeadCNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, epochs
        )
        
        model.eval()
        val_predictions = []
        val_true = []
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                val_predictions.extend(outputs.cpu().numpy())
                val_true.extend(batch_labels.cpu().numpy())
        
        val_mse = np.mean((np.array(val_predictions) - np.array(val_true)) ** 2)
        fold_results.append(val_mse)
        
    return fold_results


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    data_path = r'H:\voronoi_0522\data'
    num_samples = 2000 
    input_data = np.zeros((num_samples, 128, 128, 5), dtype=np.float32)
    labels = np.zeros((num_samples, 1), dtype=np.float32)
    
    for i in range(1, num_samples + 1):
        folder = os.path.join(data_path, f'stru_{i}')
        
        with open(os.path.join(folder, 'label.txt'), 'r') as f:
            labels[i-1, 0] = float(f.read().strip())
        
        for j in range(1, 6):
            img = Image.open(os.path.join(folder, f'{j}.png'))
            img_array = np.absolute(np.array(img.convert('L'), dtype=np.float32)-254)/254
            input_data[i-1, :, :, j-1] = img_array
    label = np.array(np.absolute((labels-140000)/40000))
   
    X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = split_data(input_data, label, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    train_start_time = time.time()
    cv_results = cross_validation(X_train, y_train, k_folds=5, epochs=50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiHeadCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    CNN_model, train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=2000)
    train_end_time = time.time()
    
    train_total_time = train_end_time-train_start_time
    print(train_total_time)
    
    train_pred, train_label, train_mse, train_mae, train_rmse = test_model(CNN_model, train_loader, device)
    val_pred, val_label, val_mse, val_mae, val_rmse = test_model(CNN_model, val_loader, device)
    test_pred, test_label, test_mse, test_mae, test_rmse = test_model(CNN_model, test_loader, device)
    
    
    
    
    