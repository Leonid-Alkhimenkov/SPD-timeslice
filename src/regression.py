import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os


class CNNVertexRegressor(nn.Module):
    """
    Advanced CNN-based Vertex Regressor with Residual Connections and Batch Normalization.
    """
    def __init__(self, input_dim=3, conv_hidden_dim=128, output_dim=3, dropout_rate=0.25):
        super(CNNVertexRegressor, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_dim, conv_hidden_dim, kernel_size=1),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.ReLU(),
            nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=1),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=1),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.ReLU(),
            nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=1),
            nn.BatchNorm1d(conv_hidden_dim)
        )
        self.residual_connection = nn.Sequential(
            nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=1),
            nn.BatchNorm1d(conv_hidden_dim)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=1),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.ReLU(),
            nn.Conv1d(conv_hidden_dim, conv_hidden_dim, kernel_size=1),
            nn.BatchNorm1d(conv_hidden_dim),
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(conv_hidden_dim, conv_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(conv_hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass for the model.
        Args:
            x: Tensor of shape (batch_size, num_hits, 3)
        Returns:
            Tensor of shape (batch_size, 3)
        """
        x = x.permute(0, 2, 1)  # Перестановка размерностей: (batch_size, 3, num_hits)
        x = self.conv_block1(x)
        residual = self.residual_connection(x)
        x = self.conv_block2(x)
        x += residual
        x = torch.relu(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x).squeeze(-1)  # Адаптивное усреднение по хитовому измерению
        x = self.fc(x)
        return x


class HitsDataset(Dataset):
    """
    Custom Dataset for handling variable-length hits and corresponding vertices.
    """
    def __init__(self, hits, vertices):
        self.hits = hits
        self.vertices = vertices

    def __len__(self):
        return len(self.hits)

    def __getitem__(self, idx):
        return self.hits[idx], self.vertices[idx]


def collate_fn(batch):
    """
    Collate function to handle variable number of hits per vertex.
    Returns a padded tensor for hits, a tensor for vertices, and a mask.
    """
    hits, vertices = zip(*batch)
    hits = [torch.tensor(hit, dtype=torch.float32) for hit in hits]
    padded_hits = torch.nn.utils.rnn.pad_sequence(hits, batch_first=True)
    vertices = torch.tensor(vertices, dtype=torch.float32)

    return padded_hits, vertices


class VertexRegressor:
    """
    Wrapper for training and inference of CNN-based Vertex Regressor.
    """
    def __init__(self, model_dir="models", learning_rate=1e-3, num_epochs=500, batch_size=32):
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNVertexRegressor().to(self.device)

    @staticmethod
    def compute_r2(y_true, y_pred):
        """
        Compute R-squared (coefficient of determination).
        Args:
            y_true: Tensor of true values (batch_size, output_dim).
            y_pred: Tensor of predicted values (batch_size, output_dim).
        Returns:
            R-squared value.
        """
        ss_total = torch.sum((y_true - torch.mean(y_true, dim=0)) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2.item()

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the CNN-based regressor.
        """
        train_dataset = HitsDataset(X_train, y_train)
        val_dataset = HitsDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        optimizer = optim.Adadelta(self.model.parameters())
        criterion = nn.MSELoss()

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            train_r2 = 0.0
            for padded_hits, vertices in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                padded_hits = padded_hits.to(self.device)
                vertices = vertices.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(padded_hits)
                loss = criterion(predictions, vertices)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_r2 += self.compute_r2(vertices, predictions)

            train_loss /= len(train_loader)
            train_r2 /= len(train_loader)

            val_loss, val_r2 = self.evaluate(val_loader, criterion)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_loss:.4f}, Train R2: {train_r2:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val R2: {val_r2:.4f}")

        model_path = f"{self.model_dir}/vertex_regressor.pth"
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def evaluate(self, val_loader, criterion):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        val_loss = 0.0
        val_r2 = 0.0
        with torch.no_grad():
            for padded_hits, vertices in val_loader:
                padded_hits = padded_hits.to(self.device)
                vertices = vertices.to(self.device)
                predictions = self.model(padded_hits)
                loss = criterion(predictions, vertices)
                val_loss += loss.item()
                val_r2 += self.compute_r2(vertices, predictions)

        val_loss /= len(val_loader)
        val_r2 /= len(val_loader)
        return val_loss, val_r2

    def predict(self, X):
        """
        Predict vertex coordinates for the given hits.
        Args:
            X: numpy array of shape (num_hits, 3) or list of such arrays.
        Returns:
            numpy array of shape (batch_size, 3) with predicted vertex coordinates.
        """
        self.model.eval()
        if isinstance(X, np.ndarray) and len(X.shape) == 1 and X.shape[0] == 3:
            X = X.reshape(1, 3)
        if isinstance(X, np.ndarray) and len(X.shape) == 2 and X.shape[1] == 3:
            X = [X]
        if not isinstance(X, list):
            raise ValueError(f"Invalid input type for X: {type(X)}")
        hits = [torch.tensor(hit, dtype=torch.float32) for hit in X]
        padded_hits = torch.nn.utils.rnn.pad_sequence(hits, batch_first=True).to(self.device)
        with torch.no_grad():
            predictions = self.model(padded_hits)

        return predictions.cpu().numpy()

    def load_model(self):
        """
        Load the trained model from disk.
        """
        model_path = f"{self.model_dir}/vertex_regressor.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()