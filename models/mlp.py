import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers=[512, 128], num_classes=7, drop_prob=0.2):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_layers[-1], num_classes)
        )

    def forward(self, x):
        return self.layers(x)

def train_mlp(embeddings_file, num_classes=7, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading embeddings from {embeddings_file}...")
    data = torch.load(embeddings_file)
    embeddings = data['embeddings']
    labels = data['labels']
    
    input_size = embeddings.shape[1]
    print(f"Input feature size: {input_size}")
    
    dataset = torch.utils.data.TensorDataset(embeddings, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = MLP(input_size=input_size, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")
    
    return model