import torch
from torch.utils.data import DataLoader
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from task import IIoTFLNet

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):

    model = IIoTFLNet()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    