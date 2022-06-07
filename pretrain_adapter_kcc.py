from transformer.src.transformers.models.bart.configuration_bart import BartConfig
from transformer.src.transformers.models.bart.modeling_bart import BartModel, BartForSequenceClassification
from transformers import BartTokenizer, AdamW, get_linear_schedule_with_warmup
import transformers
from transformers.activations import get_activation
from torch.utils.data import DataLoader
from dataset import StreamDataset
from utils import seed_everything
import torch.nn.functional as F
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
import gc

transformers.logging.set_verbosity_error()
seed_everything(42)

out = open('pretrain_adapter_kcc.txt', 'w')

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class BartClassificationCollator(object):
    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        contexts = [sequence['context'] for sequence in sequences]
        statements = [sequence['statement'] for sequence in sequences]

        labels = [sequence['label'] for sequence in sequences]
        labels = [self.labels_encoder[label] for label in labels]

        inputs = self.use_tokenizer(contexts, statements, return_tensors="pt", padding='max_length', truncation=True,  max_length=self.max_sequence_len)
        inputs.update({'labels':torch.tensor(labels)})

        return inputs

class Activation_Function_Class(nn.Module):
    def __init__(self, hidden_act):
        super().__init__()
        self.f = get_activation(hidden_act.lower())
    
    def forward(self, x):
        return self.f(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=48):
        super(AutoEncoder, self).__init__()
        self.input_size = input_dim
        self.output_size = output_dim
        
        seq_list = []
        self.non_linearity = Activation_Function_Class('relu')
        seq_list.append(nn.Linear(self.input_size, self.output_size))
        seq_list.append(self.non_linearity)

        self.adapter_down = nn.Sequential(*seq_list)
        self.adapter_up = nn.Linear(self.output_size, self.input_size)

        self.adapter_down.apply(self.init_bert_weights)
        self.adapter_up.apply(self.init_bert_weights)
    
    def forward(self, x):
        down = self.adapter_down(x)
        up = self.adapter_up(down)

        return up
    
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def train(dataloader, optimizer, scheduler, device, model):
  predictions_labels = []
  true_labels = []
  total_loss = 0
  model.to(device)
  for batch in tqdm(dataloader, total=len(dataloader)):
    true_labels += batch['labels'].numpy().flatten().tolist()

    batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}

    model.zero_grad()

    outputs = model(**batch)
    loss, logits = outputs[:2]
    total_loss += loss.item()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    logits = logits.detach().cpu().numpy()

    predictions_labels += logits.argmax(axis=-1).flatten().tolist()

  avg_epoch_loss = total_loss / len(dataloader)

  return true_labels, predictions_labels, avg_epoch_loss

def test(dataloader, device, model):
  predictions_labels = []
  true_labels = []
  total_loss = 0
  model.to(device)
  for batch in tqdm(dataloader, total=len(dataloader)):

    true_labels += batch['labels'].numpy().flatten().tolist()
    batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}

    with torch.no_grad():        
        outputs = model(**batch)

        loss, logits = outputs[:2]

        logits = logits.detach().cpu().numpy()

        total_loss += loss.item()

        predict_content = logits.argmax(axis=-1).flatten().tolist()
        predictions_labels += predict_content

  avg_epoch_loss = total_loss / len(dataloader)

  return true_labels, predictions_labels, avg_epoch_loss

def get_return_layers_target(task, current_task):
    return_layers = {}
    l = 'model.encoder.layers.0.output_adapters.adapters.' + task
    return_layers[l] = 'model.encoder.layers.0.output_adapters.adapters.' + current_task

    return return_layers

def get_return_layers():
    return_layers = {}
    l = 'model.encoder.layers.0.final_layer_norm'
    return_layers[l] = 'model.encoder.layers.0.final_layer_norm'

    return return_layers

def ssl(dataloader, device, tasks, current_task, task_list):
    pretrained_adapter = {}
    model_name = 'facebook/bart-base'

    for task in tasks:
        config = BartConfig.from_pretrained(model_name, num_labels=2)
        model_past = BartForSequenceClassification.from_pretrained(model_name, config=config)
        model_for_ssl = BartForSequenceClassification.from_pretrained(model_name, config=config)
        
        adapter_dir = './adapters/ssl_adapter/' + task + '/'
        model_past.load_adapter(adapter_dir)

        model_past.to(device)
        model_for_ssl.to(device)

        return_layers = get_return_layers_target(task, current_task)
        return_layers_for_ssl = get_return_layers()

        ae = AutoEncoder().to(device)
        criterion = nn.MSELoss()
        optimizer = AdamW(ae.parameters(), lr=1e-4, no_deprecation_warning=True)

        for epoch in tqdm(range(10)):
            for batch in tqdm(dataloader):
                gc.collect()
                batch = {k:v.type(torch.long).to(device) for k, v in batch.items()}

                model_past.set_active_adapters(task)
                mid_getter = MidGetter(model_past, return_layers=return_layers, keep_output=True)
                mid_outputs, final_output = mid_getter(**batch)
                
                mid_getter_for_ssl = MidGetter(model_for_ssl, return_layers=return_layers_for_ssl, keep_output=False)
                mid_outputs_for_ssl, _ = mid_getter_for_ssl(**batch)
                optimizer.zero_grad()
                inp = mid_outputs_for_ssl['model.encoder.layers.0.final_layer_norm'].to(device)
                layer = 'model.encoder.layers.0.output_adapters.adapters.' + current_task
                target = mid_outputs[layer][2].to(device)

                output = ae(inp)
                output = output.to(device)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                del batch, mid_getter_for_ssl, mid_outputs_for_ssl, mid_getter, mid_outputs
                gc.collect()
                torch.cuda.empty_cache()

        pretrained_adapter[layer] = ae.named_parameters()
        del ae, criterion, optimizer, loss
        gc.collect()
        torch.cuda.empty_cache()

    return pretrained_adapter

def main(args, tasks):
    at_accuracy, accuracy = {}, {}
    model_name = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    bart_classification_collator = BartClassificationCollator(use_tokenizer=tokenizer, labels_encoder=args.label_ids, max_sequence_len=args.max_length)

    #Calculate At ACC
    for i, task in enumerate(tasks):
        print(f'{task} Training...')
        config = BartConfig.from_pretrained(model_name, num_labels=2)
        model = BartForSequenceClassification.from_pretrained(model_name, config=config)
        model.resize_token_embeddings(len(tokenizer))

        train_path = '/root/capstone/dataset/StandardStream/train/' + task + '.csv'
        train_dataset = StreamDataset(path=train_path)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=bart_classification_collator, num_workers=4)

        if i != 0:
            pretrained_adapter = ssl(train_dataloader, args.device, [tasks[i-1]], tasks[i], tasks)
            model.add_adapter(task)
            state_dict = model.state_dict()
            pretrain_layer = list(pretrained_adapter.keys())
            pretrain_params = list(pretrained_adapter.values())

            pretrain_state_dict = {}
            for layer, param in zip(pretrain_layer, pretrain_params):
                layers, parameters = [], []
                for p in param:
                    layers.append(p[0])
                    parameters.append(torch.tensor(p[1], requires_grad=True))
            
                load_layers = []

                for l in layers:
                    load_layers.append(layer + '.' + l)
            
                for final_layer, final_param in zip(load_layers, parameters):
                    pretrain_state_dict[final_layer] = final_param

            current_layers = list(pretrain_state_dict.keys())
            current_params = list(pretrain_state_dict.values())

            pretrain_state_dict = {}
            for layer, param in zip(current_layers, current_params):
                pretrain_state_dict[layer] = param

            model.load_state_dict(pretrain_state_dict, strict=False)
        else:
            model.add_adapter(task)
        model.train_adapter(task)
        model.to(args.device)
        print('Current trained tasks: ', model.model.encoder.layers[0].output_adapters.adapters.keys())
        
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.eps)
        total_steps = len(train_dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        model.train()
        for epoch in tqdm(range(args.epochs)):
            print()
            train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, args.device, model)
            train_acc = accuracy_score(train_labels, train_predict)
            print("  train_loss: %.5f - train_acc: %.5f"%(train_loss, train_acc))
            print("  train_loss: %.5f - train_acc: %.5f"%(train_loss, train_acc), file=out)
            print()
        
        test_path = '/root/capstone/dataset/StandardStream/test/' + task + '.csv'
        test_dataset = StreamDataset(path=test_path)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=bart_classification_collator, num_workers=4)

        model.to(args.device)
        model.eval()
        model.set_active_adapters(task)
        test_labels, test_predict, test_loss = test(test_dataloader, args.device, model)
        test_acc = accuracy_score(test_labels, test_predict)

        print('test_loss: %.5f, test_acc: %.5f'%(test_loss, test_acc))
        print('test_loss: %.5f, test_acc: %.5f'%(test_loss, test_acc), file=out)
        at_accuracy[task] = test_acc

        adapter_dir = './adapters/ssl_adapter/' + task + '/'
        model.save_adapter(adapter_dir, task)
    
    #Calculate Final ACC
    for task in tasks:
        config = BartConfig.from_pretrained(model_name, num_labels=2)
        model = BartForSequenceClassification.from_pretrained(model_name, config=config)
        model.resize_token_embeddings(len(tokenizer))

        adapter_dir = './adapters/ssl_adapter/' + task + '/'
        model.load_adapter(adapter_dir)
        model.set_active_adapters(task)
        model.to(args.device) 
        model.eval()
        test_path = '/root/capstone/dataset/StandardStream/test/' + task + '.csv'
        test_dataset = StreamDataset(path=test_path)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=bart_classification_collator, num_workers=4)

        test_labels, test_predict, test_loss = test(test_dataloader, args.device, model)
        test_acc = accuracy_score(test_labels, test_predict)
        print('test_loss: %.5f, test_acc: %.5f'%(test_loss, test_acc), file=out)
        accuracy[task] = test_acc

    print('At Accuracy: ', at_accuracy, file=out)
    print('Final Accuracy: ', accuracy, file=out)

args = Namespace(
    batch_size=32, 
    learning_rate=1e-4,
    eps=1e-8,
    epochs=3, 
    max_length=256, 
    label_ids = {0: 0, 1: 1},
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

tasks = ['boolq', 'udpos', 'wic', 'few_rel', 'yelp_review']

main(args, tasks)
