import os
import gdown
import torch
from transformers import DistilBertTokenizer


def initialize_distilbert_transform(max_token_length):
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_token_length,
            return_tensors='pt')
        x = torch.stack((tokens['input_ids'], tokens['attention_mask']), dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform

    
def download_gdrive(url, save_path, is_folder):
    """ Download the preprocessed data from Google Drive. """
    if not is_folder:
        gdown.download(url=url, output=save_path, quiet=False)
    else:
        gdown.download_folder(url=url, output=save_path, quiet=False)


def download_arxiv(data_dir):
    download_gdrive(
        url='https://drive.google.com/u/0/uc?id=1H5xzHHgXl8GOMonkb6ojye-Y2yIp436V&export=download',
        save_path=os.path.join(data_dir, 'arxiv.pkl'),
        is_folder=False
    )


def download_fmow(data_dir):
    download_gdrive(
        url='https://drive.google.com/u/0/uc?id=1s_xtf2M5EC7vIFhNv_OulxZkNvrVwIm3&export=download',
        save_path=os.path.join(data_dir, 'fmow.pkl'),
        is_folder=False
    )


def download_huffpost(data_dir):
    download_gdrive(
        url='https://drive.google.com/u/0/uc?id=1jKqbfPx69EPK_fjgU9RLuExToUg7rwIY&export=download',
        save_path=os.path.join(data_dir, 'huffpost.pkl'),
        is_folder=False
    )


def download_yearbook(data_dir):
    if not os.path.exists(os.path.join(data_dir, 'yearbook.pkl')):
        download_gdrive(
            url='https://drive.google.com/u/0/uc?id=1mPpxoX2y2oijOvW1ymiHEYd7oMu2vVRb&export=download',
            save_path=os.path.join(data_dir, 'yearbook.pkl'),
            is_folder=False
        )



def download_detection(data_dir, dataset_file):
    if os.path.isfile(data_dir):
        raise RuntimeError('Save path should be a directory!')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if os.path.isfile(os.path.join(data_dir, dataset_file)):
        pass
    else:
        if 'arxiv' in dataset_file:
            download_arxiv(data_dir)
        elif 'fmow' in dataset_file:
            download_fmow(data_dir)
        elif 'huffpost' in dataset_file:
            download_huffpost(data_dir)
        elif 'yearbook' in dataset_file:
            download_yearbook(data_dir)
        else:
            raise RuntimeError(f'The dataset {dataset_file} does not exist in WildTime!')
