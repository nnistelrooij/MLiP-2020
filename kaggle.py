import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

from nn import BengaliNet
from utils.preprocess import preprocess

if __name__ == '__main__':
    # preprocess test images
    test_files = ['kaggle/input/bengaliai-cv19/test_image_data_0.parquet',
                  'kaggle/input/bengaliai-cv19/test_image_data_1.parquet',
                  'kaggle/input/bengaliai-cv19/test_image_data_2.parquet',
                  'kaggle/input/bengaliai-cv19/test_image_data_3.parquet']
    preprocessed_test_images = preprocess(test_files, 236, 137, 128)

    # load model from file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BengaliNet(device)
    model.load_state_dict(torch.load('model.pt', map_location=device))
    model.eval()

    # initialize normalizing transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.071374745,), std=(0.20761949,))
    ])

    # determine predictions of model on test images
    predictions = []
    batch_size = 128
    for batch_idx in tqdm(range(0, len(preprocessed_test_images), batch_size)):
        # select batch of images to process and normalize them
        batch = preprocessed_test_images[batch_idx:batch_idx + batch_size]
        x = torch.stack([transform(image) for image in batch])

        # predict class of each sub-problem for each image in batch
        y = model(x)
        preds = [pred.argmax(dim=1).tolist() for pred in y]

        for grapheme, vowel, consonant in zip(*preds):
            predictions += [consonant, grapheme, vowel]

    # save predictions to submission CSV file
    submission_df = pd.read_csv(
        'kaggle/input/bengaliai-cv19/sample_submission.csv'
    )
    submission_df.target = predictions
    submission_df.to_csv('submission.csv', index=False)
