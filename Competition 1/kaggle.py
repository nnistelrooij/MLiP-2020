import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

from nn import BengaliNet
from utils.preprocess import preprocess


def test(model, test_images, transform, batch_size=192):
    """Test the model by predicting classes of unseen images.

    Args:
        model       = [nn.Module] model to test with dataset of unseen images
        test_images = [ndarray] unseen images of which classes will be predicted
        transform   = [Compose] normalization transform applied to each image
        batch_size  = [int] number of images in a mini-batch

    Returns [list]:
        Class predictions as three consecutive integers for each test image in
        a flattened list with sub-problem order consonant diacritic,
        grapheme root, and vowel diacritic.
    """
    predictions = []
    with torch.no_grad():
        for batch_idx in tqdm(range(0, len(test_images), batch_size)):
            # select batch of images to process and normalize them
            batch = test_images[batch_idx:batch_idx + batch_size]
            x = torch.stack([transform(image) for image in batch])

            # predict class of each sub-problem for each image in batch
            y = model(x)

            # prepare predictions for Kaggle with correct sub-problem order
            preds = [y[idx].argmax(dim=-1) for idx in [2, 0, 1]]
            preds = torch.stack(preds, dim=1).flatten()

            predictions += preds.tolist()

    return predictions


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
    predictions = test(model, preprocessed_test_images, transform)

    # save predictions to submission CSV file
    submission_df = pd.read_csv(
        'kaggle/input/bengaliai-cv19/sample_submission.csv'
    )
    submission_df.target = predictions
    submission_df.to_csv('submission.csv', index=False)
