import os
import cv2
from tqdm import tqdm
from utils.metrics import EvaluationMetricsV2


def evaluate(pred_path, dataset):
    pred_root = pred_path
    metric = EvaluationMetricsV2()
    mask_root = f'./Dataset/TestDataset/{dataset}/GT'
    mask_name_list = sorted(os.listdir(pred_root))
    mask_name_list = [name for name in mask_name_list if name.endswith('.png') or name.endswith('.jpg')]

    for _, mask_name in tqdm(list(enumerate(mask_name_list))):
        pred_path = os.path.join(pred_root, mask_name)
        mask_path = os.path.join(mask_root, mask_name)
        pred = cv2.imread(pred_path, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        if pred.shape != mask.shape:
            pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        metric.step(pred=pred, gt=mask)

    metric_dic = metric.get_results()
    return metric_dic


if __name__ == '__main__':
    pred_path = f'./results'

    datasets = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']

    print(f'Evaluating...')
    for dataset in datasets:
        metric_dic = evaluate(os.path.join(pred_path, dataset), dataset)

        sm = metric_dic['sm']

        emMean = metric_dic['emMean']
        emAdp = metric_dic['emAdp']
        emMax = metric_dic['emMax']

        fmMean = metric_dic['fmMean']
        fmAdp = metric_dic['fmAdp']
        fmMax = metric_dic['fmMax']

        wfm = metric_dic['wfm']
        mae = metric_dic['mae']

        print(f'##### {dataset} #####')
        print(f'sm: {sm}')
        print(f'emMean: {emMean}')
        print(f'emAdp: {emAdp}')
        print(f'emMax: {emMax}')
        print(f'fmMean: {fmMean}')
        print(f'fmAdp: {fmAdp}')
        print(f'fmMax: {fmMax}')
        print(f'wfm: {wfm}')
        print(f'mae: {mae}')
