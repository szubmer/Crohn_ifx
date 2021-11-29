import os
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
from radiomics import featureextractor as FEE
import radiomics


def save_mat_by_dcm_base_sio(dcm_img, save_path_with_suffix, key):
    sio.savemat(save_path_with_suffix, {key: dcm_img})


fold_root_path = ''
load_final_label_root_path = ''
load_final_image_root_path = ''

label_name = os.listdir(load_final_label_root_path)

for p in label_name:
    img_path = load_final_image_root_path + p[:-12] + 'image.nii.gz'
    lab_path = load_final_label_root_path + p

    extractor = FEE.RadiomicsFeatureExtractor('Params_labels.yaml')
    extractor.loadParams('Params_labels.yaml')
    p_img = p[:-7] + '_0000.nii.gz'
    result = extractor.execute(load_final_image_root_path + p_img, load_final_label_root_path + p)

    # 保存特征
    feature = [value for key, value in result.items() if not 'diagnostics' in key]
    save_mat_by_dcm_base_sio(np.array([feature]), os.path.join(fold_root_path, p.split('.')[0] + '.mat'), 'Features')





