import pandas as pd
import os

def get_image_id(f):
    id = []
    for i in range(len(f)):
        id.append(f[i].split('_')[2])
    return id

def format_img_ids(prefix, data, path):
    path = prefix + data
    files = os.listdir(path)
    img_id = get_image_id(files)
    df_img = pd.DataFrame( {'file': files, 'location_id': img_id})
    return df_img

def add_qscore(prefix, df_img, study_id):
    # read in qscores.csv
    df = pd.read_csv(prefix + '006_place_pulse/place-pulse-2.0/place_pulse_meta/qscores.tsv', sep='\t')
    # studies = pd.read_csv(prefix + '006_place_pulse/place-pulse-2.0/place_pulse_meta/studies.tsv', sep='\t')
    df_safety = df[df['study_id'] == study_id]
    df_img.insert(0, 'trueskill.score', df_img['location_id'].map(df_safety.set_index('location_id')['trueskill.score'])) 
    return df_img

def scale_data(start, end, df_img):
    width = end - start
    max_ = df_img['trueskill.score'].max()
    min_ = df_img['trueskill.score'].min()
    df_img['trueskill.score_norm'] = df_img['trueskill.score'].apply(lambda x: ( x - min_) / (max_ - min_) * width + start)
    return df_img

def oversample(df_train):
    df_train['bins'] = df_train['trueskill.score_norm'].apply(lambda x: np.round(x))
    M = df_train['bins'].value_counts().max()
    frames = pd.DataFrame()
    for class_idx, group in df_train.groupby('bins'):
        oversample_class = group.sample(M-len(group), replace=True)
        frames = pd.concat([frames, oversample_class])
    print ('There were %s images in the original training set' % str(df_train.shape[0]))
    df_train = pd.concat([frames, df_train])
    print ('There are now %s images in the training dataset' % str(df_train.shape[0]) )  
    return df_train