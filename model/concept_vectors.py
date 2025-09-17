import os

import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import scipy.stats
import tensorflow as tf
import sklearn.linear_model
from sklearn.preprocessing import QuantileTransformer

from .architecture import neurite_classifier
from .data import enforce_4D, pattern_png_loader, convert_to_sobel

LABEL_KEYS = ['Genotype', 'Sex', 'Pattern']
MEASURES = ['Total Length', 'Branching Deviation', '# of Branches', 'Branching Density', 'Alignment', 'Alignment Deviation']
OUTPUTS = ['type_pred', 'sex_pred', 'pattern_pred']


def perform_quantile(data, seed=42):
    if isinstance(data, pd.Series):
        data = data.values
    data = data.reshape([-1, 1])
    n_quantiles = 1000 if data.shape[0] > 1000 else data.shape[0]
    qt = QuantileTransformer(n_quantiles=n_quantiles, random_state=seed)
    return qt.fit_transform(data)


def solve_regression(inputs, ys, min_score=0.0):
    all_scores = []
    all_dirs = []
    for i in range(ys.shape[-1]):
        y = ys[:, i]
        reg = sklearn.linear_model.Ridge(alpha=1.0,)
        reg.fit(inputs, y)
        trial_score = reg.score(inputs, y)
        direction = reg.coef_
        dir_norm = np.linalg.norm(direction)
        if dir_norm != 0 and trial_score > min_score:
            direction /= dir_norm
            all_scores.append(trial_score)
            all_dirs.append(direction)
        else:
            all_scores.append(0.0)
            all_dirs.append(np.zeros([inputs.shape[-1], ]))
    if len(all_scores) == 0:
        return None, None, False
    return np.stack(all_dirs, axis=1), np.array(all_scores)


def get_br(sens_scores, r2_scores, class_split=None, scale_br=False):
    """Compute the Br score for a given set of sensitivity scores and R2 scores

    Parameters
    ----------
    sens_scores : np.ndarray
        The computed sensitivity scores for a given model/dataset
    r2_scores : np.ndarray
        The computed correlation scores
    class_split : np.ndarray, optional
        A set of labels that can be used to subdivide the inputs, by default None
    scale_br : bool, optional
        Whether to scale in the Br scores by the overall maximum magnitude, by default False
    """
    total_mu = np.mean(sens_scores, axis=0)
    total_TCAV = np.mean(sens_scores > 0, axis=0)
    total_std = np.std(sens_scores, axis=0)
    total_Br = r2_scores * np.divide(total_mu, total_std, where=total_std != 0, out=np.zeros_like(r2_scores))

    scaler = np.max(np.abs(total_Br))
    class_scores = []
    for class_label in np.unique(class_split):
        class_items = class_split == class_label
        class_sens = sens_scores[class_items]
        class_mu = np.mean(class_sens, axis=0)
        class_std = np.std(class_sens, axis=0)
        class_TCAV = np.mean(class_sens > 0, axis=0)
        class_Br = r2_scores * np.divide(class_mu, class_std, where=class_std != 0, out=np.zeros_like(r2_scores))
        scaler = max(np.max(np.abs(class_Br)), scaler)
        class_scores.append([class_TCAV, class_Br, class_mu, class_std])

    if scale_br:
        class_scores = [[tcav, br / scaler, mu, std, np.zeros_like(std)] for [tcav, br, mu, std] in class_scores]
        return [[total_TCAV, total_Br / scaler, total_mu, total_std, r2_scores]] + class_scores
    else:
        class_scores = [[tcav, br, mu, std, np.zeros_like(std)] for [tcav, br, mu, std] in class_scores]
        return [[total_TCAV, total_Br, total_mu, total_std, r2_scores]] + class_scores


def indexif(data, idx):
    if hasattr(data, '__iter__'):
        return np.atleast_1d(data)[idx]
    else:
        return data


def evaluate_concepts(measure_path, validation_groups, cv_weights_paths, data_dir, output_path='Br_scores.xlsx'):
    """Evaluate the concepts using Br scores for the trained NeuriteNet models

    Parameters
    ----------
    measure_path : os.PathLike
        Path to the spreadsheet of manual measures
    validation_groups : list
        A list of lists where each sub-list is the test set paths for a given validation fold
    cv_weights_paths : list
        A list of paths to the model weights for each validation fold
    data_dir : os.PathLike
        The base directory for the images
    """
    # Gather paths only for images with measures
    df = pd.read_excel(measure_path, sheet_name='1 to 1')
    groups_to_check = [[] for i in range(5)]
    cv_lookup = {}
    for idx, group in enumerate(validation_groups):
        for path in group:
            path = os.path.basename(path)
            cv_lookup[path] = idx
    check_paths = []
    for row in df.itertuples():
        # NOTE: This assumes the path structure matchess "Full Pat vs. Unpat Image set"
        path = "{}/{} {} {} {}/{}.png".format(
            'Pattern' if row.Pattern else 'Unpatterned',
            row.Image_Set,
            'M' if row.Sex else 'F',
            'KO' if row.Genotype else 'WT',
            'Pattern' if row.Pattern else 'Blank',
            row.File_Name
        )
        if not os.path.exists(os.path.join(data_dir, path)):
            continue
        check_paths.append(os.path.join(data_dir, path))
        features = [getattr(row, key) for key in MEASURES]
        labels = [getattr(row, key) for key in LABEL_KEYS]
        groups_to_check[cv_lookup[path]].append([path, features, labels])

    # Get predictions for all models for all images
    model = neurite_classifier(out_classes=3, out_layers=1, dense_size=128, generic=False)
    results = []
    for cv_group in len(validation_groups):
        model.load_weights(cv_weights_paths[cv_group])
        for path in tqdm.tqdm(check_paths):
            image, tf_label = pattern_png_loader(path)
            image = convert_to_sobel(image)
            tf_label = np.squeeze(tf_label)
            type_pred, sex_pred, pattern_pred = np.squeeze(model(enforce_4D(image)))
            results.append({
                'path': os.path.basename(path),
                'cv_group': cv_group,
                'type': tf_label[0], 'sex': tf_label[1], 'pattern': tf_label[2],
                'type_pred': type_pred, 'sex_pred': sex_pred, 'pattern_pred': pattern_pred,
            })
    results_df = pd.DataFrame(results)
    results_df['type_correct'] = np.abs(results_df['type'] - results_df['type_pred']) < 0.5  # TODO - needed?
    results_df['sex_correct'] = np.abs(results_df['sex'] - results_df['sex_pred']) < 0.5
    results_df['pattern_correct'] = np.abs(results_df['pattern'] - results_df['pattern_pred']) < 0.5
    results_df.to_excel(os.path.join(data_dir, 'model_results.xlsx'))

    # Compute the Br scores for each concept/output/model
    feature_threshold = 0.2
    feature_confidence = 0.05
    model_layer = 'conv_flat/act'
    score_data = []
    for cv_group in len(validation_groups):
        model.load_weights(cv_weights_paths[cv_group])
        data = results_df[results_df['cv_group'] == cv_group]
        for output_idx, output in enumerate(OUTPUTS):
            relevent_measures = set([])
            for measure in MEASURES:
                quantile_data = perform_quantile(data[measure])
                r, p = scipy.stats.spearmanr(data[output], quantile_data)
                if isinstance(r, np.ndarray):
                    r = r.ravel()[0]
                if np.abs(np.round(r, 2)) >= feature_threshold and np.abs(p) <= feature_confidence:
                    relevent_measures.add(measure)
            relevent_measures = list(relevent_measures)
            if len(relevent_measures) == 0:
                print('No relevent measures found for {} in cv group {}'.format(output, cv_group))
                continue
            features = data[relevent_measures]
            labels = data[OUTPUTS]
            vectors = []
            predictions = []
            act_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(model_layer).output, model.outputs])
            for row in data.itertuples():
                image = pattern_png_loader(row.path)
                image = convert_to_sobel(image)
                image = enforce_4D(image)
                act_out, pred = act_model(image)
                act_out = tf.keras.layers.GlobalMaxPool2D()(act_out)
                predictions.append(np.squeeze(pred))
                vectors.append(tf.reshape(act_out, [-1]))
            vectors = np.stack(vectors, axis=0)
            predictions = np.stack(predictions, axis=0)
            concept_vectors, r2_scores = solve_regression(vectors, features.values)
            if np.sum(np.abs(r2_scores)) != 0:
                total_sens = []
                for row in data.itertuples():
                    image = pattern_png_loader(row.path)
                    image = convert_to_sobel(image)
                    image = enforce_4D(image)
                    with tf.GradientTape() as tape:
                        act_out, pred = act_model(image)
                        loss = pred[0][:, output_idx]
                    grad = tape.gradient(loss, act_out)
                    sens = tf.linalg.matmul(grad, concept_vectors)
                    sens = tf.keras.layers.GlobalMaxPool2d()(sens)
                    total_sens.append(np.squeeze(sens))
                total_sens = np.stack(total_sens, axis=0)
                br_scores = get_br(total_sens, r2_scores, class_split=data[output.lower()].values)
                for measure_idx, measure in enumerate(relevent_measures):
                    TCAV, Br, mu, std, r2 = [br_scores[0][i] for i in range(5)]
                    r2 = r2_scores[measure_idx]
                    row_data = {
                        'cv_group': cv_group,
                        'measure': measure,
                        'output': output,
                        'TCAV': indexif(TCAV, measure_idx),
                        'Br': indexif(Br, measure_idx),
                        'Mu': indexif(mu, measure_idx),
                        'Std': indexif(std, measure_idx),
                        'R2': indexif(r2, measure_idx)
                    }
                    score_data.append(row_data)
    score_df = pd.DataFrame(score_data)
    score_df = score_df.dropna(axis=0, how='any')
    score_df.to_excel(output_path)
