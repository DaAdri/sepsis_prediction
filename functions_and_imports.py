import numpy as np
import pandas as pd

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample # équilibrage de classes

# Préparation des données pour entrainement
from sklearn.model_selection import train_test_split


# sauvegarde des hyperparamètres
import pickle

# Explicabilité
import shap



# models
from xgboost import XGBClassifier
import optuna # optimisation des hyper paramètres
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import LSTM, SimpleRNN, Conv1D, Conv2D, Flatten, MaxPooling2D, Dense, Dropout, LayerNormalization, Add, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Metrics
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample # équilibrage de classes
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
def load_data(filepath, drop_columns=None):
    """Charge les données depuis un fichier CSV et supprime les colonnes indésirées."""
    df = pd.read_csv(filepath)
    if drop_columns is not None:
        df = df.drop(columns=drop_columns)
    return df

def display_basic_info(df):
    """Affiche les informations de base sur le DataFrame, y compris sa forme, ses colonnes,
    un résumé descriptif, les valeurs manquantes par colonne et les premières lignes."""

    print("Shape of the DataFrame:", df.shape)
    print('\nNumber of unique patients:', df['Patient_ID'].nunique())

    if 'SepsisLabel' in df.columns:
        print("\nSepsisLabel class counts:\n", df['SepsisLabel'].value_counts())

    if 'will_have_sepsis' in df.columns:
        print("\nNumber of unique patients per class in 'will_have_sepsis':\n", df.groupby('will_have_sepsis')['Patient_ID'].nunique())

    print("\nColumns in the DataFrame:\n", df.columns)
    print("\nData Types:\n", df.dtypes)

    print("\nDescriptive Statistics:\n", df.describe())

    print("\nMissing Values Per Column:\n", df.isna().sum())

    print("\nFirst 5 Rows of the DataFrame:\n", df.head())


def sort_dataset_by_patient_and_hour(df, patient_id_column='Patient_ID', hour_column='Hour'):
    """
    Trie le DataFrame en fonction des colonnes spécifiées pour Patient_ID et Hour,
    assurant que les données sont triées d'abord par patient, puis par heure pour chaque patient.

    Args:
    df (DataFrame): Le DataFrame à trier.
    patient_id_column (str): Nom de la colonne contenant les identifiants des patients.
    hour_column (str): Nom de la colonne contenant les heures des prises de mesures.

    Returns:
    DataFrame: Un DataFrame trié selon les identifiants des patients et les heures.
    """
    sorted_df = df.sort_values(by=[patient_id_column, hour_column])
    return sorted_df

def aggregate_sepsis_label(df, patient_id_column='Patient_ID', sepsis_label_column='SepsisLabel'):
    """
    Agrège les données pour chaque patient pour déterminer si le patient a eu un sepsis
    à un moment quelconque et ajoute cette information dans une nouvelle colonne.

    Args:
    df (DataFrame): Le DataFrame contenant les données des patients.
    patient_id_column (str): Le nom de la colonne contenant les identifiants des patients.
    sepsis_label_column (str): Le nom de la colonne contenant les étiquettes de sepsis.

    Returns:
    DataFrame: Un DataFrame enrichi avec une colonne indiquant si le patient a eu un sepsis.
    """
    # Aggrégation des données par patient avec le maximum de SepsisLabel
    aggregated_df = df.groupby(patient_id_column)[sepsis_label_column].max().reset_index()

    # Renommer la colonne pour clarifier qu'il s'agit du résultat de l'aggrégation
    aggregated_df = aggregated_df.rename(columns={sepsis_label_column: 'will_have_sepsis'})

    # Joindre avec les données originales pour obtenir un DataFrame complet par patient
    aggregated_full_df = df.merge(aggregated_df, on=patient_id_column)

    return aggregated_full_df


def get_nbr_rows_per_patient_v2(df, time_window):
    """Récupère 6 lignes pour chaque patient.

    - Pour les patients avec 'will_have_sepsis' = 0, prend les  time_window premières lignes.
    - Pour les patients avec 'will_have_sepsis' = 1, prend les time_window/2  lignes avant et après le premier 'SepsisLabel' = 1.
    """

    # Liste pour stocker les résultats
    result = []

    # Boucler à travers chaque patient
    for patient_id, patient_data in df.groupby('Patient_ID'):
        # Vérifier la classe will_have_sepsis
        will_have_sepsis = patient_data['will_have_sepsis'].iloc[0]

        if will_have_sepsis == 0:
            # Prendre les 6 premières lignes pour les patients avec 'will_have_sepsis' = 0
            result.append(patient_data.head(time_window))
        else:
            # Pour les patients avec 'will_have_sepsis' = 1
            # Trouver la première ligne où 'SepsisLabel' = 1
            sepsis_start = patient_data[patient_data['SepsisLabel'] == 1]

            if not sepsis_start.empty:
                # Prendre les 6 lignes à partir de la première occurrence de 'SepsisLabel' = 1
                start_index = sepsis_start.index[0]
                result.append(patient_data.loc[start_index - time_window/2 :start_index + (time_window/2 - 1)])

    # Combiner toutes les parties en un seul DataFrame
    final_df = pd.concat(result)

    return final_df



def clean_data(df, interest_columns=None, missing_value_threshold=0.3):
    """Nettoie le DataFrame en supprimant les lignes avec trop de valeurs manquantes.
    Si 'interest_columns' n'est pas spécifié, toutes les colonnes sont prises en compte."""

    if interest_columns is None:
        interest_columns = df.columns.tolist()
    seuil = missing_value_threshold * len(interest_columns)
    cleaned_df = df.dropna(subset=interest_columns, thresh=len(interest_columns) - seuil)
    return cleaned_df


def check_nbr_rows_per_patient(df, count = 6):
    """Vérifie si tous les patients ont exactement count lignes."""

    # Compter le nombre de lignes par patient
    row_counts = df.groupby('Patient_ID').size()

    # Vérifier si tous les patients ont exactement 6 lignes
    all_have_six = (row_counts == count).all()

    if all_have_six:
        print("Tous les patients ont exactement 6 lignes.")
    else:
        print("Certains patients n'ont pas exactement 6 lignes.")
        # Afficher les patients qui n'ont pas count lignes
        print(row_counts[row_counts != count])

    return all_have_six

def remove_patients_without_nbr_rows(df, count = 6):
    """Supprime les patients qui n'ont pas exactement count lignes."""

    # Compter le nombre de lignes par patient
    row_counts = df.groupby('Patient_ID').size()

    # Trouver les patients qui ont exactement count lignes
    valid_patients = row_counts[row_counts == count].index

    # Filtrer le DataFrame pour ne garder que les patients valides
    filtered_df = df[df['Patient_ID'].isin(valid_patients)]

    return filtered_df

def add_time_to_sepsis_column(df):
    # Trouver le premier instant où chaque patient a SepsisLabel = 1
    first_sepsis_time = df[df['SepsisLabel'] == 1].groupby('Patient_ID')['Hour'].min()

    # Mapper ces temps de première sepsis sur les patients dans le DataFrame
    df['time_to_first_sepsis'] = df['Patient_ID'].map(first_sepsis_time)

    # Calculer time_to_sepsis comme la différence entre l'heure de la première sepsis et HospAdmTime
    # Note: Assurez-vous que 'HospAdmTime' est bien l'heure d'admission initiale pour chaque patient.
    # Si 'HospAdmTime' change par patient et enregistrement, cela devrait être ajusté en conséquence.
    df['time_to_sepsis'] = df['time_to_first_sepsis']  # - df['HospAdmTime']

    # Supprimer la colonne intermédiaire 'time_to_first_sepsis' si non nécessaire
    df.drop(columns=['time_to_first_sepsis'], inplace=True)

    return df

def balance_classes(df, target_column, method='undersample', random_state=123):
    """
    Équilibre les classes dans un DataFrame en sous-échantillonnant la classe majoritaire ou
    en sur-échantillonnant la classe minoritaire selon le paramètre 'method'.

    Args:
    df (DataFrame): Le DataFrame à équilibrer.
    target_column (str): Nom de la colonne contenant les étiquettes de classe.
    method (str): Méthode d'équilibrage, 'undersample' pour sous-échantillonnage ou 'oversample' pour sur-échantillonnage.
    random_state (int): Graine pour la génération de nombres aléatoires pour la reproductibilité.

    Returns:
    DataFrame: Un DataFrame où les classes sont équilibrées.
    """
    # Identifier les classes majoritaire et minoritaire
    class_counts = df[target_column].value_counts()
    major_class_label = class_counts.idxmax()
    minor_class_label = class_counts.idxmin()

    major_class = df[df[target_column] == major_class_label]
    minor_class = df[df[target_column] == minor_class_label]

    if method == 'undersample':
        # Sous-échantillonnage de la classe majoritaire
        resampled_major_class = resample(major_class,
                                         replace=False,
                                         n_samples=len(minor_class),
                                         random_state=random_state)
        balanced_df = pd.concat([resampled_major_class, minor_class])
    elif method == 'oversample':
        # Sur-échantillonnage de la classe minoritaire
        resampled_minor_class = resample(minor_class,
                                         replace=True,
                                         n_samples=len(major_class),
                                         random_state=random_state)
        balanced_df = pd.concat([major_class, resampled_minor_class])

    return balanced_df

def balance_data_by_sepsis_label(df):
    """
    Équilibre les données en sélectionnant le même nombre de patients ayant 0 et 1 comme valeur de 'will_have_sepsis'.

    Args:
    df (DataFrame): Le DataFrame contenant les données des patients.

    Returns:
    DataFrame: Un DataFrame équilibré avec un nombre égal de patients ayant 0 et 1 comme valeur de 'will_have_sepsis'.
    """
    # Séparer les patients ayant 0 et 1 comme valeur de 'will_have_sepsis'
    sepsis_positive = df[df['will_have_sepsis'] == 1]
    sepsis_negative = df[df['will_have_sepsis'] == 0]

    # Trouver le nombre minimal de patients dans les deux groupes
    min_count = min(len(sepsis_positive), len(sepsis_negative))

    # Échantillonner de manière aléatoire un nombre égal de patients de chaque groupe
    sepsis_positive_sample = sepsis_positive.sample(n=min_count, random_state=42)
    sepsis_negative_sample = sepsis_negative.sample(n=min_count, random_state=42)

    # Combiner les échantillons pour créer un DataFrame équilibré
    balanced_df = pd.concat([sepsis_positive_sample, sepsis_negative_sample])

    return balanced_df

def balance_classes_by_nan(df):
    """Rééquilibre les classes en prenant tous les patients qui auront le sepsis
    et en sélectionnant le même nombre de patients non atteints ayant le moins de NaN."""

    # Séparer les patients en fonction de 'will_have_sepsis'
    sepsis_patients = df[df['will_have_sepsis'] == 1]
    non_sepsis_patients = df[df['will_have_sepsis'] == 0]

    # Calculer le pourcentage de NaN pour chaque patient non atteint
    nan_percentage_per_patient = non_sepsis_patients.isna().mean(axis=1)

    # Associer le pourcentage de NaN à chaque 'Patient_ID' en utilisant .loc pour éviter l'avertissement
    non_sepsis_patients = non_sepsis_patients.copy()  # To avoid the warning
    non_sepsis_patients['nan_percentage'] = nan_percentage_per_patient

    # Sélectionner les patients non atteints avec le moins de valeurs manquantes
    selected_non_sepsis_patients = non_sepsis_patients.groupby('Patient_ID').mean().sort_values(by='nan_percentage').head(sepsis_patients['Patient_ID'].nunique()).index

    # Filtrer le DataFrame pour obtenir les patients non atteints sélectionnés
    balanced_non_sepsis_patients = non_sepsis_patients[non_sepsis_patients['Patient_ID'].isin(selected_non_sepsis_patients)]

    # Combiner avec les patients atteints de sepsis
    balanced_df = pd.concat([sepsis_patients, balanced_non_sepsis_patients])

    # Supprimer la colonne temporaire 'nan_percentage'
    balanced_df = balanced_df.drop(columns=['nan_percentage'], errors='ignore')

    return balanced_df



def filter_rows_by_time_to_sepsis(df, time_window = 24):
    """
    Filtre les lignes du DataFrame pour conserver uniquement celles où 'time_to_sepsis' est NaN,
    ou celles où 'Hour' est compris entre 'time_to_sepsis + HospAdmTime' et 'time_to_sepsis + HospAdmTime - 24'.

    Args:
    df (DataFrame): Le DataFrame contenant les données des patients.
    time_window (int) : fenêtre temporelle à récupérer

    Returns:
    DataFrame: Un DataFrame filtré avec les lignes désirées.
    """
    # Filtrer les lignes où 'time_to_sepsis' est NaN
    nan_time_to_sepsis_df = df[df['time_to_sepsis'].isna()]

    # Filtrer les lignes où 'Hour' est dans l'intervalle spécifié
    non_nan_time_to_sepsis_df = df.dropna(subset=['time_to_sepsis'])
    filtered_df = non_nan_time_to_sepsis_df[
        (non_nan_time_to_sepsis_df['Hour'] >= non_nan_time_to_sepsis_df['time_to_sepsis'] + non_nan_time_to_sepsis_df['HospAdmTime'] - time_window) &
        (non_nan_time_to_sepsis_df['Hour'] <= non_nan_time_to_sepsis_df['time_to_sepsis'] + non_nan_time_to_sepsis_df['HospAdmTime'])
    ]

    # Combiner les DataFrames filtrés
    combined_df = pd.concat([nan_time_to_sepsis_df, filtered_df])

    return combined_df


def filter_rows_by_time_to_sepsis_min_max(df, min_time=6, max_time=12):
    """
    Filtre les lignes du DataFrame pour conserver uniquement celles où 'time_to_sepsis' est NaN,
    ou entre un minimum et un maximum spécifié (inclus).

    Args:
    df (DataFrame): Le DataFrame contenant les données des patients.
    min_time (int): La valeur minimale de 'time_to_sepsis' pour conserver la ligne.
    max_time (int): La valeur maximale de 'time_to_sepsis' pour conserver la ligne.

    Returns:
    DataFrame: Un DataFrame filtré avec les lignes désirées.
    """
    # Filtrer le DataFrame pour conserver les lignes où 'time_to_sepsis' est NaN ou dans l'intervalle spécifié
    filtered_df = df[(df['time_to_sepsis'].isna()) |
                     ((df['time_to_sepsis'] >= min_time) & (df['time_to_sepsis'] <= max_time))]
    return filtered_df

def filter_rows_by_time_to_sepsis_min(df, min_time=6):
    """
    Filtre les lignes du DataFrame pour conserver uniquement celles où 'time_to_sepsis' est NaN,
    ou entre un minimum et un maximum spécifié (inclus).

    Args:
    df (DataFrame): Le DataFrame contenant les données des patients.
    min_time (int): La valeur minimale de 'time_to_sepsis' pour conserver la ligne.
    max_time (int): La valeur maximale de 'time_to_sepsis' pour conserver la ligne.

    Returns:
    DataFrame: Un DataFrame filtré avec les lignes désirées.
    """
    # Filtrer le DataFrame pour conserver les lignes où 'time_to_sepsis' est NaN ou dans l'intervalle spécifié
    filtered_df = df[(df['time_to_sepsis'].isna()) | (df['time_to_sepsis'] >= min_time)]
    return filtered_df

def add_sepsis_label_12(df):
    """
    Ajoute une colonne 'SepsisLabel_12' au DataFrame, qui a les mêmes valeurs que 'will_have_sepsis'
    sauf pour les lignes où 'Hour' < ('time_to_sepsis' - 12), qui auront la valeur 0, tout en gérant les valeurs NaN.

    Args:
    df (DataFrame): Le DataFrame contenant les données des patients.

    Returns:
    DataFrame: Le DataFrame avec la nouvelle colonne 'SepsisLabel_12'.
    """
    # Copier la colonne 'will_have_sepsis' dans 'SepsisLabel_12'
    df['SepsisLabel_12'] = df['will_have_sepsis']

    # Mettre à jour les valeurs de 'SepsisLabel_12' en fonction de la condition
    condition = (df['time_to_sepsis'].notna()) & (df['Hour'] < (df['time_to_sepsis'] - 12))
    df.loc[condition, 'SepsisLabel_12'] = 0

    return df


##################################################################################################
###################  Préparation de l'ensemble d'entrainement et de test #########################
#################################################################################################

def prepare_train_test(df, label_column, test_size=0.2, random_state=None, stratify=True):
    """
    Prépare les ensembles d'entraînement et de test à partir d'un DataFrame donné.

    Args:
    df (DataFrame): Le DataFrame à partir duquel les ensembles doivent être créés.
    label_column (str): Le nom de la colonne qui contient les étiquettes cibles.
    test_size (float): La proportion du dataset à inclure dans l'ensemble de test.
    random_state (int): Contrôle la reproductibilité des résultats en fixant un seed pour le générateur aléatoire.
    stratify (bool): Si True, les données sont divisées de façon à préserver le même pourcentage pour chaque classe cible dans les ensembles de train et de test.

    Returns:
    tuple: Contient les ensembles X_train, X_test, y_train, y_test.
    """
    # Séparation des features et des étiquettes
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Stratification optionnelle basée sur les étiquettes
    stratify_param = y if stratify else None

    # Répartition des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        stratify=stratify_param, random_state=random_state)

    # Affichage des dimensions des ensembles pour vérification
    print("X_train:", X_train.shape, "X_test:", X_test.shape, "y_train:", y_train.shape, "y_test:", y_test.shape)

    return X_train, X_test, y_train, y_test

def split_train_test_data(df, test_size=0.2, random_seed=42):
    """
    Sépare les données en ensembles d'entraînement et de test, en s'assurant que les patients
    avec et sans sepsis sont correctement répartis sans chevauchement entre les ensembles.

    Args:
    df (DataFrame): Le DataFrame contenant les données des patients.
    test_size (float): La proportion de chaque groupe de patients à utiliser pour le test.
    random_seed (int): La graine pour la génération de nombres aléatoires pour la reproductibilité.

    Returns:
    tuple: Un tuple contenant deux DataFrames, (train_df, test_df).
    """
    np.random.seed(random_seed)  # Pour la reproductibilité

    # Identifier les patients qui ont eu un sepsis
    patients_with_sepsis = df[df['will_have_sepsis'] == 1]['Patient_ID'].unique()

    # Sélectionner une proportion pour le test parmi les patients avec sepsis
    test_patients_with_sepsis = np.random.choice(patients_with_sepsis, size=int(len(patients_with_sepsis) * test_size), replace=False)

    # Identifier et sélectionner une proportion pour le test parmi les patients sans sepsis
    patients_without_sepsis = df[df['will_have_sepsis'] == 0]['Patient_ID'].unique()
    test_patients_without_sepsis = np.random.choice(patients_without_sepsis, size=int(len(patients_without_sepsis) * test_size), replace=False)

    # Combiner les patients de test
    test_patients = np.concatenate((test_patients_with_sepsis, test_patients_without_sepsis))

    # Créer les ensembles de données
    train_df = df[~df['Patient_ID'].isin(test_patients)]
    test_df = df[df['Patient_ID'].isin(test_patients)]

    return train_df, test_df

from sklearn.model_selection import GroupShuffleSplit

def split_train_test_data_v2(df, test_size=0.2, random_seed=42):
    """
    Sépare les données en ensembles d'entraînement et de test, en évitant que les données d'un même patient soient dans les deux ensembles.

    Args:
    df (DataFrame): Le DataFrame contenant les données des patients.
    test_size (float): La proportion de chaque groupe de patients à utiliser pour le test.
    random_seed (int): La graine pour la génération de nombres aléatoires pour la reproductibilité.

    Returns:
    tuple: Un tuple contenant deux DataFrames, (train_df, test_df).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)

    # Obtenir les indices des ensembles d'entraînement et de test en groupant par 'Patient_ID'
    for train_idx, test_idx in gss.split(df, groups=df['Patient_ID']):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

    return train_df, test_df



###################################################################################################
##################### Préparation des données pour les Réseaux de Neurones ########################
###################################################################################################
def extract_sequences_and_labels(df, patient_col, time_col, feature_cols, label_col, sequence_length):
    """
    Extrait des séquences de caractéristiques et des étiquettes à partir du DataFrame groupé par patient.

    Args:
    - df (DataFrame): Le DataFrame contenant les données des patients.
    - patient_col (str): La colonne représentant l'identifiant du patient.
    - time_col (str): La colonne représentant l'heure ou le temps.
    - feature_cols (list): Les colonnes des caractéristiques à utiliser pour le modèle.
    - label_col (str): La colonne du label à prédire.
    - sequence_length (int): La longueur de la séquence à extraire (par ex. 6).

    Returns:
    - sequences (list): Une liste de tableaux numpy représentant les séquences de caractéristiques.
    - labels (list): Une liste d'étiquettes associées à chaque séquence.
    """
    sequences = []
    labels = []

    # Group by patient to get the data for each patient
    grouped = df.groupby(patient_col)

    for patient_id, patient_data in grouped:
        # Trier les données par temps
        patient_data = patient_data.sort_values(by=time_col)

        # Extraire les caractéristiques et les labels
        features = patient_data[feature_cols].values
        label = patient_data[label_col].values

        # Créer des séquences de longueur fixe
        for i in range(len(patient_data) - sequence_length + 1):
            sequences.append(features[i:i+sequence_length])
            labels.append(label[i+sequence_length-1])  # Label de la dernière heure de la séquence

    return np.array(sequences), np.array(labels)


def normalize_sequences_minmax(sequences):
    """
    Normalise chaque séquence avec MinMax scaling entre 0 et 1.

    Args:
    - sequences (np.array): Séquences à normaliser (samples, time_steps, features).

    Returns:
    - normalized_sequences (np.array): Séquences normalisées.
    """
    scaler = MinMaxScaler()

    # Reshape en 2D pour normaliser chaque feature de chaque séquence
    n_samples, n_timesteps, n_features = sequences.shape
    reshaped_sequences = sequences.reshape(-1, n_features)

    # Appliquer MinMaxScaler
    scaled_sequences = scaler.fit_transform(reshaped_sequences)

    # Reshape en 3D après normalisation
    normalized_sequences = scaled_sequences.reshape(n_samples, n_timesteps, n_features)

    return normalized_sequences, scaler


def extract_sequences_and_normalize(train_df, test_df, exclude_columns=['Patient_ID', 'Hour', 'SepsisLabel'], drop_column ='will_have_sepsis', label_column='SepsisLabel', id_column='Patient_ID', time_column='Hour', sequence_length=6):
    """
    Prépare les données d'entraînement et de test pour l'extraction de séquences et la normalisation.

    Args:
        train_df (DataFrame): Jeu de données d'entraînement.
        test_df (DataFrame): Jeu de données de test.
        exclude_columns (list): Colonnes à exclure de l'analyse des caractéristiques.
        label_column (str): Nom de la colonne des labels.
        id_column (str): Nom de la colonne identifiant les patients.
        time_column (str): Nom de la colonne représentant le temps.
        sequence_length (int): Longueur de chaque séquence (en nombre de points temporels).

    Returns:
        normalize_sequences_train (ndarray): Séquences normalisées pour l'entraînement.
        labels_train (ndarray): Labels pour l'entraînement.
        normalize_sequences_test (ndarray): Séquences normalisées pour le test.
        labels_test (ndarray): Labels pour le test.
        scaler (MinMaxScaler): Scaler utilisé pour la normalisation.
    """
    # Supprimer la colonne des labels pour les features
    train_df.drop(columns=drop_column, inplace=True)
    test_df.drop(columns=drop_column, inplace=True)

    # Sélectionner les colonnes de caractéristiques
    feature_cols = [col for col in train_df.columns if col not in exclude_columns]

    # Extraction des séquences
    sequences_train, labels_train = extract_sequences_and_labels(
        train_df, id_column, time_column, feature_cols, label_column, sequence_length
    )
    sequences_test, labels_test = extract_sequences_and_labels(
        test_df, id_column, time_column, feature_cols, label_column, sequence_length
    )

    # Normaliser les séquences d'entraînement et de test avec MinMaxScaler
    scaler = MinMaxScaler()
    normalize_sequences_train = scaler.fit_transform(
        sequences_train.reshape(-1, sequences_train.shape[2])
    ).reshape(sequences_train.shape)
    normalize_sequences_test = scaler.transform(
        sequences_test.reshape(-1, sequences_test.shape[2])
    ).reshape(sequences_test.shape)

    return normalize_sequences_train, labels_train, normalize_sequences_test, labels_test, scaler




##################################################################################################
########################### Fonctions pour le modèle XGBoost #####################################
##################################################################################################
def objective(trial, X, y, cv=7):
    """
    Fonction objectif pour l'optimisation d'hyperparamètres avec Optuna.

    Args:
    trial (optuna.trial): Un essai de Optuna pour suggérer les hyperparamètres.
    X (DataFrame): Features du dataset.
    y (Series): Étiquettes cibles du dataset.
    cv (int): Nombre de plis pour la validation croisée.

    Returns:
    float: La moyenne des scores de validation croisée pour les hyperparamètres suggérés.
    """
    # Hyperparamètres suggérés par Optuna
    learning_rate = trial.suggest_uniform('learning_rate', 0.01, 1)
    max_depth = trial.suggest_int('max_depth', 2, 15)
    n_estimators = trial.suggest_int('n_estimators', 50, 250)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    gamma = trial.suggest_uniform('gamma', 0.01, 5)
    subsample = trial.suggest_uniform('subsample', 0.01, 1)

    # Création et évaluation du modèle
    clf = XGBClassifier(learning_rate=learning_rate,
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                        min_child_weight=min_child_weight,
                        gamma=gamma,
                        subsample=subsample,
                        use_label_encoder=False,
                        eval_metric='logloss')

    # Calcul du score moyen sur les plis de validation croisée
    score = cross_val_score(clf, X, y, cv=cv)

    return np.mean(score)


def perform_hyperparameter_optimization(X, y, objective, n_trials=50, random_state=42):
    """
    Crée une étude Optuna pour optimiser les hyperparamètres d'un modèle de machine learning.

    Args:
    X (DataFrame): Les features d'entrée pour le modèle.
    y (Series): Les étiquettes cibles.
    objective (function): La fonction objective pour Optuna.
    n_trials (int): Le nombre de tentatives d'optimisation.
    random_state (int): Graine pour la génération de nombres aléatoires pour la reproductibilité.

    Returns:
    dict: Meilleurs hyperparamètres trouvés par l'étude Optuna.
    """
    # Création d'un objet study d'Optuna
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=random_state))

    # Lancement de l'optimisation
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    # Retourne les meilleurs paramètres trouvés
    return study.best_params, study

# sauvegarde des meilleurs hyperparamètres
def save_study(study, filename):
    with open(filename, 'wb') as f:
        pickle.dump(study, f)

# chargement des meilleurs hyperparamètres
def load_study(filename):
    with open(filename, 'rb') as f:
        study = pickle.load(f)
    return study

def train_and_save_xgboost_classifier(X_train, y_train, best_params, model_path):
    """
    Crée, entraîne et sauvegarde un modèle XGBoost avec des paramètres spécifiés.

    Args:
    X_train (DataFrame): Les features d'entraînement.
    y_train (Series): Les étiquettes cibles d'entraînement.
    best_params (dict): Dictionnaire contenant les meilleurs paramètres pour le modèle.
    model_path (str): Chemin du fichier où le modèle sera sauvegardé.

    Returns:
    XGBClassifier: Le modèle XGBoost entraîné.
    """
    xgbc = XGBClassifier(
        n_jobs=-1,  # Utiliser tous les processeurs disponible
        tree_method='hist',  # Utiliser 'hist' pour accélérer l'entraînement
        subsample=0.8,       # Sous-échantillonnage pour réduire le temps d'entraînement
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        min_child_weight=best_params['min_child_weight'],
        gamma=best_params['gamma'],
        #subsample=best_params['subsample'],
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Entraînement du modèle
    xgbc.fit(X_train, y_train)

    # Sauvegarde du modèle
    with open(model_path, 'wb') as file:
        pickle.dump(xgbc, file)

    return xgbc

def explain_model_predictions_with_shap(model, X_train, X_test):
    """
    Explique les prédictions d'un modèle XGBoost en utilisant SHAP.

    Args:
    model (XGBClassifier): Le modèle entraîné.
    X_train (DataFrame): Les données d'entraînement.
    X_test (DataFrame): Les données de test pour lesquelles les explications sont générées.

    Returns:
    None: Affiche les graphiques SHAP.
    """
    # Création d'un explainer SHAP
    explainer = shap.Explainer(model, X_train)

    # Calcul des valeurs SHAP
    shap_values = explainer(X_test)

    # Affichage du summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar")


def load_xgboost_classifier(model_path):
    """
    Charge un modèle XGBoost sauvegardé à partir d'un fichier.

    Args:
    model_path (str): Chemin du fichier où le modèle est sauvegardé.

    Returns:
    XGBClassifier: Le modèle XGBoost chargé.
    """
    with open(model_path, 'rb') as file:
        xgbc_loaded = pickle.load(file)

    return xgbc_loaded


##################################################################################################
######################### Fonctions d'Evaluation #################################################
##################################################################################################
def predict_and_evaluate(model, X_test, y_test):
    """
    Fait des prédictions avec un modèle donné et évalue les résultats.

    Args:
    model (XGBClassifier): Le modèle entraîné à utiliser pour les prédictions.
    X_test (DataFrame): Les features de test.
    y_test (Series): Les étiquettes cibles de test.

    Returns:
    str: Un rapport d'évaluation imprimable.
    """
    # Prédiction avec le modèle
    y_predicted = model.predict(X_test)

    # Génération du rapport d'évaluation
    evaluation_report = classification_report(y_test, y_predicted)

    return evaluation_report


def evaluate_model_performance(model, X_test, y_test, threshold=0.5):
    """
    Évalue les performances du modèle en affichant la précision, recall, accuracy, f1-score,
    matrice de confusion, et AUROC.

    Args:
    - model: Le modèle entraîné.
    - X_test: Les données d'entrée de test.
    - y_test: Les labels de test.
    - threshold: Le seuil de décision pour classer les prédictions (par défaut 0.5).
    """
    # Prédire les probabilités
    y_pred_proba = model.predict(X_test)

    # Binariser les prédictions en fonction du seuil
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)

    # Afficher les résultats
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Sepsis', 'Sepsis'], yticklabels=['No Sepsis', 'Sepsis'])
    plt.title('Matrice de Confusion')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Courbe ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    return y_pred_proba, y_pred

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur les données de test et affiche le rapport de classification et la matrice de confusion.

    Args:
    model (keras.Model): Le modèle LSTM à évaluer.
    X_test (np.array): Données de test.
    y_test (np.array): Étiquettes de test.
    """
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes))

    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, annot_kws={"size": 16})
    plt.title("Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return y_pred

def plot_training_history(history):
    """
    Affiche les courbes de précision et de perte pour les ensembles d'entraînement et de validation.

    Args:
    - history: Historique de l'entraînement du modèle (history object de Keras).
    """
    # Récupérer les données de l'historique
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(accuracy) + 1)

    # Tracé de la précision
    plt.figure(figsize=(14, 5))

    # Courbe d'accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, 'b-', label='Accuracy entraînement')
    plt.plot(epochs, val_accuracy, 'r-', label='Accuracy validation')
    plt.title('Accuracy - Entraînement vs Validation')
    plt.xlabel('Épochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Courbe de perte (loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Perte entraînement')
    plt.plot(epochs, val_loss, 'r-', label='Perte validation')
    plt.title('Perte - Entraînement vs Validation')
    plt.xlabel('Épochs')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
