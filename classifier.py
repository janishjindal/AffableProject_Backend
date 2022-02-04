
import pandas as pd
import numpy as np
import pickle
import ast
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

"""**ML Model predictor**"""

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

"""
This function creates a Bag-of-Words for Col 'imageLabels'
:param pandas.DataFrame df : input dataframe of user-profile
:param float confidence_threshold : threshold for an image tag to be considered as a valid image tag
:return : Bag-of-Words of image label tags above threshold  
:rtype: list of str 
"""
def BoW_imageLabels(df, confidence_threshold=0.65):
    imageLabelBoWCol = []
    total_rows = len(df)
    for row in range(total_rows):
        try:
            imageLabelList = ast.literal_eval(df.imageLabels[row])
            bow = ""
            for tup in imageLabelList:
                if (tup[1] > confidence_threshold):
                    bow += tup[0] + " "
            imageLabelBoWCol.append(bow)
        except ValueError:
            # print("Encountered null")
            imageLabelBoWCol.append("")
            continue
        except:
            imageLabelBoWCol.append("")
            continue
    return imageLabelBoWCol


"""
This function checks if any person-like features exist in Col 'imageLabels'
:param pandas.DataFrame df : input dataframe on whose imageLabels Col we have check person-like features
:param float confidence_threshold : threshold for an image tag to be considered as a valid image tag
:param float personFeature_condifence : threshold for percent of image tags belonging to person-like features
:return : 2 lists:
      list1 (rtype:str) containining imageLabel tags above the specified confidence threshold 
      list2 (rtype: bool) specifying whether the imageLabels col has person-like features or not
"""


def CheckPersonFeatures_in_imageLabels(df, confidence_threshold=0.65, personFeature_condifence=0.0):
    ## list of person-like features extracted from imageLabels column using data-engineering
    personFeatures = {'neck', 'hair', 'eyelash', 'beard', 'human leg', 'face', 'facial expression', 'lip', 'chin',
                      'moustache', 'blond', 'human body', 'arm', 'human', 'waist', 'man', 'woman', 'wrist', 'finger',
                      'nose', 'person', 'leg', 'cheek', 'tongue', 'lady', 'muscle', 'forehead', 'shoulder', 'eye',
                      'boy', 'mouth', 'male', 'girl', 'skin', 'thigh', 'smile', 'chest', 'nail', 'head', 'hand',
                      'people', 'eyebrow', 'layered hair', 'hair coloring', 'brown hair', 'facial hair',
                      'hair accessory', 'thumb', 'jaw', 'hip'}
    total_rows = len(df)
    isPerson = []
    newDfColumn = []
    for row in range(total_rows):
        try:
            imageLabelList = ast.literal_eval(df.imageLabels[row])
            newImageLabelList = []
            for tup in imageLabelList:
                if (tup[1] >= confidence_threshold):
                    newImageLabelList.append(tup)
            pf_count = 0
            for tup in newImageLabelList:
                if (tup[0] in personFeatures):
                    pf_count += 1
            pf_percent = pf_count / len(newImageLabelList)
            newDfColumn.append(newImageLabelList)
            if (pf_percent > personFeature_condifence):
                isPerson.append(1)
            else:
                isPerson.append(0)
        except ValueError:
            print("in ValueError", file=sys.stderr)
            # print("Encountered null")
            newDfColumn.append([])
            isPerson.append(0)
            continue
        except ZeroDivisionError:
            print("in ZeroDivisionError", file=sys.stderr)
            # newImageList is empty => no image label with high confidence tag
            newDfColumn.append([])
            isPerson.append(0)
            continue
        except:
            print("in except", file=sys.stderr)
            newDfColumn.append([])
            isPerson.append(0)
            continue
    return newDfColumn, isPerson


"""
This function classifies if a given user-profile is brand or not
:param dict input_dict : user-profile from UI as dict 
:return : username(str) of profile and if it isBrand(bool)
:rtype : dict    
"""
def brand_influencer_classifier(input_dict):
    del input_dict['id']
    df = pd.DataFrame([input_dict])
    print(df, file=sys.stderr)
    total_rows = len(df)

    if (total_rows == 0):
        # empty input
        return {}

    df_input = df.copy()
    # Feature expansion from Col ImageLabels (isPerson)
    confidence_threshold = 0.65
    personFeature_condifence = 0.0
    df["newImageLabels"], df["isPerson"] = CheckPersonFeatures_in_imageLabels(df, confidence_threshold,
                                                                              personFeature_condifence)

    # Bag of Word (BoW) on ImageLabel tags
    df["imageLabelBoW"] = BoW_imageLabels(df, confidence_threshold)

    # Feature Selection : Removing unwanted cols based on domain knowledge
    df.drop(['biography', 'username', 'externalUrl', 'picture', 'cityId', 'latitude', 'longitude', 'zip',
             'contactPhoneNumber', 'imageLabels', 'newImageLabels'], axis=1, inplace=True)

    # Loading fitted CountVectorizer (using trained data) to apply CountVectorizer on ImageLabelBoW
    cv = load_model('AffableModels/cv_pkl.pickle')
    cv_fit = cv.transform(df['imageLabelBoW'])
    bow_df = pd.DataFrame(cv_fit.toarray(), columns=cv.get_feature_names_out())
    bow_df.rename(columns={'brand': 'brand_bow'}, inplace=True)

    # Handling Missing Values
    numerical_vars = ['mediaCount', 'followingCount', 'followerCount', 'usertagsCount', 'geoMediaCount',
                      'shoppablePostsCount', 'followingTagCount']

    # Loading training model to fill missing values
    df_train = load_model('AffableModels/df_train_pkl.pickle')
    for i in numerical_vars:
        df[i].fillna(df_train[i].mean(), inplace=True)
    categorical_vars = ['isBusiness', 'category', 'hasEmail', 'isPerson']
    for i in categorical_vars:
        df[i].fillna(df_train[i].mode()[0], inplace=True)

    # Concatenating CountVectorizer of ImageLabels with Original df
    df = pd.concat([df, bow_df], axis=1)
    df.drop('imageLabelBoW', axis=1, inplace=True)

    # One-hot encoding for categical vars
    categorical_vars = ['isBusiness', 'category', 'hasEmail', 'isPerson']

    # Loading all the trained One-hot-Encoders for categorical vars to fit the test data
    category_encoder = load_model('AffableModels/category_ohe_pkl.pickle')
    category_array = category_encoder.transform(df[['category']]).toarray()
    category_ohe_df = pd.DataFrame(category_array, columns=category_encoder.get_feature_names_out(['category']))

    isBusiness_encoder = load_model('AffableModels/isBusiness_ohe_pkl.pickle')
    isBusiness_array = isBusiness_encoder.transform(df[['isBusiness']]).toarray()
    isBusiness_ohe_df = pd.DataFrame(isBusiness_array, columns=isBusiness_encoder.get_feature_names_out(['isBusiness']))

    hasEmail_encoder = load_model('AffableModels/hasEmail_ohe_pkl.pickle')
    hasEmail_array = hasEmail_encoder.transform(df[['hasEmail']]).toarray()
    hasEmail_ohe_df = pd.DataFrame(hasEmail_array, columns=hasEmail_encoder.get_feature_names_out(['hasEmail']))

    isPerson_encoder = load_model('AffableModels/isPerson_ohe_pkl.pickle')
    isPerson_array = isPerson_encoder.transform(df[['isPerson']]).toarray()
    isPerson_ohe_df = pd.DataFrame(isPerson_array, columns=isPerson_encoder.get_feature_names_out(['isPerson']))

    # Expanding categorical var columns using One-Hot-Encoders and expanding input datafram
    ohe_df = pd.concat([category_ohe_df, isBusiness_ohe_df, hasEmail_ohe_df, isPerson_ohe_df], axis=1)
    df = pd.concat([df, ohe_df], axis=1)
    df.drop(categorical_vars, axis=1, inplace=True)

    # Loading saved XG Boost Model (used bin file format to support different version of XG Boost Classifiers)
    # Predicting using XG Boost ML Model (Original + No Scaling + No SVD) -> Best
    XGBoost_model = XGBClassifier()
    XGBoost_model.load_model('AffableModels/XGBoost.bin')

    # Generating prediction in the requisite dict format from the model predictions
    predictions = XGBoost_model.predict(df)
    pred_df = pd.DataFrame(predictions, columns=['Brand'])
    pred_df['Brand'] = pred_df['Brand'].astype(str)
    username_df = pd.DataFrame(df_input['username'], columns=['username'])
    username_df['username'] = username_df['username'].astype(str)
    output_df = pd.concat([username_df, pred_df], axis=1)
    output = {}
    for i in range(len(output_df)):
        output["username"] = output_df["username"][i]
        if (output_df["Brand"][i] == "1"):
            output["isBrand"] = "True"
        else:
            output["isBrand"] = "False"
    return output

if __name__ == '__main__':
    # Unit Testing --------
    input_dict = {}
    input_dict['id'] = 1
    input_dict['username'] = "janish"
    input_dict['biography'] = "newbie at angular"
    input_dict['mediaCount'] = 1
    input_dict['cityId'] = 1
    input_dict['latitude'] = 1
    input_dict['longitude'] = 1
    input_dict['zip'] = 1
    input_dict['contactPhoneNumber'] = 1
    input_dict['followingCount'] = 1
    input_dict['followerCount'] = 1
    input_dict['usertagsCount'] = 1
    input_dict['isBusiness'] = False
    input_dict['externalUrl'] = "www.myntra.com"
    input_dict['category'] = "actor"
    input_dict['geoMediaCount'] = 1
    input_dict['shoppablePostsCount'] = 1
    input_dict['followingTagCount'] = 1
    input_dict['picture'] = "www.google.com"
    input_dict['imageLabels'] = "[('neck', 0.95, 0.95), ('head', 0.72, 0.72)]"
    input_dict['hasEmail'] = True
    output_dict = brand_influencer_classifier(input_dict)
    print(output_dict)

