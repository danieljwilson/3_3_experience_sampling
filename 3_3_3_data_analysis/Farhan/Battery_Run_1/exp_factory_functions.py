import json
import pandas as pd
import pandas
import numpy
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys
import os
import re

from pandas.api.types import CategoricalDtype
import scipy.stats 
from math import exp
from itertools import product
from scipy.stats import norm
from scipy.stats.distributions import beta


def get_response(path, point) -> list[int]:
    '''
    Description: 
        Function gets the responses of a EXP Factory experiment in a list
    Input: 
        Relative File Path, Likert Scale Point
    Output: 
        Integer list of responses
    '''
    with open(path, 'r') as filey:
        content = json.load(filey)
    
    response = []
    for item in content:
        if 'value' in item:
            response.append(int(content[item]))
    return response


def get_subscale_scores(response: list[int], subscales_dict: dict) -> dict:
    '''
    Description: 
        Gets the subscale scores of a experiment
    Input: 
        List of responses, Dict of Subscale indices
    Output: 
        Dict of Subscale Scores
    '''
    index = []; subscale_scores = {}; counter = 0
    
    for key in subscales_dict:
        subscale_scores[key] = 0
        index = subscales_dict[key]
        for i in range(len(response)):
            if i in index:
                counter += 1
                subscale_scores[key] = subscale_scores[key] + response[i]
        subscale_scores[key] = subscale_scores[key]/counter
        counter = 0
    return subscale_scores


def get_id(path):
    ''' 
    Input: 
        Path of the ID file
    Output: 
        Name and Student ID
    '''
    with open(path, 'r') as filey:
        content = json.load(filey)

        name = ''
        student_id = ''

        for object in content:
            if '0' in object and 'value' in object:
                name = content[object]
            elif '1' in object and 'value' in object:
                student_id = content[object]
        return name, student_id

## Attention Network Task
class ATN:
    """
        Attention Network Task
    """
    def __init__(self, path, df= {}):
        with open(path, 'r') as f:
            content = json.load((f))
            results = json.loads(content['data'])
            self.df = pd.DataFrame.from_dict(results)

    def get_post_error_slow(self, df):
        """df should only be one subject's trials where each row is a different trial. Must have at least 4 suitable trials
        to calculate post-error slowing
        """
        index = [(j-1, j+1) for j in [df.index.get_loc(i) for i in df.query('correct == False and rt != -1').index] if j not in [0,len(df)-1]]
        post_error_delta = []
        for i,j in index:
            pre_rt = df.iloc[i]['rt']
            post_rt = df.iloc[j]['rt']
            if pre_rt != -1 and post_rt != -1 and df.iloc[i]['correct'] and df.iloc[j]['correct']:
                post_error_delta.append(post_rt - pre_rt) 
        if len(post_error_delta) >= 4:
            return numpy.mean(post_error_delta)
        else:
            return numpy.nan
    
    def calc_attention_network_task_DV(self, dvs = {}):
        df = self.df
        """ Calculate dv for attention network task: Accuracy and average reaction time
        
        There are two versions of this task. A full version, and a reduced version
        used for fMRI which is missing the central and nocue conditions, as well
        as the neutral flanker condition
        
        :return dv: dictionary of dependent variables
        :return description: descriptor of DVs
        """
        # add columns for congruency sequence effect
        df.insert(0,'flanker_shift', df.flanker_type.shift(1))
        df.insert(0, 'correct_shift', df.correct.shift(1))
        
        # post error slowing
        post_error_slowing = self.get_post_error_slow(df.query('exp_stage == "test"'))
        
        # subset df
        missed_percent = (df['rt']==-1).mean()
        df = df.query('rt != -1').reset_index(drop = True)
        df_correct = df.query('correct == True')
        
        # Calculate basic statistics - accuracy, RT and error RT
        dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
        dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
        dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
        dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
        dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
        dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
        dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
        
        # Get three network effects
        cue_rt = df_correct.groupby('cue').rt.median()
        flanker_rt = df_correct.groupby('flanker_type').rt.median()
        cue_acc = df.groupby('cue').correct.mean()
        flanker_acc = df.groupby('flanker_type').correct.mean()
        
        try:
            dvs['congruent_rt'] = {'value':  flanker_rt.loc['congruent'], 'valence': 'Pos'}
            dvs['incongruent_rt'] = {'value':  flanker_rt.loc['incongruent'], 'valence': 'Pos'}
            dvs['conflict_rt'] = {'value':  (flanker_rt.loc['incongruent'] - flanker_rt.loc['congruent']), 'valence': 'Neg'}
            dvs['conflict_acc'] = {'value':  (flanker_acc.loc['incongruent'] - flanker_acc.loc['congruent']), 'valence': 'Pos'}
        except KeyError:
            pass
        
        # fmri verison of this task does not have neutral condition (and others)
        if 'neutral' in flanker_rt:
            try:
                dvs['neutral_rt'] = {'value':  flanker_rt.loc['neutral'], 'valence': 'Pos'}
                dvs['alerting_acc'] = {'value':  (cue_acc.loc['nocue'] - cue_acc.loc['double']), 'valence': 'NA'}
                dvs['alerting_rt'] = {'value':  (cue_rt.loc['nocue'] - cue_rt.loc['double']), 'valence': 'Pos'}
            except KeyError:
                pass
            orienting_comparison = 'center' # which condition to contrast against spatial
        else:
            orienting_comparison = 'double' # which condition to contrast against spatial

        try:
            dvs['orienting_rt'] = {'value':  (cue_rt.loc[orienting_comparison] - cue_rt.loc['spatial']), 'valence': 'Pos'}
            dvs['orienting_acc'] = {'value':  (cue_acc.loc[orienting_comparison] - cue_acc.loc['spatial']), 'valence': 'NA'}
        except KeyError:
                pass

        # remove unnecessary cue dvs
        for key in list(dvs.keys()):
            if any(x in key for x in ['center','double', 'spatial', 'nocue']):
                del dvs[key]
                
        #congruency sequence effect
        congruency_seq_rt = df_correct.query('correct_shift == True').groupby(['flanker_shift','flanker_type']).rt.median()
        congruency_seq_acc = df.query('correct_shift == True').groupby(['flanker_shift','flanker_type']).correct.mean()
        
        try:
            seq_rt = (congruency_seq_rt['congruent','incongruent'] - congruency_seq_rt['congruent','congruent']) - \
                (congruency_seq_rt['incongruent','incongruent'] - congruency_seq_rt['incongruent','congruent'])
            seq_acc = (congruency_seq_acc['congruent','incongruent'] - congruency_seq_acc['congruent','congruent']) - \
                (congruency_seq_acc['incongruent','incongruent'] - congruency_seq_acc['incongruent','congruent'])
            dvs['congruency_seq_rt'] = {'value':  seq_rt, 'valence': 'NA'}
            dvs['congruency_seq_acc'] = {'value':  seq_acc, 'valence': 'NA'}
        except KeyError:
            pass
        
        description = """
        DVs for "alerting", "orienting" and "conflict" attention networks are of primary
        interest for the ANT task, all concerning differences in RT. 
        
        Alerting is defined as nocue - double cue trials. Positive values indicate the benefit of an alerting double cue. 
        
        Orienting is defined as center - spatial cue trials. Positive values indicate the benefit of a spatial cue. 
        
        Conflict is defined as incongruent - congruent flanker trials. Positive values indicate the benefit of congruent trials (or the cost of incongruent trials). 
        
        RT measured in ms and median. RT are used for all comparisons.
        """
        return dvs

class KeepTrackTask:
    def __init__(self, path):
        with open(path, 'r') as f:
            content = json.load((f))
            results = json.loads(content['data'])
            self.df = pd.DataFrame.from_dict(results)
    
    def keep_track_post(self):
        df = self.df
        for i,row in df.iterrows():
            if not pandas.isnull(row['responses']) and row['trial_id'] == 'response':
                response = row['responses']
                response = response[response.find('":"')+3:-2]
                response = re.split(r'[,; ]+', response)
                response = [x.lower().strip() for x in response]
                df.at[i,'responses'] = response

        if 'correct_responses' in df.columns:
            df.loc[:,'possible_score'] = numpy.nan
            df.loc[:,'score'] = numpy.nan
            subset = df[[isinstance(i,dict) for i in df['correct_responses']]]
            for i,row in subset.iterrows():
                targets = row['correct_responses'].values()
                response = row['responses']
                score = sum([word in targets for word in response])
                df.at[i,'score'] = score
                df.at[i,'possible_score'] = len(targets)
        return df

    def calc_keep_track_DV(self, dvs = {}):
        df = self.keep_track_post()
        """ Calculate dv for choice reaction time
        :return dv: dictionary of dependent variables
        :return description: descriptor of DVs
        """
        df = df.query('rt != -1').reset_index(drop = True)
        score = df['score'].sum()/df['possible_score'].sum()
        dvs['score'] = {'value': score, 'valence': 'Pos'} 
        description = 'percentage of items remembered correctly'  
        return dvs, description



def get_boredom_proneness_dvs(path):
    '''
    Measure: Boredome Proneness Scale
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    response = get_response(path, 0)
    return sum(response)/len(response)


def get_berlin_numeracy_score(path):
    """
    Berlin Numeracy Score
    """
    with open(path, 'r') as f:
        content = json.load((f))
        results = json.loads(content['data'])

        return_val = 0

        for i in range(2, 6):
            return_val = return_val + int(results[i]['response']['response'])
        
        return return_val/4

    
def get_bscs_dvs(path):
    '''
    Measure: Brief Self Control Scale
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    response = get_response(path, 5)
    return sum(response)/len(response)


def get_BSSS_score(path):
    '''
    Measure: Brief Sensation Seeking Scale
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    with open(path, 'r') as filey:
        content = json.load(filey)

        ## Index of the Subscales
        exp_seek = [0, 4]; boredom_sus = [1, 5]
        thrill = [2, 6]; disinhibition =  [3, 7]

        scores = []
        divisor = 0
        for object in content:
            if 'value' in object:
                scores.append(int(content[object]))
                divisor += 1
        
        bsss_subscales = {}
        for i in range(len(scores)):
            if i in exp_seek:
                if 'exp_seek' in bsss_subscales:
                    bsss_subscales['exp_seek'].append(scores[i])
                else:
                    bsss_subscales['exp_seek'] = [scores[i]]
            if i in boredom_sus:
                if 'boredom_sus' in bsss_subscales:
                    bsss_subscales['boredom_sus'].append(scores[i])
                else:
                    bsss_subscales['boredom_sus'] = [scores[i]]
            if i in thrill:
                if 'thrill' in bsss_subscales:
                    bsss_subscales['thrill'].append(scores[i])
                else:
                    bsss_subscales['thrill'] = [scores[i]]
            if i in disinhibition:
                if 'disinhibition' in bsss_subscales:
                    bsss_subscales['disinhibition'].append(scores[i])
                else:
                    bsss_subscales['disinhibition'] = [scores[i]]
        
        bsss_subscales['thrill'] = sum(bsss_subscales['thrill'])/len(bsss_subscales['thrill'])
        bsss_subscales['disinhibition'] = sum(bsss_subscales['disinhibition'])/len(bsss_subscales['disinhibition'])
        bsss_subscales['boredom_sus'] = sum(bsss_subscales['boredom_sus'])/len(bsss_subscales['boredom_sus'])
        bsss_subscales['exp_seek'] = sum(bsss_subscales['exp_seek'])/len(bsss_subscales['exp_seek'])
        return sum(scores)/divisor, bsss_subscales


def get_cognitive_estimation_test_scores(path):
    with open(path, 'r') as f:
        content = json.load((f))
        results = json.loads(content['data'])

    return_val = 0
    for i in range(2, len(results) - 1):
        return_val += int(results[i]['response']['response'])

    return return_val/9


def get_cognitive_ref_dv(path):
    '''
    Measure: Cognitive Reflection Task
    Input: 
        Path of the response file
    Output: 
        Correct Proportion, Intuitive Proportion
    '''
    with open(path, 'r') as filey:
        content = json.load(filey)
    
        correct = ['3', '15', '4', '29', '20', 'c']
        intuitive = ['6', '20', '9', '30', '10', 'b']
        CRT_responses = []
        for object in content:
            if 'value' in object:
                CRT_responses.append(content[object])

        correct_prop = 0;
        intuitive_prop = 0;
        divisor = 0;
        for i in range(len(CRT_responses)):
            if CRT_responses[i] == correct[i]:
                correct_prop += 1
            else:
                intuitive_prop +=1
            divisor += 1
        correct_prop = correct_prop/divisor
        intuitive_prop = intuitive_prop/divisor
        return correct_prop, intuitive_prop


## DVs are emotion, games, and stuff but subscales do not match that??
def get_competetiveness_index_score(path):
    '''
    Measure: Competitiveness Index Revised
    Input: 
        Path of the response file
    Output: 
        Mean of the response list, enjoyment of competition, Contentiousness
    '''
    response = get_response(path, 5)
    enjoyment_of_competition = []
    contentiousness = []

    for i in range(len(response)):
        if i < 9:
            enjoyment_of_competition.append(response[i])
        else:
            contentiousness.append(response[i])
    
    eoc_score = -100
    con_score = -100

    if len(enjoyment_of_competition) != 0:
        eoc_score = sum(enjoyment_of_competition)/len(enjoyment_of_competition)
    
    if len(contentiousness) != 0:
        con_score = sum(contentiousness)/len(contentiousness)
        
    return sum(response)/len(response), eoc_score, con_score


## 4 point likert ## 0-3
## Scores is sum and not mean // But we are taking mean for simplicity
## Scores multiplied by 2 for comparison in original paper
def get_DASS_scores(path):
    '''
    Measure: DASS
    Input: 
        Path of the response file
    Output: 
        Overall, Stress, Depression, Anxiety Scores
    '''
    response = get_response(path, 4)

    ## Getting depression score sum
    depression_subscale_dict = {'dysphoria': [12], 'hopelessness': [9], 'devaluation': [20], 'self_deprecation': [16],
                               'lack_of_interest': [15], 'anhedonia': [2], 'inertia': [4]}
    depression_subscale = get_subscale_scores(response, depression_subscale_dict)
    depression_index = [12, 9, 20, 16, 15, 2, 4]
    depression = 0
    for i in range(len(response)):
        if i in depression_index:
            depression += response[i]

    ## Getting anxiety score sum
    anxiety_subscale_dict = {'autonomic_arousal': [1, 3, 18], 'skeletal_musculature_effects': [6], 
                             'situational_anxiety': [8, 14], 'anxious_effect': [19]}
    anxiety_index = [1, 3, 18, 6, 8, 14, 19]
    anxiety = 0
    for i in range(len(response)):
        if i in anxiety_index:
            anxiety += response[i]

    ## Getting Stress score sum
    stress_subscale_dict = {'difficulty_relaxing': [0, 11], 'nervous_arousal': [7], 'easily_upset': [10],
                           'irritable': [4, 17], 'impatient': [13]}
    stress_index = [0, 11, 7, 10, 4, 17, 13]
    stress = 0
    for i in range(len(response)):
        if i in stress_index:
            stress += response[i]

    ## Return Final DASS scores
    ## depression = depression*2; anxiety = anxiety*2; stress = stress*2
    
    return sum(response)/len(response), stress/7, depression/7, anxiety/7


def get_ambition_score(path):
    '''
    Measure: five Item Ambition Scale
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    response = get_response(path, 5)
    return sum(response)/len(response)
    

def get_effort_avoidance_threshold(path):
    with open(path, 'r') as f:
        content = json.load(f)

        difficulty = []
        rewardsDict = {}

        counter = 0
        keys = list(content.keys())
        for i in range(len(keys)):
            if 'difficulty' in keys[i]:
                difficulty.append(content[keys[i]])
            if 'reward' in keys[i]:
                rewardsDict[counter] = content[keys[i]]
                counter += 1

        sortedRewards = []
        temp = sorted(rewardsDict.items(), key=lambda x:x[1]) 
        for item in temp:
            sortedRewards.append(item[0])

        for i in range(len(sortedRewards)):
            if i == 0:
                continue
            if difficulty[sortedRewards[i - 1]] == 'hard':
                return (rewardsDict[sortedRewards[i - 1]])

            if difficulty[sortedRewards[i - 1]] == 'easy' and difficulty[sortedRewards[i]] == 'hard':
                return rewardsDict[sortedRewards[i]]


def get_future_time_perspective_scores(path):
    with open(path, 'r') as f:
        content = json.load(f)
        responses = []
        for item in content:
            if 'value' in item:
                responses.append(int(content[item]))
        return sum(responses)/len(responses)
    

def get_gses(path):
    '''
    Measure: General Self Efficacy Scale
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    response = get_response(path, 4)
    return sum(response)/len(response)
    
    
def calc_go_nogo_DV(df, dvs = {}):
    """ Calculate dv for go-nogo task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    dvs['acc'] = {'value': df['correct'].mean(), 'valence': 'Pos'} 
    dvs['go_acc'] = {'value': df[df['condition'] == 'go']['correct'].mean(), 'valence': 'Pos'} 
    dvs['nogo_acc'] = {'value': df[df['condition'] == 'nogo']['correct'].mean(), 'valence': 'Pos'} 
    dvs['go_rt'] ={'value':  df[(df['condition'] == 'go') & (df['rt'] != -1)]['rt'].median(), 'valence': 'Pos'} 
    # calculate Dprime, adjusting extreme values using the fourth suggestion from 
    # Stanislaw, H., & Todorov, N. (1999). Calculation of signal detection theory measures.
    hit_rate = min(df.query('condition == "go"').correct.mean(), 1-(.5/len(df)))
    FA_rate = max((1-df.query('condition == "nogo"').correct.mean()), .5/len(df))
    dprime = norm.ppf(hit_rate) - norm.ppf(FA_rate)
    bias = -.5 * (norm.ppf(hit_rate)+norm.ppf(FA_rate))
    dvs['dprime'] = {'value': dprime, 'valence': 'Pos'}
    dvs['bias'] = {'value': bias, 'valence': 'NA'}

    dvs['commission_errors'] = {'value': 1-df.query('condition == "nogo"').correct.mean(), 'valence':'Neg'}
    dvs['omission_errors'] = {'value': 1-df.query('condition == "go"').correct.mean(), 'valence':'Neg'}

    description = """
        Calculated accuracy for go/stop conditions. 75% of trials are go. D_prime is calculated as the P(response|go) - P(response|nogo)
    """
    return dvs, description


def get_go_nogo_dvs(path):
    '''
    Measure: Go Nogo
    Input: 
        Path of the response file
    Output: 
        Dataframe containing the Go NoGo Dvs
    '''
    with open(path, 'r') as filey:
        content = json.load(filey)
    
        # The the required key value pairs in a new dict
        # Loading portion of an existing dictionary, so use 'loads'
        results = json.loads(content['data'])

        # Convert into a dataframe from a dictionary
        df = pd.DataFrame.from_dict(results)
        go_nogo_dvs, description = calc_go_nogo_DV(df)
        return go_nogo_dvs
    
    
## Grit scale
def get_grit_scale_score(path):
    '''
    Measure: Grit Scale
    Input: File Path
    Output: Overall score, Perseverence of Effort Score, Conflict of Interest Score
    '''
    response = get_response(path, 5)
    CI_index = [0, 2, 4, 5]
    CI = 0; PE = 0

    for i in range(len(response)):
        if i in CI_index:
            CI += response[i]
        else:
            PE += response[i]
        
    CI = CI/4; PE = PE/4
    return (PE + CI)/2, PE, CI
   
   
## Holt Laury Risk Aversion
## DV is the number of safe choices
## Safe choice is the first one in each pair
def get_risk_aversion_score(path):
    '''
    Measure: Holt Laury Risk Aversion
    Input: 
        Path of the response file
    Output: 
       Number of Safe Choices made by the participant
    '''
    response = get_response(path, 2)

    ## Just counting the number of safe choices
    counter = 0
    for i in response:
        if i != 1:
            counter += 1
    return counter
    

def get_leisure_time_activity_results(path):
    with open(path, 'r') as f:
        content = json.load(f)
        responses = []
        for item in content:
            if 'value' in item:
                responses.append(int(content[item]))
        return sum(responses)/len(responses)

## Life Orientation Test
## 5 point Likert
def get_lot_score(path):
    '''
    Measure: Life orientation Test
    Input: 
        Path of the response file
    Output: 
        Mean of the response list (except fillers)
    '''
    fillers = [1, 4, 5, 7]
    ## reverse_index = [2, 6, 8]
    response = get_response(path, 5)

    lot_score = 0
    divisor = 0
    for i in range(len(response)):
        if i not in fillers:
            divisor += 1
            lot_score += response[i] 
    return lot_score/divisor
    
    
## Maximizing Scale
## 5 point likert 
## Get the global score
def get_maximizing_scale_results(path):
    '''
    Measure: Maximizing Scale
    Input: 
        Path of the response file
    Output: 
        Dictionary of Subscale scores, Mean of the response list
        Subscales: Alternative search, decision difficulty, high standards, MTS, SMTS
    '''
    ## reverse_index = []
    response = get_response(path, 5)

    subscales_dict = {'alternative_search': [0, 1, 2, 3, 4, 5], 'decision_difficulty': [6, 7, 8, 9], 
                      'high_standards': [10, 11, 12], 'MTS': [10, 11, 12, 13, 14, 15, 16, 17, 18],
                      'SMTS': [11, 12, 13, 14, 15, 16, 17]}
    subscale_scores = get_subscale_scores(response, subscales_dict)
    return subscale_scores, sum(response)/len(response)
    

def get_mindfulness_attention_score(path):
    responses = get_response(path, 5)
    return sum(responses)/len(responses)


def get_need_for_cognition_score(path):
    '''
    Measure: 
        Need for Cognition Scale (18 Items)
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    ## reverse_index = [0, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 17, 18]
    response = get_response(path, 5)
    return sum(response)/len(response)
    
    
## Perceived stress scale
## 5 point likert
def get_pss_score(path):
    '''
    Measure: 
        Perceived Stress Scale
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    ## reverse_index = [3, 4, 6, 7]
    response = get_response(path, 5)
    return sum(response)/len(response)
    
    
## Plus Minus Task
def get_plus_minus_dv(path):
    '''
    Measure: Plus Minus Task
    Input: 
        Path of the response file
    Output: 
        Differene between (mean reaction time for add/sub task) - rt of alternate tasks
    '''
    with open(path, 'r') as filey:
        content = json.load(filey)

        results = json.loads(content['data'])

        ## Convert into a dataframe from a dictionary
        df = pd.DataFrame.from_dict(results)

        # Clean out the instructions, practice, and end trials
        df = df[df['exp_stage'] == 'test']

        ## Using reaction time
        pmt_results = {}
        pmt_results['add'] = df['rt'].values[0]
        pmt_results['sub'] = df['rt'].values[1]
        pmt_results['alt'] = df['rt'].values[2]
        pmt_score = (pmt_results['add'] + pmt_results['sub'])/2 - pmt_results['alt']
        return pmt_score
        
        
def get_positive_negative_affect_score(path):
    '''
    Measure: 
        Positive and Negative Affect Schedule-Short-Form (PANAS-Short)
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    ## reverse_index = []
    response = get_response(path, 5)
    subscales_dict = {'positive_affect': [0, 1, 2, 3, 4], 'negative_affect': [5, 6, 7, 8, 9]}
    subscales_scores = get_subscale_scores(response, subscales_dict)
    return subscales_scores
  
  
## Probabilistic Selection Task
## Used Ian Eisenberg's DV function
## Erased additional literature review DVs
## Ask if manipulation checks necessary
def calc_probabilistic_selection_DV(df, dvs = {}):
    """ Calculate dv for probabilistic selection task
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # define helper functions
    def get_value_diff(lst, values):
        lst = [int(i) for i in lst]
        return values[lst[0]] - values[lst[1]]
    def get_value_sum(lst,values):
        lst = [int(i) for i in lst]
        return values[lst[0]] + values[lst[1]]
    
    # convert stim chosen to int
    missed_percent = (df['rt']==-1).mean()
    df = df[df['rt'] != -1].reset_index(drop = True)
    
    #Calculate regression DVs
    train = df.query('exp_stage == "training"')
    values = train.groupby('stim_chosen')['feedback'].mean()
    # fill in values if the subject never selected that stimulus
    for v in [20,30,40,60,70,80]:
        if v not in values.index:
            values.loc[v] = v/100.0
        
    df.loc[:,'value_diff'] = df['condition'].apply(lambda x: get_value_diff(x.split('_'), values) if x==x else numpy.nan)
    df.loc[:,'value_sum'] = df['condition'].apply(lambda x: get_value_sum(x.split('_'), values) if x==x else numpy.nan)  
    df.loc[:, 'choice'] = df['key_press'].map(lambda x: ['right','left'][x == 37]).astype(CategoricalDtype(categories = ['left','right']))
    df.loc[:,'choice_lag'] = df['choice'].shift(1)
    test = df.query('exp_stage == "test"')
    rs = smf.glm(formula = 'choice ~ value_diff*value_sum - value_sum + choice_lag', data = test, family = sm.families.Binomial()).fit()
    
    dvs['value_sensitivity'] = {'value':  rs.params['value_diff'], 'valence': 'Pos'} 
    dvs['positive_learning_bias'] = {'value':  rs.params['value_diff:value_sum'], 'valence': 'NA'}
    dvs['log_ll'] = {'value':  rs.llf, 'valence': 'NA'}
    dvs['num_trials'] = {'value':  test.shape[0], 'valence': 'Pos'} 
    dvs['overall_test_acc'] = {'value':  test['correct'].mean(), 'valence': 'Pos'} 
    dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'} 

    description = """
        The primary DV in this task is whether people do better choosing
        positive stimuli or avoiding negative stimuli. Two different measurements
        are calculated. The first is a regression that predicts participant
        accuracy based on the value difference between the two options (defined by
        the participant's actual experience with the two stimuli) and the sum of those
        values. A significant effect of value difference would say that participants
        are more likely to be correct on easier trials. An interaction between the value
        difference and value-sum would say that this effect (the relationship between
        value difference and accuracy) differs based on the sum. A positive learning bias
        would say that the relationship between value difference and accuracy is greater 
        when the overall value is higher.
        
        Another way to calculate a similar metric is to calculate participant accuracy when 
        choosing the two most positive stimuli over other novel stimuli (not the stimulus they 
        were trained on). Negative accuracy can similarly be calculated based on the 
        probability the participant avoided the negative stimuli. Bias is calculated as
        their positive accuracy/negative accuracy. Thus positive values indicate that the
        subject did better choosing positive stimuli then avoiding negative ones. 
        Reference: http://www.sciencedirect.com/science/article/pii/S1053811914010763
    """
    return dvs, description


def get_probabilistic_selc_dv(path):
    with open(path, 'r') as filey:
        content = json.load(filey)

        # The the required key value pairs in a new dict
        # Loading portion of an existing dictionary, so use 'loads'
        results = json.loads(content['data'])

        # Convert into a dataframe from a dictionary
        df = pd.DataFrame.from_dict(results)
        probabilistic_selection_dvs, description = calc_probabilistic_selection_DV(df)
        return probabilistic_selection_dvs
        
      
## Propensity to plan
## Six Point Likert
def get_propensity_to_plan_score(path):
    ## reverse_index = []
    response = get_response(path, 6)

    money_short_run = []; money_long_run = []
    time_short_run = []; time_long_run = []

    for i in range(len(response)):
        if 0 <= i < 6:
            money_short_run.append(response[i])
        elif 6 <= i < 12:
            money_long_run.append(response[i])
        elif 12 <= i < 18:
            time_short_run.append(response[i])
        elif i >= 18:
            time_long_run.append(response[i])

    msr_score = sum(money_short_run)/len(money_short_run)
    mlr_score = sum(money_long_run)/len(money_long_run)
    tsr_score = sum(time_short_run)/len(time_short_run)
    
    if len(time_long_run) == 0:
        tlr_score = 'Error'
    else:
        tlr_score = sum(time_long_run)/len(time_long_run)
    return msr_score, mlr_score, tsr_score, tlr_score
    

def get_psychomotor_vigilance(path):
    with open(path, 'r') as f:
        content = json.load(f)
        responses = list(content.values())
        
        for i in range(len(responses)):
            responses[i] = int(responses[i])
        return sum(responses)/len(responses)


## Ravens Advanced Progressive Matrices
## Get Average for this one as well
class Ravens:
    def __init__(self, path):
        with open(path, 'r') as f:
            content = json.load(f)
            results = json.loads(content['data'])
            self.df = pd.DataFrame.from_dict(results)
    
    def ravens_post(self):
        df = self.df
        # label trials
        practice_questions = df.stim_question.map(lambda x: 'practice' in x if not pandas.isnull(x) else False) 
        practice_question_index = practice_questions[practice_questions].index

        test_questions = df.stim_question.map(lambda x: 'practice' not in x and 'bottom' in x if not pandas.isnull(x) else False)    
        test_question_index = test_questions[test_questions].index

        for item in practice_question_index.to_list():
            df.at[item,'exp_stage'] = 'practice'
            df.at[item,'trial_id'] = 'question'
        
        for item in test_question_index.to_list():
            df.at[item,'exp_stage'] = 'test'
            df.at[item,'trial_id'] = 'question'

        # relabel trial nums
        df.loc[test_question_index,'trial_num'] += 2

        indices_list = (df[df.stim_question.map(lambda x: 'practice_bottom_2' in x if  not pandas.isnull(x) else False)].index).to_list()
        for item in indices_list:
            df.at[item, 'trial_num'] =  1

        # score questions
        correct_responses = {0: 'C', 1: 'F', 2: 'B', 3: 'E', 4: 'G', 5: 'B', 6: 'C', 7: 'B',
                            8: 'E', 9: 'B', 10: 'B', 11: 'E', 12: 'A', 13: 'E',
                            14: 'A', 15: 'C', 16: 'B', 17: 'E', 18: 'F', 19: 'D'}   
        df.insert(0,'correct_response',df.trial_num.replace(correct_responses))
        df.insert(0,'correct', df.stim_response == df.correct_response)
        df.drop(['qnum', 'score_response', 'times_viewed', 'response_range'], axis = 1, inplace = True)
        return df


    def calc_ravens_DV(self, dvs = {}):
        df = self.ravens_post()
        """ Calculate dv for ravens task
        :return dv: dictionary of dependent variables
        :return description: descriptor of DVs
        """
        df = df.query('exp_stage == "test" and trial_id == "question"').reset_index(drop = True)
        dvs['score'] = {'value':  df['correct'].sum(), 'valence': 'Pos'} 
        description = 'Score is the number of correct responses out of 18'
        return dvs, description  
    
def get_rosenberg_SES_score(path):
    '''
    Measure: Rosenberg Self Esteem Scale
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    ## reverse_index = [1, 4, 5, 7, 8]
    response = get_response(path, 4)
    return sum(response)/len(response)
    
    
## Secular Measure of work ethic
## Return overall scores as well
def get_secular_measure_work_ethic_score(path):
    response = get_response(path, 6)

    cli_index =[2, 3, 7, 8, 9]
    moral_index = [1, 4, 6] 
    iwm_index = [0, 5]

    ## No reverse scoring
    cli_score = 0; moral_score = 0; iwm_score = 0

    for i in range(len(response)):
        if i in cli_index:
            cli_score += response[i]
        elif i in moral_index:
            moral_score += response[i]
        else:
            iwm_score += response[i]
    scores = {'cli_score': cli_score/5, 'moral_score': moral_score/3, 'iwm_score': iwm_score/2}

    overall = -1000
    if len(response) != 0:
        overall = sum(response)/len(response)
    return scores, overall
    

class Simon:
    def __init__(self, path):
        with open(path, 'r') as f:
            content = json.load(f)
            results = json.loads(content['data'])
            self.df = pd.DataFrame.from_dict(results)

    def get_post_error_slow(self, df):
        """df should only be one subject's trials where each row is a different trial. Must have at least 4 suitable trials
        to calculate post-error slowing
        """
        index = [(j-1, j+1) for j in [df.index.get_loc(i) for i in df.query('correct == False and rt != -1').index] if j not in [0,len(df)-1]]
        post_error_delta = []
        for i,j in index:
            pre_rt = df.iloc[i]['rt']
            post_rt = df.iloc[j]['rt']
            if pre_rt != -1 and post_rt != -1 and df.iloc[i]['correct'] and df.iloc[j]['correct']:
                post_error_delta.append(post_rt - pre_rt) 
        if len(post_error_delta) >= 4:
            return numpy.mean(post_error_delta)
        else:
            return numpy.nan
        
    def simon_post(self):
        df = self.df
        df.loc[:,'correct'] = df['correct'].astype(float)
        subset = df[df['trial_id']=='stim']
        condition = (subset.stim_side.map(lambda x: 37 if x=='left' else 39) == subset.correct_response).map \
                    (lambda y: 'congruent' if y else 'incongruent')
        df.loc[subset.index,'condition'] =  condition
        return df

    def calc_simon_DV(self, dvs = {}):
        df = self.simon_post()
        """ Calculate dv for simon task. Incongruent-Congruent, median RT and Percent Correct
        :return dv: dictionary of dependent variables
        :return description: descriptor of DVs
        """
        # add columns for congruency sequence effect
        df.insert(0,'condition_shift', df.condition.shift(1))
        df.insert(0, 'correct_shift', df.correct.shift(1))
        
        # post error slowing
        post_error_slowing = self.get_post_error_slow(df)
        
        # subset df
        missed_percent = (df['rt']==-1).mean()
        df = df.query('rt != -1').reset_index(drop = True)
        df_correct = df.query('correct == True').reset_index(drop = True)
        
        # Calculate basic statistics - accuracy, RT and error RT
        dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
        dvs['avg_rt_error'] = {'value':  df.query('correct == False').rt.median(), 'valence': 'NA'}
        dvs['std_rt_error'] = {'value':  df.query('correct == False').rt.std(), 'valence': 'NA'}
        dvs['avg_rt'] = {'value':  df_correct.rt.median(), 'valence': 'Neg'}
        dvs['std_rt'] = {'value':  df_correct.rt.std(), 'valence': 'NA'}
        dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
        dvs['post_error_slowing'] = {'value':  post_error_slowing, 'valence': 'Pos'}
        
        # Get congruency effects
        rt_contrast = df_correct.groupby('condition').rt.median()
        acc_contrast = df.groupby('condition').correct.mean()
        dvs['simon_rt'] = {'value':  rt_contrast['incongruent']-rt_contrast['congruent'], 'valence': 'Neg'} 
        dvs['simon_acc'] = {'value':  acc_contrast['incongruent']-acc_contrast['congruent'], 'valence': 'Pos'} 
        
        
        #congruency sequence effect
        congruency_seq_rt = df_correct.query('correct_shift == True').groupby(['condition_shift','condition']).rt.median()
        congruency_seq_acc = df.query('correct_shift == True').groupby(['condition_shift','condition']).correct.mean()
        
        try:
            seq_rt = (congruency_seq_rt['congruent','incongruent'] - congruency_seq_rt['congruent','congruent']) - \
                (congruency_seq_rt['incongruent','incongruent'] - congruency_seq_rt['incongruent','congruent'])
            seq_acc = (congruency_seq_acc['congruent','incongruent'] - congruency_seq_acc['congruent','congruent']) - \
                (congruency_seq_acc['incongruent','incongruent'] - congruency_seq_acc['incongruent','congruent'])
            dvs['congruency_seq_rt'] = {'value':  seq_rt, 'valence': 'NA'} 
            dvs['congruency_seq_acc'] = {'value':  seq_acc, 'valence': 'NA'} 
        except KeyError:
            pass

        dvs['congruent_acc'] = {'value': df.query('condition == "congruent"').correct.mean(), 'valence': 'Pos'}
        dvs['incongruent_acc'] = {'value': df.query('condition == "incongruent"').correct.mean(), 'valence': 'Pos'}
        dvs['congruent_avg_rt'] = {'value': df.query('condition == "congruent"').rt.median(), 'valence': 'Neg'}
        dvs['incongruent_avg_rt'] = {'value': df.query('condition == "incongruent"').rt.median(), 'valence': 'Neg'}
        dvs['congruent_sd_rt'] = {'value': df.query('condition == "congruent"').rt.std(), 'valence': 'NA'}
        dvs['incongruent_sd_rt'] = {'value': df.query('condition == "incongruent"').rt.std(), 'valence': 'NA'}
        
        description = """
            simon effect calculated for accuracy and RT: incongruent-congruent.
            RT measured in ms and median RT is used for comparison.
            """
        return dvs

class SimpleRT:
    def __init__(self, path):
        with open(path, 'r') as f:
            content = json.load(f)
            results = json.loads(content['data'])
            self.df = pd.DataFrame.from_dict(results)
    
    def calc_simple_RT_DV(self, dvs = {}):
        df = self.df
        """ Calculate dv for simple reaction time. Average Reaction time
            :return dv: dictionary of dependent variables
            :return description: descriptor of DVs
        """
        missed_percent = (df['rt']==-1).mean()
        df = df.query('rt != -1').reset_index(drop = True)
        dvs['avg_rt'] = {'value':  df['rt'].median(), 'valence': 'Pos'} 
        dvs['std_rt'] = {'value':  df['rt'].std(), 'valence': 'NA'} 
        dvs['missed_percent'] = {'value':  missed_percent, 'valence': 'Neg'}
        description = 'average reaction time'  
        return dvs, description
    
## Stroop Task
def get_stroop_dvs(path):
    with open(path, 'r') as filey:
        content = json.load(filey)

        # The the required key value pairs in a new dict
        # Loading portion of an existing dictionary, so use 'loads'
        results = json.loads(content['data'])

        # Convert into a dataframe from a dictionary
        df = pd.DataFrame.from_dict(results)

        # Clean out the instructions, practice, and end trials
        # df_clean = df.loc[df['trial_id'] == 'fixation' or df['trial_id'] == 'stim']
        df = df[df['exp_stage'] == 'test']


        # Get mean rt, var, sd for congruent trials
        congruent = df[df['condition']== 'congruent']
        con_avg = congruent['rt'].apply(pd.to_numeric).mean()
        con_sd = congruent['rt'].apply(pd.to_numeric).std()

        # Get mean rt, var, sd for incongruent trials
        incongruent = df[df['condition']== 'incongruent']
        incon_avg = incongruent['rt'].apply(pd.to_numeric).mean()
        incon_sd = incongruent['rt'].apply(pd.to_numeric).std()

        # Calculate stroop effect
        # Measured as the difference in mean rt between congruent and incongruent tasks
        stroop_effect = incon_avg - con_avg

        # Get Accuracy
        correct = df[df['correct'] == True]
        accuracy = (len(correct)/len(df))*100

        # Final results
        return {'stroop_effect': stroop_effect, 'stroop_accuracy': accuracy, 'stroop_con_avg': con_avg, 
                 'stroop_con_sd': con_sd, 'stroop_incon_avg': incon_avg, 'stroop_incon_sd': incon_sd}
            
            
def get_TEI_score(path):
    '''
    Measure: Trait Emotional Intelligence
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    ## reverse_index = [1, 3, 4, 6, 7, 9, 11, 12, 13, 15, 17, 21, 24, 25, 27]
    response = get_response(path, 7)
    return sum(response)/len(response)
    
    
def get_THC_score(path):
    '''
    Measure: Trait Hedonic Capacity
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    ## reverse_index = []
    response = get_response(path, 5)
    return sum(response)/len(response)
    

class TOL:
    def __init__(self, path):
        with open(path, 'r') as f:
            content = json.load(f)
            results = json.loads(content['data'])
            self.df = pd.DataFrame.from_dict(results)

    def TOL_post(self):
        df = self.df
        index = df.query('trial_id == "feedback"').index
        i_index = [df.index.get_loc(i)-1 for i in index]
        df.loc[index,'num_moves_made'] = df.iloc[i_index]['num_moves_made'].tolist()
        df.loc[index,'min_moves'] = df.iloc[i_index]['min_moves'].tolist()
        df.loc[:,'correct'] = df['correct'].astype(float)
        return df
    
    def calc_TOL_DV(self, dvs = {}):
        df = self.TOL_post()
        feedback_df = df.query('trial_id == "feedback"')

        reqVal = df.problem_id.dropna().drop([2, 3]).max() + 2

        analysis_df = pd.DataFrame(index = range(reqVal))
        analysis_df.loc[:,'optimal_solution'] = feedback_df.apply(lambda x: x['num_moves_made'] == x['min_moves'] and x['correct'] == True, axis = 1).tolist()
        analysis_df.loc[:,'planning_time'] = df.query('num_moves_made == 1 and trial_id == "to_hand"').groupby('problem_id').rt.mean()
        analysis_df.loc[:,'solution_time'] = df.query('trial_id in ["to_hand", "to_board"]').groupby('problem_id').rt.sum() - analysis_df.loc[:,'planning_time']
        analysis_df.loc[:,'num_moves_made'] = df.groupby('problem_id').num_moves_made.max()  
        analysis_df.loc[:,'extra_moves'] = analysis_df.loc[:,'num_moves_made'] - feedback_df.min_moves.tolist()
        analysis_df.loc[:,'avg_move_time'] = analysis_df['solution_time']/(analysis_df['num_moves_made']-1)
        analysis_df.loc[:,'correct_solution'] = feedback_df.correct.tolist()
        
        dvs['num_correct'] = {'value':  analysis_df['correct_solution'].sum(), 'valence': 'Pos'} 
        # When they got it correct, did they make the minimum number of moves?
        dvs['num_optimal_solutions'] = {'value':   analysis_df['optimal_solution'].sum(), 'valence': 'Pos'}
        # how many extra moves did they take over the entire task?
        dvs['num_extra_moves'] = {'value':   analysis_df['extra_moves'].sum(), 'valence': 'Neg'}
        # how long did it take to make the first move?    
        dvs['planning_time'] = {'value':  analysis_df['planning_time'].mean(), 'valence': 'NA'} 
        # how long did it take on average to take an action on correct trials
        dvs['avg_move_time'] = {'value':  analysis_df.query('correct_solution == True')['avg_move_time'].mean(), 'valence': 'NA'} 
        # how many moves were made on optimally completed solutiongs
        dvs['weighted_performance_score'] = {'value':  analysis_df.query('optimal_solution == True').num_moves_made.sum(), 'valence': 'Pos'} 
        description = 'many dependent variables related to tower of london performance'
        return dvs, description


def get_trsc_score(path):
    '''
    Measure: Trait Robustness of Self Confidence Survey
    Input: 
        Path of the response file
    Output: 
        Mean of the response list
    '''
    ## reverse_index = [0, 1, 6]
    response = get_response(path, 9)
    return sum(response)/len(response)