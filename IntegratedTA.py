#!/usr/bin/env python
# coding: utf-8
import time
from warnings import catch_warnings
import numpy as np
from numpy.core.defchararray import index
from numpy.core.numeric import True_
import pandas as pd
import seaborn as sns;
import matplotlib.pyplot as plt
from seaborn.external.docscrape import header
import textwrap
from configparser import ConfigParser, NoSectionError
import os
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
from sklearn.metrics.pairwise import cosine_similarity
def get_Threshold_for_Percentile(data, percentile):
        if type(data) is pd.DataFrame:
            return np.percentile(data.to_numpy().flatten(),percentile)
        else:
            return np.percentil(data,percentile)
def get_Histogram(data,bins=20,figsize=(100,40)):
    histogram_data=np.histogram(data,bins=20)
    histogram_df=pd.DataFrame([histogram_data[0],histogram_data[1]]).fillna(0)
    fig, ax = plt.subplots(figsize=figsize)    
    sns.set_theme(style="whitegrid")
    Hist_Chart = sns.histplot(data,stat='count',bins=bins)
    fig.tight_layout()
    plt.close()
    return histogram_df,Hist_Chart
def get_BarChart(data_x,data_y,figsize=(100,40),x_label="Topics", y_label="Document_Count",sort_data=False,wrap=False):
    fig, ax = plt.subplots(figsize=figsize) 
    sns.set_theme(style="whitegrid")
    if sort_data:
        enumerate_object = enumerate(data_y)
        sorted_indices = [i[0] for i in sorted(enumerate_object, key=lambda x:x[1], reverse=True)]
        dist_data_x=[data_x[sorted_indices[i]] for i in range(len(sorted_indices))]
        dist_data_y=[data_y[sorted_indices[i]] for i in range(len(sorted_indices))]
    else:
        dist_data_x=data_x
        dist_data_y=data_y
    sns_doc_dist_plot=sns.barplot(x=dist_data_x, y=dist_data_y)
    sns_doc_dist_plot.set_xlabel(x_label, fontsize = 100)
    sns_doc_dist_plot.set_ylabel(y_label, fontsize = 100)
    if wrap:
        ax.set_xticklabels([textwrap.fill(e, 30) for e in dist_data_x])
    fig.tight_layout()
    plt.close()
    return pd.DataFrame([data_x,data_y], index=[x_label,y_label]), sns_doc_dist_plot
def get_heatmap(data_df,figsize=(200,100), labels=np.empty(0),wrap=False,xticklabels=[], yticklabels=[],mask=False):
    '''
    Computes heat map for a DataFrame

    Parameters
    ----------
    data_df: DataFrame of numbers
    Returns
    -------
    heat map of data_df
    '''
    fig, ax = plt.subplots(figsize=(figsize[0],figsize[1]))    
    sns.set_theme()
    sns.set(font_scale=1.4)
    no_cols=len(data_df.columns)
    max_columns=[np.max(data_df[data_df.columns[i]][i+1:no_cols]) for i in range(no_cols-1)]
    max=np.max(max_columns)
    v_min=0
    v_max=max
    if len(xticklabels)==0:
        xticklabels=data_df.columns
    if len(yticklabels)==0:
        yticklabels=data_df.index.values
    if labels.shape[0]>0:
        if mask:
            mask = np.zeros_like(data_df, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True
            sns_plot = sns.heatmap(data_df.to_numpy(), vmin=v_min, vmax=v_max,annot=labels,mask=mask,fmt='',square=True, ax=ax,xticklabels=xticklabels, yticklabels=yticklabels)
        else:sns_plot = sns.heatmap(data_df.to_numpy(), vmin=v_min, vmax=v_max,annot=labels,fmt='',square=True, ax=ax,xticklabels=xticklabels, yticklabels=yticklabels)
    elif mask:
        sns_plot = sns.heatmap(data_df.to_numpy(), vmin=v_min, vmax=v_max,annot=True,mask=mask,square=True,ax=ax, fmt='.2f',xticklabels=xticklabels, yticklabels=yticklabels)
    else:
        sns_plot = sns.heatmap(data_df.to_numpy(), vmin=v_min, vmax=v_max,annot=True,square=True,ax=ax, fmt='.2f',xticklabels=xticklabels, yticklabels=yticklabels)
    sns.set(font_scale=2)
    fig.tight_layout()
    plt.close()
    return sns_plot
def get_Set_Jaccard_Similarity_Between_Vectors(V1,V2, Threshold=0.8):
    '''
    For a given pair of vectors V1 and V2, the Set Jacard Similarity is computed by the formula
    number of indices having both values greater than the Threshold divided by number of indices at least one has a value greater than the Threshold
    Parameters
    ----------
    V1 : Vector of numeric values
    V2 : Vector of numeric values
    Returns
    -------
    JS : Returns a real value
        DESCRIPTION: returns Set Jacard Similarity between pair of non negative vectors
    '''
    assert len(V1)==len(V2)
    numerator=len([i for i in range(len(V1)) if V1[i] >=Threshold and V2[i]>=Threshold ])
    denominator=len([i for i in range(len(V1)) if V1[i] >=Threshold  or V2[i] >=Threshold])
    if numerator==0:
        return 0
    return numerator/denominator
def get_Weighted_Jaccard_Similarity_Between_Vectors(V1,V2):
    '''
    For a given pair of vectors V1 and V2, the Jacard similarity is computed by the formula
    SUM(min(V1(i), V2(i)))/SUM(max(V1(i), V2(i))) for i is in range(len(V1))
    Parameters
    ----------
    V1 : Vector
    V2 : Vector
    Returns
    -------
    JS : Returns a real value
        DESCRIPTION: returns Jacard Similarity between pair of non negative vectors
    '''
    assert len(V1)==len(V2)
    numerator_list=[min(V1[j],V2[j]) for j in range(len(V1))]
    denominator_list=[max(V1[j],V2[j]) for j in range(len(V1))]
    numerator=np.sum(numerator_list)
    denominator=np.sum(denominator_list)
    if numerator==0:
        return 0
    return numerator/denominator
def get_Jaccard_Similarity_DataFrame(data_df, method='Weighted_Jaccard', Threshold=0.8):
    sim_size=len(data_df.columns)
    sim_array=np.empty((sim_size,sim_size), dtype=object)
    for i in range(len(data_df.columns)):
        sim_array[i,i]=1
        for j in range(i+1,len(data_df.columns)):
            if method is 'Weighted_Jaccard':
                sim_value=get_Weighted_Jaccard_Similarity_Between_Vectors(np.array(data_df[data_df.columns[i]].values),np.array(data_df[data_df.columns[j]].values))
            elif method is 'Set_Jaccard':
                sim_value=get_Set_Jaccard_Similarity_Between_Vectors(np.array(data_df[data_df.columns[i]].values),np.array(data_df[data_df.columns[j]].values), Threshold=Threshold)
            sim_array[i,j]=sim_value
            sim_array[j,i]=sim_value
    sim_df=pd.DataFrame(data = sim_array,index = data_df.columns, columns = data_df.columns,dtype=np.float)
    return sim_df

def get_NMF_UD_Jaccard_Similarity_DataFrame(data_df,method='Weighted_Jaccard',NMF_Topics=40, UD_Topics=31):
    sim_array=np.empty((NMF_Topics,UD_Topics), dtype=object)
    for i in range(NMF_Topics):
        for j in range(UD_Topics):
            if method is 'Weighted_Jaccard':
                sim_value=get_Weighted_Jaccard_Similarity_Between_Vectors(np.array(data_df[data_df.columns[i]].values),np.array(data_df[data_df.columns[j+NMF_Topics]].values))
            elif method is 'Set_Jaccard':
                sim_value=get_Set_Jaccard_Similarity_Between_Vectors(np.array(data_df[data_df.columns[i]].values),np.array(data_df[data_df.columns[j+NMF_Topics]].values))
            sim_array[i,j]=sim_value
    sim_df=pd.DataFrame(data = sim_array,index = data_df.columns[0:NMF_Topics], columns = data_df.columns[NMF_Topics:NMF_Topics+UD_Topics],dtype=np.float)
    return sim_df
class Company_Wise_Analysis():
    def __init__(self, Company_Wise_Data):
        self.Company_Wise_Sentence_to_NMF_Topic_Distribution=Company_Wise_Data[0][0]
        self.Company_Sentence_to_NMF_Topic_Distribution_Heatmap=Company_Wise_Data[0][1]
        self.Company_Wise_Sentence_to_UD_Topic_Distribution=Company_Wise_Data[1][0]
        self.Company_Sentence_to_UD_Topic_Distribution_Heatmap=Company_Wise_Data[1][1]
        self.Company_Wise_Paragraph_Max_to_NMF_Topic_Distribution=Company_Wise_Data[2][0]
        self.Company_Paragraph_Max_to_NMF_Topic_Distribution_Heatmap=Company_Wise_Data[2][1]
        self.Company_Wise_Paragraph_Max_to_UD_Topic_Distribution=Company_Wise_Data[3][0]
        self.Company_Paragraph_Max_to_UD_Topic_Distribution_Heatmap=Company_Wise_Data[3][1]
        self.Company_Wise_Paragraph_Mean_to_NMF_Topic_Distribution=Company_Wise_Data[4][0]
        self.Company_Paragraph_Mean_to_NMF_Topic_Distribution_Heatmap=Company_Wise_Data[4][1]
        self.Company_Wise_Paragraph_Mean_to_UD_Topic_Distribution=Company_Wise_Data[5][0]
        self.Company_Paragraph_Mean_to_UD_Topic_Distribution_Heatmap=Company_Wise_Data[5][1]
        self.Company_Topic_Score_By_Sentence_Mean_Topic_Score=Company_Wise_Data[6]
class Basic_Analysis():
    def __init__(self, Basic_Data):
        self.filtered_data=Basic_Data[0]
        self.NMF_Score_Histogram=Basic_Data[1]
        self.UD_Score_Histogram=Basic_Data[2]
        self.NMF_Highest_Score_For_Any_Topic_Histogram=Basic_Data[3]
        self.UD_Highest_Score_For_Any_Topic_Histogram=Basic_Data[4]
        self.NMF_Filterd_Score_Histogram=Basic_Data[5]
        self.UD_Filtered_Score_Histogram=Basic_Data[6]
        self.NMF_Topic_Distribution=Basic_Data[7]
        self.UD_Topic_Distribution=Basic_Data[8]
        self.NMF_Highest_Score_Topic_Distribution=Basic_Data[9]
        self.UD_Highest_Score_Topic_Distribution=Basic_Data[10]
class Similarity_Analysis_DataFrame():
    def __init__(self, Similarity_DataFrame):
        self.NMF_Similarity_DataFrame=Similarity_DataFrame[0]
        self.UD_Similarity_DataFrame=Similarity_DataFrame[1]
        self.NMF_UD_Similarity_DataFrame=Similarity_DataFrame[2]
class Similarity_Analysis_HeatMap():
    def __init__(self, Similarity_HeatMap):
        self.NMF_Similarity_HeatMap=Similarity_HeatMap[0]
        self.UD_Similarity_HeatMap=Similarity_HeatMap[1]
        self.NMF_UD_Similarity_HeatMap=Similarity_HeatMap[2]
        
class Embeddings_Similarity_Analysis_DataFrame():
    def __init__(self, Similarity_DataFrame):
        self.NMF_Embeddings_Similarity_DataFrame=Similarity_DataFrame[0]
        self.UD_Embeddings_Similarity_DataFrame=Similarity_DataFrame[1]
        self.NMF_UD_Embeddings_Similarity_DataFrame=Similarity_DataFrame[2]
class Embeddings_Similarity_Analysis_HeatMap():
    def __init__(self, Similarity_HeatMap):
        self.NMF_Embeddings_Similarity_HeatMap=Similarity_HeatMap[0]
        self.UD_Embeddings_Similarity_HeatMap=Similarity_HeatMap[1]
        self.NMF_UD_Embeddings_Similarity_HeatMap=Similarity_HeatMap[2]
'''
Topic Analysis is a class to hold the data and all required analysis methods
'''
class Topic_Analysis:
    def __init__(self,dataset, topic_top_words, NMF_Topics_Start=4,NMF_Topics=40,UD_Topics_Start=44, UD_Topics=31,NMF_Percentile_Threshold=75, UD_Threshold=.8):
        if type(dataset) is str:
            try:
                self.data=pd.read_csv(dataset, index_col='Index')
            except:
                print("The input is not a csv file and Topic Analysis cannot be done")
        elif type(dataset) is pd.DataFrame:
            self.data=dataset
        else:
            print("The input is not either csv file or a Pandas DataFrame for Data and Topic Analysis cannot be done")
        if type(topic_top_words) is str:
            try:
                self.topic_top_words_df=pd.read_csv(topic_top_words, index_col='Index')
            except:
                print("The input is not a csv file and Topic Analysis cannot be done")
        elif type(topic_top_words) is pd.DataFrame:
            self.topic_top_wordss_df=topic_top_words
        else:
            print("The input is not either csv file or a Pandas DataFrame for Topic Top Words and Topic Analysis cannot be done")
        self.Top_Words=self.get_Topic_Top_Words()
        self.Sentences=self.data.shape[0]
        self.Paragraphs=len(set(self.data['Para_id'].to_list()))
        self.Sentences_List=self.data['Sentences'].to_list()
        self.Paragraphs_List=self.data['Paragraps'].to_list()
        self.Company_Names=list(set(self.data['Company'].to_list()))
        self.Companies=len(set(self.data['Company'].to_list()))
        self.NMF_Topics_Start=NMF_Topics_Start
        self.NMF_Topics=NMF_Topics
        self.UD_Topics_Start=UD_Topics_Start
        self.UD_Topics=UD_Topics
        self.NMF_Topics_Names=list(self.data.columns[self.NMF_Topics_Start:self.NMF_Topics_Start+self.NMF_Topics])
        self.UD_Topics_Names=list(self.data.columns[self.UD_Topics_Start:self.UD_Topics_Start+self.UD_Topics])
        self.NMF_Percentile_Threshold=NMF_Percentile_Threshold
        self.UD_Threshold=UD_Threshold
        self.NMF_Threshold=self.get_NMF_threshold_for_percentile()
        self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score=self.data.copy(deep=True).groupby('Para_id').max()
        self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score=self.data.copy(deep=True).groupby('Para_id').mean()
        self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['Company']=self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['Company'].values
        self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score['Company']=self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['Company'].values
        self.Company_Sentence_Distribution=self.data.groupby(["Company"]).size()
        self.Company_Paragraph_Distribution=self.data[['Company','Para_id']].drop_duplicates().groupby(["Company"]).size()
        self.Company_Wise_No_Of_Sentence_Scoring_Above_Threshold_For_Each_Topic=self.get_Company_Wise_No_Of_Sentences_Scoring_Above_Threshold_For_Each_Topic()
        self.Company_Wise_No_Of_Paragraph_Max_Scoring_Above_Threshold_For_Each_Topic=self.get_Company_Wise_No_Of_Paragraphs_Max_Scoring_Above_Threshold_For_Each_Topic()
        self.Company_Wise_No_Of_Paragraph_Mean_Scoring_Above_Threshold_For_Each_Topic=self.get_Company_Wise_No_Of_Paragraphs_Mean_Scoring_Above_Threshold_For_Each_Topic()
        self.create_Sentence_NMF_Topic_By_Highest_Score()
        self.create_Sentence_UD_Topic_By_Highest_Score()
        self.create_Paragraph_Max_NMF_Topic_By_Highest_Score()
        self.create_Paragraph_Max_UD_Topic_By_Highest_Score()
        self.create_Paragraph_Mean_NMF_Topic_By_Highest_Score()
        self.create_Paragraph_Mean_UD_Topic_By_Highest_Score()
        self.NO_Sentences_Highest_NMF_Score_Above_Threshold=self.get_no_of_Sentences_with_highest_score_greater_than_NMF_Threshold()
        self.No_Sentences_Highest_UD_Score_Above_Threshold=self.get_no_of_Sentences_with_highest_score_greater_than_UD_Threshold()
        self.NO_Paragraphs_Max_Highest_NMF_Score_Above_Threshold=self.get_no_of_Paragraphs_Max_with_highest_score_greater_than_NMF_Threshold()
        self.No_Paragraphs_Max_Highest_UD_Score_Above_Threshold=self.get_no_of_Paragraphs_Max_with_highest_score_greater_than_UD_Threshold()
        self.NO_Paragraphs_Mean_Highest_NMF_Score_Above_Threshold=self.get_no_of_paragraphs_mean_with_highest_score_greater_than_NMF_Threshold()
        self.No_Paragraphs_Mean_Highest_UD_Score_Above_Threshold=self.get_no_of_paragraphs_mean_with_highest_score_greater_than_UD_Threshold()
        self.Sentence_filtered_data=self.get_Sentence_filtered_data()
        self.Paragraph_Max_filtered_data=self.get_Paragraph_Max_filtered_data()
        self.Paragraph_Mean_filtered_data=self.get_Paragraph_Mean_filtered_data()
        self.Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame=None
        self.Sentence_NMF_Set_Jaccard_Similarity_DataFrame=None
        self.Sentence_UD_Weighted_Jaccard_Similarity_DataFrame=None
        self.Sentence_UD_Set_Jaccard_Similarity_DataFrame=False
        self.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame=None
        self.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame=None
        self.Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame=None
        self.Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame=False
        self.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame=None
        self.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame=None
        self.Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame=None
        self.Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame=None
        self.Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame=None
        self.Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_DataFrame=None
        self.Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame=None
        self.Paragraphs_Sentences_Index=self.get_Paragraphs_Sentences_Index()
        self.Sentences_Vectors_Dict=self.get_Sentences_Vectors()
        self.Paragraphs_Vectors_Dict=self.get_Paragraphs_Vectors()
        self.Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame=None
        self.Sentence_NMF_Set_Embeddings_Similarity_DataFrame=None
        self.Sentence_UD_Weighted_Embeddings_Similarity_DataFrame=None
        self.Sentence_UD_Set_Embeddings_Similarity_DataFrame=False
        self.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame=None
        self.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame=None
        self.Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame=None
        self.Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame=False
        self.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame=None
        self.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame=None
        self.Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame=None
        self.Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame=None
        self.Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame=None
        self.Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_DataFrame=None
        self.Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame=None 
    def get_Paragraphs_Sentences_Index(self):
        fl=[]
        for i in range(0,len(self.Sentences_List)):
            ps=self.Paragraphs_List[i]
            st=self.Sentences_List[i]
            fl.append((ps,self.Sentences_List.index(st)))

        return fl
    def get_Sentences_Vectors(self):
        snt_vec_dict={}
        for j in self.Sentences_List:
                x=model.encode(j)
                snt_vec_dict[j]=x
        return snt_vec_dict
    def get_Paragraphs_Vectors(self):
        para_vec_dict={}
        for j in self.Paragraphs_List:
                x=model.encode(j)
                para_vec_dict[j]=x
        return para_vec_dict
    def similarity_analysis_NMF(self,em):
        a11=np.zeros(shape=(len(em),len(em)))
        for i in range(0,len(em)):
            for j in range(0,len(em)):
                if i!=j:
                    sr=cosine_similarity([em[i]],[em[j]])
                    a11[i][j]=sr[0][0]
                    if sr[0][0]==0:
                        a11[i][j]=0.0000001
                else:
                    a11[i][j]=0
        sim_df=pd.DataFrame(data = a11,index = self.NMF_Topics_Names, columns = self.NMF_Topics_Names)
        return sim_df
    def similarity_analysis_UD(self,em):
        a11=np.zeros(shape=(len(em),len(em)))
        for i in range(0,len(em)):
            for j in range(0,len(em)):
                if i!=j:
                    sr=cosine_similarity([em[i]],[em[j]])
                    a11[i][j]=sr[0][0]
                    if sr[0][0]==0:
                        a11[i][j]=0.0000001
                else:
                    a11[i][j]=0
        sim_df=pd.DataFrame(data = a11,index =self.UD_Topics_Names, columns = self.UD_Topics_Names)
        return sim_df
    def similarity_analysis_NMF_UD(self,em1,em):
        a11=np.zeros(shape=(len(em),len(em1)))
        for i in range(0,len(em)):
            for j in range(0,len(em1)):
                sr=cosine_similarity([em[i]],[em1[j]])
                a11[i][j]=sr[0][0]
                if sr[0][0]==0:
                    a11[i][j]=0.0000001
        sim_df=pd.DataFrame(data = a11,index = self.NMF_Topics_Names, columns =self.UD_Topics_Names)
        return sim_df
    def get_Sentence_NMF_Set_Embeddings_Similarity_DataFrame(self):
        """
        DESCRIPTION: finds the similarities between the two nmf topics at sentence level
        INPUT: nmf sentence csv file
        OUTPUT: sentences vectors of nmf topics, array of similarity scores
        """
        df_label=self.NMF_Topics_Names
        dim=len(self.NMF_Topics_Names)
        val2=[]
        for i in range(0,dim):
                tp1=df_label[i]
                val1 =list(self.Sentence_filtered_data[tp1].values[0:1657])
                for k in range(0,len(val1)):
                    if val1[k]>=self.NMF_Threshold:
                        val1[k]=self.Sentences_List[k]
                    else:
                        val1[k]="0"
                val2.append(val1)
        em=[]
        for i in range(0,dim): 
            tp=[]
            wts=[]
            for j in range(0,len(val2[i])):
                if val2[i][j] in self.Sentences_Vectors_Dict.keys():
                    x=self.Sentences_Vectors_Dict[val2[i][j]] # getting the vector of a sentence using the sentence vector dictionary
                    tp.append(x)
            if len(tp)!=0:
                em.append(sum(tp)/len(tp))
            else:
                em.append(np.zeros((384,), dtype=int))
        return em,self.similarity_analysis_NMF(em)
    def get_Sentence_UD_Set_Embeddings_Similarity_DataFrame(self):
            """
            DESCRIPTION: finds the similarities between the two nmf topics at sentence level
            INPUT: nmf sentence csv file
            OUTPUT: sentences vectors of nmf topics, array of similarity scores
            """
            df_label=self.UD_Topics_Names
            dim=len(self.UD_Topics_Names)
            val2=[]
            for i in range(0,dim):
                    tp1=df_label[i]
                    val1 =list(self.Sentence_filtered_data[tp1].values[0:1657])
                    for k in range(0,len(val1)):
                        if val1[k]>=self.NMF_Threshold:
                            val1[k]=self.Sentences_List[k]
                        else:
                            val1[k]="0"
                    val2.append(val1)
            em=[]
            for i in range(0,dim): 
                tp=[]
                wts=[]
                for j in range(0,len(val2[i])):
                    if val2[i][j] in self.Sentences_Vectors_Dict.keys():
                        x=self.Sentences_Vectors_Dict[val2[i][j]] # getting the vector of a sentence using the sentence vector dictionary
                        tp.append(x)
                if len(tp)!=0:
                    em.append(sum(tp)/len(tp))
                else:
                    em.append(np.zeros((384,), dtype=int))
            return em,self.similarity_analysis_UD(em)
    def get_Sentence_NMF_UD_Set_Embeddings_Similarity_DataFrame(self):
        NMF_Vectors=self.get_Sentence_NMF_Set_Embeddings_Similarity_DataFrame()[0]
        UD_Vectors=self.get_Sentence_UD_Set_Embeddings_Similarity_DataFrame()[0]
        NMF_UD_Vectors=NMF_Vectors+UD_Vectors
        return NMF_UD_Vectors,self.similarity_analysis_NMF_UD(UD_Vectors,NMF_Vectors)    
    def get_Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame(self):
    
        """
        DESCRIPTION: finds the similarities between the two nmf topics at sentence level
        INPUT: nmf sentence csv file
        OUTPUT: sentences vectors of nmf topics, array of similarity scores
        """
        df_label=self.NMF_Topics_Names
        dim=len(self.NMF_Topics_Names)
        val2=[]
        val21=[]
        for i in range(0,dim):
                tp1=df_label[i]
                val11 =list(self.Sentence_filtered_data[tp1].values[0:1657])
                val1 =list(self.Sentence_filtered_data[tp1].values[0:1657])
                for k in range(0,len(val1)):
                    if val1[k]>=self.NMF_Threshold:
                        val1[k]=self.Sentences_List[k]
                    else:
                        val1[k]="0"
                        val11[k]=0
                val2.append(val1)
                val21.append(val11)
        em=[]
        for i in range(0,dim): 
            tp=[]
            wts=[]
            for j in range(0,len(val2[i])):
                if val2[i][j] in  self.Sentences_Vectors_Dict.keys():
                    x= self.Sentences_Vectors_Dict[val2[i][j]]# getting the vector of a sentence using the sentence vector dictionary
                    ind=self.Sentences_List.index(val2[i][j])
                    if val21[i][ind]>0:
                        wt=val21[i][ind]
                    else:
                        wt=0.0
                    x=x*wt
                    tp.append(x)
                    wts.append(wt)
            if sum(wts)!=0:
                em.append(sum(tp)/sum(wts))
            else:
                em.append(np.zeros((384,), dtype=int))
        return em,self.similarity_analysis_NMF(em)
    def get_Sentence_UD_Weighted_Embeddings_Similarity_DataFrame(self):
            """
            DESCRIPTION: finds the similarities between the two nmf topics at sentence level
            INPUT: nmf sentence csv file
            OUTPUT: sentences vectors of nmf topics, array of similarity scores
            """
            df_label=self.UD_Topics_Names
            dim=len(self.UD_Topics_Names)
            val2=[]
            val21=[]
            for i in range(0,dim):
                    tp1=df_label[i]
                    val11 =list(self.Sentence_filtered_data[tp1].values[0:1657])
                    val1 =list(self.Sentence_filtered_data[tp1].values[0:1657])
                    for k in range(0,len(val1)):
                        if val1[k]>=self.UD_Threshold:
                            val1[k]=self.Sentences_List[k]
                        else:
                            val1[k]="0"
                            val11[k]=0
                    val2.append(val1)
                    val21.append(val11)
            em=[]
            for i in range(0,dim): 
                tp=[]
                wts=[]
                for j in range(0,len(val2[i])):
                    if val2[i][j] in  self.Sentences_Vectors_Dict.keys():
                        x=self.Sentences_Vectors_Dict[val2[i][j]]# getting the vector of a sentence using the sentence vector dictionary
                        ind=self.Sentences_List.index(val2[i][j])
                        if val21[i][ind]>0:
                            wt=val21[i][ind]
                        else:
                            wt=0.0
                        x=x*wt
                        tp.append(x)
                        wts.append(wt)
                if sum(wts)!=0:
                    em.append(sum(tp)/sum(wts))
                else:
                    em.append(np.zeros((384,), dtype=int))
            return em,self.similarity_analysis_UD(em)    
    def get_Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame(self):
        NMF_Vectors=self.get_Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame()[0]
        UD_Vectors=self.get_Sentence_UD_Weighted_Embeddings_Similarity_DataFrame()[0]
        NMF_UD_Vectors=NMF_Vectors+UD_Vectors
        return NMF_UD_Vectors,self.similarity_analysis_NMF_UD(UD_Vectors,NMF_Vectors)    
    def get_Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame(self):
    
        """
        DESCRIPTION: finds the similarities between the two nmf topics at sentence level
        INPUT: nmf sentence csv file
        OUTPUT: sentences vectors of nmf topics, array of similarity scores
        """
        df_label=self.NMF_Topics_Names
        dim=len(self.NMF_Topics_Names)
        val2=[]
        val21=[]
        for i in range(0,dim):
                tp1=df_label[i]
                val1 =list(self.Paragraph_Max_filtered_data[tp1].values[0:1657])
                for k in range(0,len(val1)):
                    if val1[k]>=self.NMF_Threshold:
                        val1[k]=self.Paragraphs_List[k]
                    else:
                        val1[k]="0"
                val2.append(val1)
        em=[]
        for i in range(0,dim): 
            tp=[]
            wts=[]
            for j in range(0,len(val2[i])):

                if val2[i][j] in self.Paragraphs_Vectors_Dict.keys():
                    x=self.Paragraphs_Vectors_Dict[val2[i][j]] # getting the vector of a sentence using the sentence vector dictionary
                    tp.append(x)
            if len(tp)!=0:
                em.append(sum(tp)/len(tp))
            else:
                em.append(np.zeros((384,), dtype=int))
        return em,self.similarity_analysis_NMF(em)
    def get_Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame(self):
            """
            DESCRIPTION: finds the similarities between the two nmf topics at sentence level
            INPUT: nmf sentence csv file
            OUTPUT: sentences vectors of nmf topics, array of similarity scores
            """            
            df_label=self.UD_Topics_Names
            dim=len(self.UD_Topics_Names)
            val2=[]
            val21=[]
            for i in range(0,dim):
                    tp1=df_label[i]
                    val1 =list(self.Paragraph_Max_filtered_data[tp1].values[0:1657])
                    for k in range(0,len(val1)):
                        if val1[k]>=self.UD_Threshold:
                            val1[k]=self.Paragraphs_List[k]
                        else:
                            val1[k]="0"
                    val2.append(val1)
            em=[]
            for i in range(0,dim): 
                tp=[]
                wts=[]
                for j in range(0,len(val2[i])):
                    if val2[i][j] in self.Paragraphs_Vectors_Dict.keys():
                        x=self.Paragraphs_Vectors_Dict[val2[i][j]] # getting the vector of a sentence using the sentence vector dictionary
                        tp.append(x)
                if len(tp)!=0:
                    em.append(sum(tp)/len(tp))
                else:
                    em.append(np.zeros((384,), dtype=int))
            return em,self.similarity_analysis_UD(em)
    def get_Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_DataFrame(self):
        NMF_Vectors=self.get_Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame()[0]
        UD_Vectors=self.get_Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame()[0]
        NMF_UD_Vectors=NMF_Vectors+UD_Vectors
        return NMF_UD_Vectors,self.similarity_analysis_NMF_UD(UD_Vectors,NMF_Vectors)
    def get_Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame(self):

        """
        DESCRIPTION: finds the similarities between the two nmf topics at sentence level
        INPUT: nmf sentence csv file
        OUTPUT: sentences vectors of nmf topics, array of similarity scores
        """
        df_label=self.NMF_Topics_Names
        dim=len(self.NMF_Topics_Names)
        val2=[]
        val21=[]
        for i in range(0,dim):
                tp1=df_label[i]
                val11 =list(self.Paragraph_Max_filtered_data[tp1].values[0:1657])
                val1 =list(self.Paragraph_Max_filtered_data[tp1].values[0:1657])
                for k in range(0,len(val1)):
                    if val1[k]>=self.NMF_Threshold:
                        val1[k]=self.Paragraphs_List[k]
                    else:
                        val1[k]="0"
                        val11[k]=0
                val2.append(val1)
                val21.append(val11)
        em=[]
        for i in range(0,dim): 
            tp=[]
            wts=[]
            for j in range(0,len(val2[i])):
                """
                self.Paragraphs_Sentences_Index
            self.Paragraphs_Vectors_Dict
                """
                if val2[i][j] in self.Paragraphs_Vectors_Dict.keys():
                    x=self.Paragraphs_Vectors_Dict[val2[i][j]]# getting the vector of a sentence using the sentence vector dictionary
                    ind=self.Paragraphs_Sentences_Index[j][1]
                    if val21[i][ind]>0:
                        wt=val21[i][ind]
                    else:
                        wt=0.0
                    x=x*wt
                    tp.append(x)
                    wts.append(wt)
            if sum(wts)!=0:
                em.append(sum(tp)/sum(wts))
            else:
                em.append(np.zeros((384,), dtype=int))
        return em,self.similarity_analysis_NMF(em)
    def get_Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame(self):
            """
            DESCRIPTION: finds the similarities between the two nmf topics at sentence level
            INPUT: nmf sentence csv file
            OUTPUT: sentences vectors of nmf topics, array of similarity scores
            """
            df_label=self.UD_Topics_Names
            dim=len(self.UD_Topics_Names)
            val2=[]
            val21=[]
            for i in range(0,dim):
                    tp1=df_label[i]
                    val11 =list(self.Paragraph_Max_filtered_data[tp1].values[0:1657])
                    val1 =list(self.Paragraph_Max_filtered_data[tp1].values[0:1657])
                    for k in range(0,len(val1)):
                        if val1[k]>=self.UD_Threshold:
                            val1[k]=self.Paragraphs_List[k]
                        else:
                            val1[k]="0"
                            val11[k]=0
                    val2.append(val1)
                    val21.append(val11)
            em=[]
            for i in range(0,dim): 
                tp=[]
                wts=[]
                for j in range(0,len(val2[i])):
                    if val2[i][j] in self.Paragraphs_Vectors_Dict.keys():
                        x=self.Paragraphs_Vectors_Dict[val2[i][j]]# getting the vector of a sentence using the sentence vector dictionary
                        ind=self.Paragraphs_Sentences_Index[j][1]
                        if val21[i][ind]>0:
                            wt=val21[i][ind]
                        else:
                            wt=0.0
                        x=x*wt
                        tp.append(x)
                        wts.append(wt)
                if sum(wts)!=0:
                    em.append(sum(tp)/sum(wts))
                else:
                    em.append(np.zeros((384,), dtype=int))
            return em,self.similarity_analysis_UD(em)
    def get_Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_DataFrame(self):
        NMF_Vectors=self.get_Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame()[0]
        UD_Vectors=self.get_Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame()[0]
        NMF_UD_Vectors=NMF_Vectors+UD_Vectors
        return NMF_UD_Vectors,self.similarity_analysis_NMF_UD(UD_Vectors,NMF_Vectors)    
    def get_Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame(self):
    
        """
        DESCRIPTION: finds the similarities between the two nmf topics at sentence level
        INPUT: nmf sentence csv file
        OUTPUT: sentences vectors of nmf topics, array of similarity scores
        """
        df_label=self.NMF_Topics_Names
        dim=len(self.NMF_Topics_Names)
        val2=[]
        for i in range(0,dim):
                tp1=df_label[i]
                val1 =list(self.Paragraph_Mean_filtered_data[tp1].values[0:1657])
                for k in range(0,len(val1)):
                    if val1[k]>=self.NMF_Threshold:
                        val1[k]=self.Paragraphs_List[k]
                    else:
                        val1[k]="0"
                val2.append(val1)
        em=[]
        for i in range(0,dim): 
            tp=[]
            wts=[]
            for j in range(0,len(val2[i])):
                if val2[i][j] in self.Paragraphs_Vectors_Dict.keys():
                    x=self.Paragraphs_Vectors_Dict[val2[i][j]] # getting the vector of a sentence using the sentence vector dictionary
                    tp.append(x)
            if len(tp)!=0:
                em.append(sum(tp)/len(tp))
            else:
                em.append(np.zeros((384,), dtype=int))
        return em,self.similarity_analysis_NMF(em)

    def get_Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame(self):
            """
            DESCRIPTION: finds the similarities between the two nmf topics at sentence level
            INPUT: nmf sentence csv file
            OUTPUT: sentences vectors of nmf topics, array of similarity scores
            """
            df_label=self.UD_Topics_Names
            dim=len(self.UD_Topics_Names)
            val2=[]
            for i in range(0,dim):
                    tp1=df_label[i]
                    val1 =list(self.Paragraph_Mean_filtered_data[tp1].values[0:1657])
                    for k in range(0,len(val1)):
                        if val1[k]>=self.UD_Threshold:
                            val1[k]=self.Paragraphs_List[k]
                        else:
                            val1[k]="0"
                    val2.append(val1)
            em=[]
            for i in range(0,dim): 
                tp=[]
                wts=[]
                for j in range(0,len(val2[i])):
                    if val2[i][j] in self.Paragraphs_Vectors_Dict.keys():
                        x=self.Paragraphs_Vectors_Dict[val2[i][j]] # getting the vector of a sentence using the sentence vector dictionary
                        tp.append(x)
                if len(tp)!=0:
                    em.append(sum(tp)/len(tp))
                else:
                    em.append(np.zeros((384,), dtype=int))
            return em,self.similarity_analysis_UD(em)

    def get_Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_DataFrame(self):
        NMF_Vectors=self.get_Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame()[0]
        UD_Vectors=self.get_Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame()[0]
        NMF_UD_Vectors=NMF_Vectors+UD_Vectors
        return NMF_UD_Vectors,self.similarity_analysis_NMF_UD(UD_Vectors,NMF_Vectors)

    def get_Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame(self):

        """
        DESCRIPTION: finds the similarities between the two nmf topics at sentence level
        INPUT: nmf sentence csv file
        OUTPUT: sentences vectors of nmf topics, array of similarity scores
        """
        df_label=self.NMF_Topics_Names
        dim=len(self.NMF_Topics_Names)
        val2=[]
        val21=[]
        for i in range(0,dim):
                tp1=df_label[i]
                val11 =list(self.Paragraph_Mean_filtered_data[tp1].values[0:1657])
                val1 =list(self.Paragraph_Mean_filtered_data[tp1].values[0:1657])
                for k in range(0,len(val1)):
                    if val1[k]>=self.NMF_Threshold:
                        val1[k]=self.Paragraphs_List[k]
                    else:
                        val1[k]="0"
                        val11[k]=0
                val2.append(val1)
                val21.append(val11)
        em=[]
        for i in range(0,dim): 
            tp=[]
            wts=[]
            for j in range(0,len(val2[i])):
                if val2[i][j] in self.Paragraphs_Vectors_Dict.keys():
                    x=self.Paragraphs_Vectors_Dict[val2[i][j]]# getting the vector of a sentence using the sentence vector dictionary
                    ind=self.Paragraphs_Sentences_Index[j][1]
                    if val21[i][ind]>0:
                        wt=val21[i][ind]
                    else:
                        wt=0.0
                    x=x*wt
                    tp.append(x)
                    wts.append(wt)
            if sum(wts)!=0:
                em.append(sum(tp)/sum(wts))
            else:
                em.append(np.zeros((384,), dtype=int))
        return em,self.similarity_analysis_NMF(em)
    def get_Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame(self):
            """
            DESCRIPTION: finds the similarities between the two nmf topics at sentence level
            INPUT: nmf sentence csv file
            OUTPUT: sentences vectors of nmf topics, array of similarity scores
            """
            df_label=self.UD_Topics_Names
            dim=len(self.UD_Topics_Names)
            val2=[]
            val21=[]
            for i in range(0,dim):
                    tp1=df_label[i]
                    val11 =list(self.Paragraph_Mean_filtered_data[tp1].values[0:1657])
                    val1 =list(self.Paragraph_Mean_filtered_data[tp1].values[0:1657])
                    for k in range(0,len(val1)):
                        if val1[k]>=self.UD_Threshold:
                            val1[k]=self.Paragraphs_List[k]
                        else:
                            val1[k]="0"
                            val11[k]=0
                    val2.append(val1)
                    val21.append(val11)
            em=[]
            for i in range(0,dim): 
                tp=[]
                wts=[]
                for j in range(0,len(val2[i])):
                    if val2[i][j] in self.Paragraphs_Vectors_Dict.keys():
                        x=self.Paragraphs_Vectors_Dict[val2[i][j]]# getting the vector of a sentence using the sentence vector dictionary
                        ind=self.Paragraphs_Sentences_Index[j][1]
                        if val21[i][ind]>0:
                            wt=val21[i][ind]
                        else:
                            wt=0.0
                        x=x*wt
                        tp.append(x)
                        wts.append(wt)
                if sum(wts)!=0:
                    em.append(sum(tp)/sum(wts))
                else:
                    em.append(np.zeros((384,), dtype=int))

            return em,self.similarity_analysis_UD(em)

    def get_Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame(self):

        NMF_Vectors=self.get_Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame()[0]
        UD_Vectors=self.get_Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame()[0]
        NMF_UD_Vectors=NMF_Vectors+UD_Vectors

        return NMF_UD_Vectors,self.similarity_analysis_NMF_UD(UD_Vectors,NMF_Vectors)
    def get_Topic_Top_Words(self):
        cols=self.topic_top_words_df.columns
        Top_Words=[]
        for i in range(len(cols)):
            words=self.topic_top_words_df[cols[i]].values
            Top_Words.append(words)
        return Top_Words
    def get_NMF_threshold_for_percentile(self,percentile=None,NMF_Topic_Columns=None):
        if not percentile:
            percentile=self.NMF_Percentile_Threshold
        if not NMF_Topic_Columns:
            NMF_Topic_Columns=self.NMF_Topics_Names
        return get_Threshold_for_Percentile(self.data[NMF_Topic_Columns],percentile)
    
    def get_Company_Wise_No_Of_Sentences_Scoring_Above_Threshold_For_Each_Topic(self,NMF_percentile=75,UD_Threshold=0.8):
        NMF_Threshold=self.get_NMF_threshold_for_percentile(percentile=NMF_percentile)
        data=self.data[["Company"]+self.NMF_Topics_Names+self.UD_Topics_Names].copy(deep=True)
        for col in self.NMF_Topics_Names:
            data[col]=np.where(data[col]>=NMF_Threshold,1,data[col])
            data[col]=np.where(data[col]<NMF_Threshold,0,data[col])
        for col in self.UD_Topics_Names:
            data[col]=np.where(data[col]>=UD_Threshold,1,data[col])
            data[col]=np.where(data[col]<UD_Threshold,0,data[col])
        return data.groupby(['Company']).sum()
    def get_Company_Wise_No_Of_Paragraphs_Max_Scoring_Above_Threshold_For_Each_Topic(self,NMF_percentile=75,UD_Threshold=0.8):
        NMF_Threshold=self.get_NMF_threshold_for_percentile(percentile=NMF_percentile)
        data=self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score[["Company"]+self.NMF_Topics_Names+self.UD_Topics_Names].copy(deep=True)
        for col in self.NMF_Topics_Names:
            data[col]=np.where(data[col]>=NMF_Threshold,1,data[col])
            data[col]=np.where(data[col]<NMF_Threshold,0,data[col])
        for col in self.UD_Topics_Names:
            data[col]=np.where(data[col]>=UD_Threshold,1,data[col])
            data[col]=np.where(data[col]<UD_Threshold,0,data[col])
        return data.groupby(['Company']).sum()
    def get_Company_Wise_No_Of_Paragraphs_Mean_Scoring_Above_Threshold_For_Each_Topic(self,NMF_percentile=75,UD_Threshold=0.8):
        NMF_Threshold=self.get_NMF_threshold_for_percentile(percentile=NMF_percentile)
        data=self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score[["Company"]+self.NMF_Topics_Names+self.UD_Topics_Names].copy(deep=True)
        for col in self.NMF_Topics_Names:
            data[col]=np.where(data[col]>=NMF_Threshold,1,data[col])
            data[col]=np.where(data[col]<NMF_Threshold,0,data[col])
        for col in self.UD_Topics_Names:
            data[col]=np.where(data[col]>=UD_Threshold,1,data[col])
            data[col]=np.where(data[col]<UD_Threshold,0,data[col])
        return data.groupby(['Company']).sum()
    def get_filtered_data(self,data,NMF=True,NMF_Threshold_Percentile=None,NMF_Topic_Columns=None,UD=True,UD_Threshold=None,UD_Topic_Columns=None):
        if NMF:
            if not NMF_Threshold_Percentile:
                NMF_Threshold_Percentile=self.NMF_Percentile_Threshold
                NMF_Threshold=self.NMF_Threshold
            else:
                NMF_Threshold=self.get_NMF_threshold_score_for_percentile()
            if not NMF_Topic_Columns:
                NMF_Topic_Columns=self.NMF_Topics_Names
            else:
                NMF_Topic_Columns=[col for col in NMF_Topic_Columns if col in self.NMF_Topics_Names]
            for col in NMF_Topic_Columns:
                data[col]=np.where(data[col]<NMF_Threshold,0,data[col])
        if UD:
            if not UD_Threshold:
                UD_Threshold=self.UD_Threshold
            if not UD_Topic_Columns:
                UD_Topic_Columns=self.UD_Topics_Names
            else:
                for col in NMF_Topic_Columns:
                    data[col]=np.where(data[col]<UD_Threshold,0,data[col])
            return data
    def get_Sentence_filtered_data(self,NMF=True,NMF_Threshold_Percentile=None,NMF_Topic_Columns=None,UD=True,UD_Threshold=None,UD_Topic_Columns=None):
        data=self.data.copy(deep=True)
        return self.get_filtered_data(data,NMF=True,NMF_Threshold_Percentile=None,NMF_Topic_Columns=None,UD=True,UD_Threshold=None,UD_Topic_Columns=None)
    def get_Paragraph_Mean_filtered_data(self,NMF=True,NMF_Threshold_Percentile=None,NMF_Topic_Columns=None,UD=True,UD_Threshold=None,UD_Topic_Columns=None):
        data=self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score.copy(deep=True)
        return self.get_filtered_data(data,NMF=True,NMF_Threshold_Percentile=None,NMF_Topic_Columns=None,UD=True,UD_Threshold=None,UD_Topic_Columns=None)
    def get_Paragraph_Max_filtered_data(self,NMF=True,NMF_Threshold_Percentile=None,NMF_Topic_Columns=None,UD=True,UD_Threshold=None,UD_Topic_Columns=None):
        data=self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score.copy(deep=True)
        return self.get_filtered_data(data,NMF=True,NMF_Threshold_Percentile=None,NMF_Topic_Columns=None,UD=True,UD_Threshold=None,UD_Topic_Columns=None)
    def update_Thresholds_Filtered_Data(self,NMF_Percentile_Threshold=None, UD_Threshold=None,update_filtered_data=True):
        if NMF_Percentile_Threshold:
            self.NMF_Percentile_Threshold=NMF_Percentile_Threshold
            self.NMF_Threshold=self.get_NMF_threshold_for_percentile()
        if UD_Threshold:
            self.UD_Threshold=UD_Threshold
        self.filtered_data=self.get_filtered_data()
    def create_Sentence_NMF_Topic_By_Highest_Score(self):
        self.data['NMF_Topic']=[np.argmax(self.data[self.NMF_Topics_Names].iloc[i].to_numpy()) for i in self.data.index]
        self.data['NMF_Topic_Score']=[np.max(self.data[self.NMF_Topics_Names].iloc[i].to_numpy()) for i in self.data.index]
    def get_no_of_Sentences_with_highest_score_greater_than_NMF_Threshold(self):
        return self.data['NMF_Topic_Score'][self.data['NMF_Topic_Score']>self.NMF_Threshold].count()
    def create_Paragraph_Max_NMF_Topic_By_Highest_Score(self):
        self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['NMF_Topic']=[np.argmax(self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score[self.NMF_Topics_Names].iloc[i].to_numpy()) for i in self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score.index]
        self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['NMF_Topic_Score']=[np.max(self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score[self.NMF_Topics_Names].iloc[i].to_numpy()) for i in self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score.index]
        return self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score
    def get_no_of_Paragraphs_Max_with_highest_score_greater_than_NMF_Threshold(self):
        return self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['NMF_Topic_Score'][self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['NMF_Topic_Score']>self.NMF_Threshold].count()
    def create_Paragraph_Mean_NMF_Topic_By_Highest_Score(self):
        self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score['NMF_Topic']=[np.argmax(self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score[self.NMF_Topics_Names].iloc[i].to_numpy()) for i in self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score.index]
        self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score['NMF_Topic_Score']=[np.max(self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score[self.NMF_Topics_Names].iloc[i].to_numpy()) for i in self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score.index]
        return self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score
    def get_no_of_paragraphs_mean_with_highest_score_greater_than_NMF_Threshold(self):
        return self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score['NMF_Topic_Score'][self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score['NMF_Topic_Score']>self.NMF_Threshold].count()
    def create_Sentence_UD_Topic_By_Highest_Score(self):
        self.data['UD_Topic']=[np.argmax(self.data[self.UD_Topics_Names].iloc[i].to_numpy()) for i in self.data.index]
        self.data['UD_Topic_Score']=[np.max(self.data[self.UD_Topics_Names].iloc[i].to_numpy()) for i in self.data.index]
    def get_no_of_Sentences_with_highest_score_greater_than_UD_Threshold(self):
        return self.data['UD_Topic_Score'][self.data['UD_Topic_Score']>self.UD_Threshold].count()
    def create_Paragraph_Max_UD_Topic_By_Highest_Score(self):
        self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['UD_Topic']=[np.argmax(self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score[self.UD_Topics_Names].iloc[i].to_numpy()) for i in self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score.index]
        self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['UD_Topic_Score']=[np.max(self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score[self.UD_Topics_Names].iloc[i].to_numpy()) for i in self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score.index]
        return self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score
    def get_no_of_Paragraphs_Max_with_highest_score_greater_than_UD_Threshold(self):
        return self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['UD_Topic_Score'][self.data['UD_Topic_Score']>self.UD_Threshold].count()
    def create_Paragraph_Mean_UD_Topic_By_Highest_Score(self):
        self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score['UD_Topic']=[np.argmax(self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score[self.UD_Topics_Names].iloc[i].to_numpy()) for i in self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score.index]
        self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score['UD_Topic_Score']=[np.max(self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score[self.UD_Topics_Names].iloc[i].to_numpy()) for i in self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score.index]
    def get_no_of_paragraphs_mean_with_highest_score_greater_than_UD_Threshold(self):
        return self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score['UD_Topic_Score'][self.data['UD_Topic_Score']>self.UD_Threshold].count()
    def get_Sentence_NMF_Score_Histogram(self):
        return  get_Histogram(self.data[self.NMF_Topics_Names].to_numpy().flatten())
    def get_Sentence_UD_Score_Histogram(self):
        return  get_Histogram(self.data[self.UD_Topics_Names].to_numpy().flatten())
    def get_Paragraph_Max_NMF_Score_Histogram(self):
        return  get_Histogram(self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score[self.NMF_Topics_Names].to_numpy().flatten())
    def get_Paragraph_Max_UD_Score_Histogram(self):
        return  get_Histogram(self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score[self.UD_Topics_Names].to_numpy().flatten())
    def get_Paragraph_Mean_NMF_Score_Histogram(self):
        return  get_Histogram(self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score[self.NMF_Topics_Names].to_numpy().flatten())
    def get_Paragraph_Mean_UD_Score_Histogram(self):
        return  get_Histogram(self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score[self.UD_Topics_Names].to_numpy().flatten())
    def get_Sentence_NMF_Highest_Score_For_Any_Topic_Histogram(self):
        return  get_Histogram(self.data['NMF_Topic_Score'].to_numpy().flatten())
    def get_Sentence_UD_Highest_Score_For_Any_Topic_Histogram(self):
        return  get_Histogram(self.data['UD_Topic_Score'].to_numpy().flatten())
    def get_Paragraph_Max_NMF_Highest_Score_For_Any_Topic_Histogram(self):
        return  get_Histogram(self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['NMF_Topic_Score'].to_numpy().flatten())
    def get_Paragraph_Max_UD_Highest_Score_For_Any_Topic_Histogram(self):
        return  get_Histogram(self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score['UD_Topic_Score'].to_numpy().flatten())
    def get_Paragraph_Mean_NMF_Highest_Score_For_Any_Topic_Histogram(self):
        return  get_Histogram(self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score['NMF_Topic_Score'].to_numpy().flatten())
    def get_Paragraph_Mean_UD_Highest_Score_For_Any_Topic_Histogram(self):
        return  get_Histogram(self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score['UD_Topic_Score'].to_numpy().flatten())
    def get_Sentence_NMF_Filterd_Score_Histogram(self):
        return get_Histogram(self.Sentence_filtered_data[self.NMF_Topics_Names].to_numpy().flatten())
    def get_Sentence_UD_Filtered_Score_Histogram(self):
        return get_Histogram(self.Sentence_filtered_data[self.UD_Topics_Names].to_numpy().flatten())
    def get_Paragraph_Max_NMF_Filterd_Score_Histogram(self):
        return get_Histogram(self.Paragraph_Max_filtered_data[self.NMF_Topics_Names].to_numpy().flatten())
    def get_Paragraph_Max_UD_Filterd_Score_Histogram(self):
        return get_Histogram(self.Paragraph_Max_filtered_data[self.UD_Topics_Names].to_numpy().flatten())
    def get_Paragraph_Mean_NMF_Filterd_Score_Histogram(self):
        return get_Histogram(self.Paragraph_Mean_filtered_data[self.NMF_Topics_Names].to_numpy().flatten())
    def get_Paragraph_Mean_UD_Filterd_Score_Histogram(self):
        return get_Histogram(self.Paragraph_Mean_filtered_data[self.UD_Topics_Names].to_numpy().flatten())
    def get_Company_Sentence_Distribution(self, x_label="Company", y_label="Sentence_Count",sort_data=True):
        Topics=self.Company_Sentence_Distribution.index.values
        Sentence_Count=self.Company_Sentence_Distribution.values
        return get_BarChart(Topics,Sentence_Count,x_label=x_label, y_label=y_label,sort_data=sort_data,wrap=True)
    def get_Company_Paragraph_Distribution(self, x_label="Company", y_label="Sentence_Count",sort_data=True):
        Topics=self.Company_Paragraph_Distribution.index.values
        Paragraph_Count=self.Company_Paragraph_Distribution.values
        return get_BarChart(Topics,Paragraph_Count,x_label=x_label, y_label=y_label,sort_data=sort_data,wrap=True)
    def get_NMF_Topic_Sentence_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True,No_of_top_words=5):
        if No_of_top_words>20:
            No_of_top_words=20
        Top_Words=[', '.join(self.Top_Words[i][0:No_of_top_words]) for i in range(len(self.Top_Words))]
        Topics=[self.NMF_Topics_Names[i]+", "+Top_Words[i] for i in range(self.NMF_Topics)]
        Sentence_Count=[(self.Sentence_filtered_data[self.NMF_Topics_Names[i]] != 0).sum() for i in range(self.NMF_Topics)]
        return get_BarChart(Topics,Sentence_Count,x_label=x_label, y_label=y_label,sort_data=sort_data,wrap=True)
    def get_UD_Topic_Sentence_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True):
        Topics=[self.UD_Topics_Names[i] for i in range(self.UD_Topics)]
        Sentence_Count=[(self.Sentence_filtered_data[self.UD_Topics_Names[i]] != 0).sum() for i in range(self.UD_Topics)]
        return get_BarChart(Topics,Sentence_Count,x_label=x_label, y_label=y_label,sort_data=sort_data,wrap=True)
    def get_NMF_Topic_Paragraph_Max_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True,No_of_top_words=5):
        if No_of_top_words>20:
            No_of_top_words=20
        Top_Words=[', '.join(self.Top_Words[i][0:No_of_top_words]) for i in range(len(self.Top_Words))]
        Topics=[self.NMF_Topics_Names[i]+", "+Top_Words[i] for i in range(self.NMF_Topics)]
        Sentence_Count=[(self.Paragraph_Max_filtered_data[self.NMF_Topics_Names[i]] != 0).sum() for i in range(self.NMF_Topics)]
        return get_BarChart(Topics,Sentence_Count,x_label=x_label, y_label=y_label,sort_data=sort_data,wrap=True)
    def get_UD_Topic_Paragraph_Max_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True):
        Topics=[self.UD_Topics_Names[i] for i in range(self.UD_Topics)]
        Sentence_Count=[(self.Paragraph_Max_filtered_data[self.UD_Topics_Names[i]] != 0).sum() for i in range(self.UD_Topics)]
        return get_BarChart(Topics,Sentence_Count,x_label=x_label, y_label=y_label,sort_data=sort_data,wrap=True)
    def get_NMF_Topic_Paragraph_Mean_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True,No_of_top_words=5):
        if No_of_top_words>20:
            No_of_top_words=20
        Top_Words=[', '.join(self.Top_Words[i][0:No_of_top_words]) for i in range(len(self.Top_Words))]
        Topics=[self.NMF_Topics_Names[i]+", "+Top_Words[i] for i in range(self.NMF_Topics)]
        Sentence_Count=[(self.Paragraph_Mean_filtered_data[self.NMF_Topics_Names[i]] != 0).sum() for i in range(self.NMF_Topics)]
        return get_BarChart(Topics,Sentence_Count,x_label=x_label, y_label=y_label,sort_data=sort_data,wrap=True)
    def get_UD_Topic_Paragraph_Mean_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True):
        Topics=[self.UD_Topics_Names[i] for i in range(self.UD_Topics)]
        Sentence_Count=[(self.Paragraph_Mean_filtered_data[self.UD_Topics_Names[i]] != 0).sum() for i in range(self.UD_Topics)]
        return get_BarChart(Topics,Sentence_Count,x_label=x_label, y_label=y_label,sort_data=sort_data,wrap=True)

    def get_NMF_Highest_Score_Topic_Sentence_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True,No_of_top_words=5):
        counts=self.data.groupby(["NMF_Topic"]).size()
        topic_indices=counts.index.values
        if No_of_top_words>20:
            No_of_top_words=20
        Top_Words=[', '.join(self.Top_Words[i][0:No_of_top_words]) for i in range(len(self.Top_Words))]
        Topics=[self.NMF_Topics_Names[i]+", "+Top_Words[i] for i in range(self.NMF_Topics)]
        Sentence_Count=counts.to_numpy()
        return get_BarChart(Topics,Sentence_Count,x_label=x_label,y_label=y_label,sort_data=sort_data,wrap=True)
    def get_UD_Highest_Score_Topic_Sentence_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True):
        counts=self.data.groupby(["UD_Topic"]).size()
        topic_indices=counts.index.values
        Topics=[self.UD_Topics_Names[i] for i in range(len(topic_indices))]
        Sentence_Count=counts.to_numpy()
        return get_BarChart(Topics,Sentence_Count,x_label=x_label,y_label=y_label,sort_data=sort_data)

    def get_NMF_Highest_Score_Topic_Paragraph_Max_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True,No_of_top_words=5):
        counts=self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score.groupby(["NMF_Topic"]).size()
        topic_indices=counts.index.values
        if No_of_top_words>20:
            No_of_top_words=20
        Top_Words=[', '.join(self.Top_Words[i][0:No_of_top_words]) for i in range(len(self.Top_Words))]
        Topics=[self.NMF_Topics_Names[i]+", "+Top_Words[i] for i in range(self.NMF_Topics)]
        Sentence_Count=counts.to_numpy()
        return get_BarChart(Topics,Sentence_Count,x_label=x_label,y_label=y_label,sort_data=sort_data,wrap=True)
    def get_UD_Highest_Score_Topic_Paragraph_Max_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True):
        counts=self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score.groupby(["UD_Topic"]).size()
        topic_indices=counts.index.values
        Topics=[self.UD_Topics_Names[i] for i in range(len(topic_indices))]
        Sentence_Count=counts.to_numpy()
        return get_BarChart(Topics,Sentence_Count,x_label=x_label,y_label=y_label,sort_data=sort_data)

    def get_NMF_Highest_Score_Topic_Paragraph_Mean_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True,No_of_top_words=5):
        counts=self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score.groupby(["NMF_Topic"]).size()
        topic_indices=counts.index.values
        if No_of_top_words>20:
            No_of_top_words=20
        Top_Words=[', '.join(self.Top_Words[i][0:No_of_top_words]) for i in range(len(self.Top_Words))]
        Topics=[self.NMF_Topics_Names[i]+", "+Top_Words[i] for i in range(self.NMF_Topics)]
        Sentence_Count=counts.to_numpy()
        return get_BarChart(Topics,Sentence_Count,x_label=x_label,y_label=y_label,sort_data=sort_data,wrap=True)
    def get_UD_Highest_Score_Topic_Paragraph_Mean_Distribution(self,x_label="Topics",y_label="Sentence_Count",sort_data=True):
        counts=self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score.groupby(["UD_Topic"]).size()
        topic_indices=counts.index.values
        Topics=[self.UD_Topics_Names[i] for i in range(len(topic_indices))]
        Sentence_Count=counts.to_numpy()
        return get_BarChart(Topics,Sentence_Count,x_label=x_label,y_label=y_label,sort_data=sort_data)

    def get_Similarity_DataFrame(self, level='Sentence', topics='NMF', method='Weighted_Jaccard', Threshold=None):
        if topics is 'NMF':
            columns=self.NMF_Topics_Names
            if not Threshold:
                Threshold=self.NMF_Threshold
        elif topics is 'UD':
            columns=self.UD_Topics_Names
            if not Threshold:
                Threshold=self.UD_Threshold
        if level is 'Sentence':
            data=self.data[columns]
        elif level is 'Paragraph_Max':
            data=self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score[columns]
        elif level is 'Paragraph_Mean':
            data=self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score[columns]
        return get_Jaccard_Similarity_DataFrame(data, method=method, Threshold=Threshold)
    def get_Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Sentence', topics='NMF', method='Weighted_Jaccard')
    def get_Sentence_NMF_Set_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Sentence', topics='NMF', method='Set_Jaccard',Threshold=self.NMF_Threshold)
    def get_Sentence_UD_Weighted_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Sentence', topics='UD', method='Weighted_Jaccard')
    def get_Sentence_UD_Set_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Sentence', topics='UD', method='Set_Jaccard', Threshold=self.UD_Threshold)
    def get_Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Paragraph_Max', topics='NMF', method='Weighted_Jaccard')
    def get_Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Paragraph_Max', topics='NMF', method='Set_Jaccard',Threshold=self.NMF_Threshold)
    def get_Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Paragraph_Max', topics='UD', method='Weighted_Jaccard')
    def get_Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Paragraph_Max', topics='UD', method='Set_Jaccard',Threshold=self.UD_Threshold)
    def get_Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Paragraph_Mean', topics='NMF', method='Weighted_Jaccard')
    def get_Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Paragraph_Mean', topics='NMF', method='Set_Jaccard', Threshold=self.NMF_Threshold)
    def get_Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Paragraph_Mean', topics='UD', method='Weighted_Jaccard')
    def get_Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame(self):
        return self.get_Similarity_DataFrame(level='Paragraph_Mean', topics='UD', method='Set_Jaccard', Threshold=self.UD_Threshold)
    def get_NMF_UD_Similarity_DataFrame(self, level='Sentence', method='Weighted_Jaccard'):
        NMF_Topics=self.NMF_Topics
        UD_Topics=self.UD_Topics
        if level is 'Sentence':
            data=self.data[self.NMF_Topics_Names+self.UD_Topics_Names]
        elif level is 'Paragraph_Max':
            data=self.Paragraphs_Topic_Score_BY_Sentence_Max_Topic_Score[self.NMF_Topics_Names+self.UD_Topics_Names]
        elif level is 'Paragraph_Mean':
            data=self.Paragraphs_Topic_Score_BY_Sentence_Mean_Topic_Score[self.NMF_Topics_Names+self.UD_Topics_Names]
        return get_NMF_UD_Jaccard_Similarity_DataFrame(data,method='Weighted_Jaccard',NMF_Topics=self.NMF_Topics, UD_Topics=self.UD_Topics)
    def get_Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame(self):
        return self.get_NMF_UD_Similarity_DataFrame(level='Sentence', method='Weighted_Jaccard')
    def get_Sentence_NMF_UD_Set_Jaccard_Similarity_DataFrame(self):
        return self.get_NMF_UD_Similarity_DataFrame(level='Sentence', method='Set_Jaccard')
    def get_Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_DataFrame(self):
        return self.get_NMF_UD_Similarity_DataFrame(level='Paragraph_Max', method='Weighted_Jaccard')
    def get_Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_DataFrame(self):
        return self.get_NMF_UD_Similarity_DataFrame(level='Paragraph_Max', method='Set_Jaccard')
    def get_Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame(self):
        return self.get_NMF_UD_Similarity_DataFrame(level='Paragraph_Mean', method='Weighted_Jaccard')
    def get_Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_DataFrame(self):
        return self.get_NMF_UD_Similarity_DataFrame(level='Paragraph_Mean', method='Set_Jaccard')
    def get_Sentence_NMF_Weighted_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame.columns)):
                array.append(self.Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame.columns[i]+'-'+self.Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame[self.Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame, labels=labels, mask=True)
    def get_Sentence_NMF_Set_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Sentence_NMF_Set_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Sentence_NMF_Set_Embeddings_Similarity_DataFrame.columns)):
                array.append(self.Sentence_NMF_Set_Embeddings_Similarity_DataFrame.columns[i]+'-'+self.Sentence_NMF_Set_Embeddings_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Sentence_NMF_Set_Embeddings_Similarity_DataFrame[self.Sentence_NMF_Set_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Sentence_NMF_Set_Embeddings_Similarity_DataFrame, labels=labels, mask=True)
    def get_Sentence_UD_Weighted_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Sentence_UD_Weighted_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Sentence_UD_Weighted_Embeddings_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Sentence_UD_Weighted_Embeddings_Similarity_DataFrame[self.Sentence_UD_Weighted_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Sentence_UD_Weighted_Embeddings_Similarity_DataFrame, labels=labels,mask=True)
    def get_Sentence_UD_Set_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Sentence_UD_Set_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Sentence_UD_Set_Embeddings_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Sentence_UD_Set_Embeddings_Similarity_DataFrame[self.Sentence_UD_Set_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Sentence_UD_Set_Embeddings_Similarity_DataFrame, labels=labels,mask=True)
    def get_Sentence_NMF_UD_Weighted_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame[self.Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame.index.values
        xticklabels=self.Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels)
    def get_Sentence_NMF_UD_Set_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Sentence_NMF_UD_Set_Embeddings_Similarity_DataFrame[self.Sentence_NMF_UD_Set_Embeddings_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Sentence_NMF_Set_Embeddings_Similarity_DataFrame.index.values
        xticklabels=self.Sentence_NMF_UD_Set_Embeddings_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Sentence_NMF_UD_Set_Embeddings_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels)
    def get_Paragraph_Max_NMF_Weighted_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame.columns)):
                array.append(self.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame.columns[i]+'-'+self.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame[self.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame, labels=labels, mask=True)
    def get_Paragraph_Max_NMF_Set_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame.columns)):
                array.append(self.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame.columns[i]+'-'+self.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame[self.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame, labels=labels, mask=True)
    def get_Paragraph_Max_UD_Weighted_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame[self.Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame, labels=labels,mask=True)
    def get_Paragraph_Max_UD_Set_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame[self.Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame, labels=labels,mask=True)
    def get_Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_DataFrame[self.Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_DataFrame.index.values
        xticklabels=self.Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels)
    def get_Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_DataFrame[self.Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame.index.values
        xticklabels=self.Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels)

    def get_Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame.columns)):
                array.append(self.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame.columns[i]+'-'+self.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame[self.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame, labels=labels, mask=True)
    def get_Paragraph_Mean_NMF_Set_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame.columns)):
                array.append(self.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame.columns[i]+'-'+self.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame[self.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame, labels=labels, mask=True)
    def get_Paragraph_Mean_UD_Weighted_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame[self.Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame, labels=labels,mask=True)
    def get_Paragraph_Mean_UD_Set_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame[self.Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame, labels=labels,mask=True)
    def get_Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame[self.Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame.index.values
        xticklabels=self.Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels)
    def get_Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_DataFrame[self.Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame.index.values
        xticklabels=self.Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels) 

    def get_Sentence_NMF_Weighted_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame.columns)):
                array.append(self.Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame.columns[i]+'-'+self.Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame[self.Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame, labels=labels, mask=True)
    def get_Sentence_NMF_Set_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Sentence_NMF_Set_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Sentence_NMF_Set_Jaccard_Similarity_DataFrame.columns)):
                array.append(self.Sentence_NMF_Set_Jaccard_Similarity_DataFrame.columns[i]+'-'+self.Sentence_NMF_Set_Jaccard_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Sentence_NMF_Set_Jaccard_Similarity_DataFrame[self.Sentence_NMF_Set_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Sentence_NMF_Set_Jaccard_Similarity_DataFrame, labels=labels, mask=True)
    def get_Sentence_UD_Weighted_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Sentence_UD_Weighted_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Sentence_UD_Weighted_Jaccard_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Sentence_UD_Weighted_Jaccard_Similarity_DataFrame[self.Sentence_UD_Weighted_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Sentence_UD_Weighted_Jaccard_Similarity_DataFrame, labels=labels,mask=True)
    def get_Sentence_UD_Set_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Sentence_UD_Set_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Sentence_UD_Set_Jaccard_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Sentence_UD_Set_Jaccard_Similarity_DataFrame[self.Sentence_UD_Set_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Sentence_UD_Set_Jaccard_Similarity_DataFrame, labels=labels,mask=True)
    def get_Sentence_NMF_UD_Weighted_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame[self.Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame.index.values
        xticklabels=self.Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels)
    def get_Sentence_NMF_UD_Set_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Sentence_NMF_UD_Set_Jaccard_Similarity_DataFrame[self.Sentence_NMF_UD_Set_Jaccard_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Sentence_NMF_Set_Jaccard_Similarity_DataFrame.index.values
        xticklabels=self.Sentence_NMF_UD_Set_Jaccard_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Sentence_NMF_UD_Set_Jaccard_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels)
    def get_Paragraph_Max_NMF_Weighted_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame.columns)):
                array.append(self.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame.columns[i]+'-'+self.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame[self.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame, labels=labels, mask=True)
    def get_Paragraph_Max_NMF_Set_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame.columns)):
                array.append(self.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame.columns[i]+'-'+self.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame[self.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame, labels=labels, mask=True)
    def get_Paragraph_Max_UD_Weighted_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame[self.Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame, labels=labels,mask=True)
    def get_Paragraph_Max_UD_Set_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame[self.Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame, labels=labels,mask=True)
    def get_Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_DataFrame[self.Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_DataFrame.index.values
        xticklabels=self.Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels)
    def get_Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_DataFrame[self.Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame.index.values
        xticklabels=self.Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels)

    def get_Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame.columns)):
                array.append(self.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame.columns[i]+'-'+self.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame[self.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame, labels=labels, mask=True)
    def get_Paragraph_Mean_NMF_Set_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame.columns)):
                array.append(self.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame.columns[i]+'-'+self.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame.columns[j]+':'+str(float_formatter(self.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame[self.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame, labels=labels, mask=True)
    def get_Paragraph_Mean_UD_Weighted_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame[self.Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame, labels=labels,mask=True)
    def get_Paragraph_Mean_UD_Set_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(len(self.Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame.columns)):
            array=[]
            for j in range(len(self.Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame.columns)):
                array.append('UD'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame[self.Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame.columns[i]][j])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        return get_heatmap(self.Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame, labels=labels,mask=True)
    def get_Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame[self.Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame.index.values
        xticklabels=self.Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels)
    def get_Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_HeatMap(self):
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.NMF_Topics):
            array=[]
            for j in range(self.UD_Topics):
                array.append('T'+str(i+1)+'-UD'+str(j+1)+':'+str(float_formatter(self.Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_DataFrame[self.Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_DataFrame.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        yticklabels=self.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame.index.values
        xticklabels=self.Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_DataFrame[0:self.UD_Topics]
        return get_heatmap(self.Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_DataFrame, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels)
    
    def get_Company_Sentence_to_NMF_Topic_Distribution_Heatmap(self, No_of_top_words=5):
        self.Company_Wise_Sentence_to_NMF_Topic_Distribution=self.Company_Wise_No_Of_Sentence_Scoring_Above_Threshold_For_Each_Topic[self.NMF_Topics_Names].div(self.Company_Wise_No_Of_Sentence_Scoring_Above_Threshold_For_Each_Topic[self.NMF_Topics_Names].sum(axis=1), axis=0)
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.Companies):
            array=[]
            for j in range(self.NMF_Topics):
                array.append(self.Company_Names[i]+'-NMF'+str(j+1)+':'+str(float_formatter(self.Company_Wise_Sentence_to_NMF_Topic_Distribution[self.Company_Wise_Sentence_to_NMF_Topic_Distribution.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        Top_Words=[', '.join(self.Top_Words[i][0:No_of_top_words]) for i in range(len(self.Top_Words))]
        Topics=[self.NMF_Topics_Names[i]+", "+Top_Words[i] for i in range(self.NMF_Topics)]
        xticklabels=Topics
        yticklabels=self.Company_Names
        return self.Company_Wise_Sentence_to_NMF_Topic_Distribution,get_heatmap(self.Company_Wise_Sentence_to_NMF_Topic_Distribution, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels,wrap=True)
    def get_Company_Sentence_to_UD_Topic_Distribution_Heatmap(self,No_of_top_words=5):
        self.Company_Wise_Sentence_to_UD_Topic_Distribution=self.Company_Wise_No_Of_Sentence_Scoring_Above_Threshold_For_Each_Topic[self.UD_Topics_Names].div(self.Company_Wise_No_Of_Sentence_Scoring_Above_Threshold_For_Each_Topic[self.UD_Topics_Names].sum(axis=1), axis=0)
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.Companies):
            array=[]
            for j in range(self.UD_Topics):
                
                array.append(self.Company_Names[i]+'-UD'+str(j+1)+':'+str(float_formatter(self.Company_Wise_Sentence_to_UD_Topic_Distribution[self.Company_Wise_Sentence_to_UD_Topic_Distribution.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        Topics=[self.UD_Topics_Names[i] for i in range(self.UD_Topics)]
        xticklabels=Topics
        yticklabels=self.Company_Names
        return self.Company_Wise_Sentence_to_UD_Topic_Distribution,get_heatmap(self.Company_Wise_Sentence_to_UD_Topic_Distribution, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels,wrap=True)
    def get_Company_Paragraph_Max_to_NMF_Topic_Distribution_Heatmap(self,No_of_top_words=5):
        self.Company_Wise_Paragraph_Max_to_NMF_Topic_Distribution=self.Company_Wise_No_Of_Paragraph_Max_Scoring_Above_Threshold_For_Each_Topic[self.NMF_Topics_Names].div(self.Company_Wise_No_Of_Paragraph_Max_Scoring_Above_Threshold_For_Each_Topic[self.NMF_Topics_Names].sum(axis=1), axis=0)
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.Companies):
            array=[]
            for j in range(self.NMF_Topics):
                
                array.append(self.Company_Names[i]+'-NMF'+str(j+1)+':'+str(float_formatter(self.Company_Wise_Paragraph_Max_to_NMF_Topic_Distribution[self.Company_Wise_Paragraph_Max_to_NMF_Topic_Distribution.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        Top_Words=[', '.join(self.Top_Words[i][0:No_of_top_words]) for i in range(len(self.Top_Words))]
        Topics=[self.NMF_Topics_Names[i]+", "+Top_Words[i] for i in range(self.NMF_Topics)]
        xticklabels=Topics
        yticklabels=self.Company_Names
        return self.Company_Wise_Paragraph_Max_to_NMF_Topic_Distribution,get_heatmap(self.Company_Wise_Paragraph_Max_to_NMF_Topic_Distribution, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels,wrap=True)
    def get_Company_Paragraph_Max_to_UD_Topic_Distribution_Heatmap(self,No_of_top_words=5):
        self.Company_Wise_Paragraph_Max_to_UD_Topic_Distribution=self.Company_Wise_No_Of_Paragraph_Max_Scoring_Above_Threshold_For_Each_Topic[self.UD_Topics_Names].div(self.Company_Wise_No_Of_Paragraph_Max_Scoring_Above_Threshold_For_Each_Topic[self.UD_Topics_Names].sum(axis=1), axis=0)
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.Companies):
            array=[]
            for j in range(self.UD_Topics):
                
                array.append(self.Company_Names[i]+'-UD'+str(j+1)+':'+str(float_formatter(self.Company_Wise_Paragraph_Max_to_UD_Topic_Distribution[self.Company_Wise_Paragraph_Max_to_UD_Topic_Distribution.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        Topics=[self.UD_Topics_Names[i] for i in range(self.UD_Topics)]
        xticklabels=Topics
        yticklabels=self.Company_Names
        return self.Company_Wise_Paragraph_Max_to_UD_Topic_Distribution,get_heatmap(self.Company_Wise_Paragraph_Max_to_UD_Topic_Distribution, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels,wrap=True)
        
    def get_Company_Paragraph_Mean_to_NMF_Topic_Distribution_Heatmap(self,No_of_top_words=5):
        self.Company_Wise_Paragraph_Mean_to_NMF_Topic_Distribution=self.Company_Wise_No_Of_Paragraph_Mean_Scoring_Above_Threshold_For_Each_Topic[self.NMF_Topics_Names].div(self.Company_Wise_No_Of_Paragraph_Mean_Scoring_Above_Threshold_For_Each_Topic[self.NMF_Topics_Names].sum(axis=1), axis=0)
        
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.Companies):
            array=[]
            for j in range(self.NMF_Topics):
                array.append(self.Company_Names[i]+'-NMF'+str(j+1)+':'+str(float_formatter(self.Company_Wise_Paragraph_Mean_to_NMF_Topic_Distribution[self.Company_Wise_Paragraph_Max_to_NMF_Topic_Distribution.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        Top_Words=[', '.join(self.Top_Words[i][0:No_of_top_words]) for i in range(len(self.Top_Words))]
        Topics=[self.NMF_Topics_Names[i]+", "+Top_Words[i] for i in range(self.NMF_Topics)]
        xticklabels=Topics
        yticklabels=self.Company_Names
        return self.Company_Wise_Paragraph_Mean_to_NMF_Topic_Distribution,get_heatmap(self.Company_Wise_Paragraph_Mean_to_NMF_Topic_Distribution, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels,wrap=True)
    def get_Company_Paragraph_Mean_to_UD_Topic_Distribution_Heatmap(self,No_of_top_words=5):
        self.Company_Wise_Paragraph_Mean_to_UD_Topic_Distribution=self.Company_Wise_No_Of_Paragraph_Mean_Scoring_Above_Threshold_For_Each_Topic[self.UD_Topics_Names].div(self.Company_Wise_No_Of_Paragraph_Mean_Scoring_Above_Threshold_For_Each_Topic[self.UD_Topics_Names].sum(axis=1), axis=0)
        labels=[]
        float_formatter = "{:.2f}".format
        for i in range(self.Companies):
            array=[]
            for j in range(self.UD_Topics):
                array.append(self.Company_Names[i]+'-UD'+str(j+1)+':'+str(float_formatter(self.Company_Wise_Paragraph_Mean_to_UD_Topic_Distribution[self.Company_Wise_Paragraph_Max_to_UD_Topic_Distribution.columns[j]][i])))
            labels.append(array)
        labels=np.array(labels, dtype = str)
        Topics=[self.UD_Topics_Names[i] for i in range(self.UD_Topics)]
        xticklabels=Topics
        yticklabels=self.Company_Names
        return self.Company_Wise_Paragraph_Mean_to_UD_Topic_Distribution, get_heatmap(self.Company_Wise_Paragraph_Mean_to_UD_Topic_Distribution, labels=labels,xticklabels=xticklabels,yticklabels=yticklabels,wrap=True)
    def to_String(self):
        return 'Total_Companies='+str(self.Companies)+'\n'+'Total_Sentences='+str(self.Sentences)+'\n'+'Total_Paragraphs='+str(self.Paragraphs)+'\n'+'Total_NMF_Topics='+str(self.NMF_Topics)+'\n'+'Total_UD_Topics='+str(self.UD_Topics)+'\n'+'NMF_Percentile_threshold='+str(self.NMF_Percentile_Threshold)+'\n'+'NMF_Threshold='+str(self.NMF_Threshold)+'\n'+'UD_Threshold='+str(self.UD_Threshold)+'\n'+'Sentences_Highest_NMF_Score_Above_Threshold='+str(self.NO_Sentences_Highest_NMF_Score_Above_Threshold)+'\n'+'Sentences_Highest_UD_Score_Above_Threshold='+str(self.No_Sentences_Highest_UD_Score_Above_Threshold)+'\n'+'Paragraphs_Max_Highest_NMF_Score_Above_Threshold='+str(self.NO_Paragraphs_Max_Highest_NMF_Score_Above_Threshold)+'\n'+'Paragraphs_Max_Highest_UD_Score_Above_Threshold='+str(self.No_Paragraphs_Max_Highest_UD_Score_Above_Threshold)+'\n'+'Paragraphs_Mean_Highest_NMF_Score_Above_Threshold='+str(self.NO_Paragraphs_Mean_Highest_NMF_Score_Above_Threshold)+'\n'+'Paragraphs_Mean_Highest_UD_Score_Above_Threshold='+str(self.No_Paragraphs_Mean_Highest_UD_Score_Above_Threshold)
    def get_Company_Wise_Analysis(self, set_values=True):
        CA=Company_Wise_Analysis([self.get_Company_Sentence_to_NMF_Topic_Distribution_Heatmap(),self.get_Company_Sentence_to_UD_Topic_Distribution_Heatmap(),self.get_Company_Paragraph_Max_to_NMF_Topic_Distribution_Heatmap(),self.get_Company_Paragraph_Max_to_UD_Topic_Distribution_Heatmap(),self.get_Company_Paragraph_Mean_to_NMF_Topic_Distribution_Heatmap(),self.get_Company_Paragraph_Mean_to_UD_Topic_Distribution_Heatmap(),self.data[['Company']+list(self.NMF_Topics_Names)+list(self.UD_Topics_Names)].groupby('Company').mean()])
        if set_values:
            self.Company_Wise_Sentence_to_NMF_Topic_Distribution=CA.Company_Wise_Sentence_to_NMF_Topic_Distribution
            self.Company_Sentence_to_NMF_Topic_Distribution_Heatmap=CA.Company_Sentence_to_NMF_Topic_Distribution_Heatmap
            self.Company_Wise_Sentence_to_UD_Topic_Distribution=CA.Company_Wise_Sentence_to_UD_Topic_Distribution
            self.Company_Sentence_to_UD_Topic_Distribution_Heatmap=CA.Company_Sentence_to_UD_Topic_Distribution_Heatmap
            self.Company_Wise_Paragraph_Max_to_NMF_Topic_Distribution=CA.Company_Wise_Paragraph_Max_to_NMF_Topic_Distribution
            self.Company_Paragraph_Max_to_NMF_Topic_Distribution_Heatmap=CA.Company_Paragraph_Max_to_NMF_Topic_Distribution_Heatmap
            self.Company_Wise_Paragraph_Max_to_UD_Topic_Distribution=CA.Company_Wise_Paragraph_Max_to_UD_Topic_Distribution
            self.Company_Paragraph_Max_to_UD_Topic_Distribution_Heatmap=CA.Company_Paragraph_Max_to_UD_Topic_Distribution_Heatmap
            self.Company_Wise_Paragraph_Mean_to_NMF_Topic_Distribution=CA.Company_Wise_Paragraph_Mean_to_NMF_Topic_Distribution
            self.Company_Paragraph_Mean_to_NMF_Topic_Distribution_Heatmap=CA.Company_Paragraph_Mean_to_NMF_Topic_Distribution_Heatmap
            self.Company_Wise_Paragraph_Mean_to_UD_Topic_Distribution=CA.Company_Wise_Paragraph_Mean_to_UD_Topic_Distribution
            self.Company_Paragraph_Mean_to_UD_Topic_Distribution_Heatmap=CA.Company_Paragraph_Mean_to_UD_Topic_Distribution_Heatmap
            self.Company_Topic_Score_By_Sentence_Mean_Topic_Score=CA.Company_Topic_Score_By_Sentence_Mean_Topic_Score
        return CA
    def get_Basic_Analysis(self,level='Sentence',set_values=True):
        if level is 'Sentence':
            BA=Basic_Analysis([self.get_Sentence_filtered_data(),self.get_Sentence_NMF_Score_Histogram(),self.get_Sentence_UD_Score_Histogram(),self.get_Sentence_NMF_Highest_Score_For_Any_Topic_Histogram(),self.get_Sentence_UD_Highest_Score_For_Any_Topic_Histogram(),self.get_Sentence_NMF_Filterd_Score_Histogram(),self.get_Sentence_UD_Filtered_Score_Histogram(),self.get_NMF_Topic_Sentence_Distribution(),self.get_UD_Topic_Sentence_Distribution(),self.get_NMF_Highest_Score_Topic_Sentence_Distribution(),self.get_UD_Highest_Score_Topic_Sentence_Distribution()])
            if set_values:
                self.Sentence_filtered_data=BA.filtered_data
                self.Sentence_NMF_Score_Histogram=BA.NMF_Score_Histogram
                self.Sentence_UD_Score_Histogram=BA.UD_Score_Histogram
                self.Sentence_NMF_Highest_Score_For_Any_Topic_Histogram=BA.NMF_Highest_Score_For_Any_Topic_Histogram
                self.Sentence_UD_Highest_Score_For_Any_Topic_Histogram=BA.UD_Highest_Score_For_Any_Topic_Histogram
                self.Sentence_NMF_Filterd_Score_Histogram=BA.NMF_Filterd_Score_Histogram
                self.Sentence_UD_Filtered_Score_Histogram=BA.UD_Filtered_Score_Histogram
                self.NMF_Topic_Sentence_Distribution=BA.NMF_Topic_Distribution
                self.UD_Topic_Sentence_Distribution=BA.UD_Topic_Distribution
                self.NMF_Highest_Score_Topic_Sentence_Distribution=BA.NMF_Highest_Score_Topic_Distribution
                self.UD_Highest_Score_Topic_Sentence_Distribution=BA.UD_Highest_Score_Topic_Distribution
            return BA
        elif level is 'Paragraph_Max':
            BA=Basic_Analysis([self.get_Paragraph_Max_filtered_data(),self.get_Paragraph_Max_NMF_Score_Histogram(),self.get_Paragraph_Max_UD_Score_Histogram(),self.get_Paragraph_Max_NMF_Highest_Score_For_Any_Topic_Histogram(),self.get_Paragraph_Max_UD_Highest_Score_For_Any_Topic_Histogram(),self.get_Paragraph_Max_NMF_Filterd_Score_Histogram(),self.get_Paragraph_Max_UD_Filterd_Score_Histogram(),self.get_NMF_Topic_Paragraph_Max_Distribution(),self.get_UD_Topic_Paragraph_Max_Distribution(),self.get_NMF_Highest_Score_Topic_Paragraph_Max_Distribution(),self.get_UD_Highest_Score_Topic_Paragraph_Max_Distribution()])
            if set_values:
                self.Paragraph_Max_filtered_data=BA.filtered_data
                self.Paragraph_Max_NMF_Score_Histogram=BA.NMF_Score_Histogram
                self.Paragraph_Max_UD_Score_Histogram=BA.UD_Score_Histogram
                self.Paragraph_Max_NMF_Highest_Score_For_Any_Topic_Histogram=BA.NMF_Highest_Score_For_Any_Topic_Histogram
                self.Paragraph_Max_UD_Highest_Score_For_Any_Topic_Histogram=BA.UD_Highest_Score_For_Any_Topic_Histogram
                self.Paragraph_Max_NMF_Filterd_Score_Histogram=BA.NMF_Filterd_Score_Histogram
                self.Paragraph_Max_UD_Filtered_Score_Histogram=BA.UD_Filtered_Score_Histogram
                self.NMF_Topic_Paragraph_Max_Distribution=BA.NMF_Topic_Distribution
                self.UD_Topic_Paragraph_Max_Distribution=BA.UD_Topic_Distribution
                self.NMF_Highest_Score_Topic_Paragraph_Max_Distribution=BA.NMF_Highest_Score_Topic_Distribution
                self.UD_Highest_Score_Topic_Paragraph_Max_Distribution=BA.UD_Highest_Score_Topic_Distribution
            return BA
        elif level is 'Paragraph_Mean':
            BA=Basic_Analysis([self.get_Paragraph_Mean_filtered_data(),self.get_Paragraph_Mean_NMF_Score_Histogram(),self.get_Paragraph_Mean_UD_Score_Histogram(),self.get_Paragraph_Mean_NMF_Highest_Score_For_Any_Topic_Histogram(),self.get_Paragraph_Mean_UD_Highest_Score_For_Any_Topic_Histogram(),self.get_Paragraph_Mean_NMF_Filterd_Score_Histogram(),self.get_Paragraph_Mean_UD_Filterd_Score_Histogram(),self.get_NMF_Topic_Paragraph_Mean_Distribution(),self.get_UD_Topic_Paragraph_Mean_Distribution(),self.get_NMF_Highest_Score_Topic_Paragraph_Mean_Distribution(),self.get_UD_Highest_Score_Topic_Paragraph_Mean_Distribution()])
            if set_values:
                self.Paragraph_Mean_filtered_data=BA.filtered_data
                self.Paragraph_Mean_NMF_Score_Histogram=BA.NMF_Score_Histogram
                self.Paragraph_Mean_UD_Score_Histogram=BA.UD_Score_Histogram
                self.Paragraph_Mean_NMF_Highest_Score_For_Any_Topic_Histogram=BA.NMF_Highest_Score_For_Any_Topic_Histogram
                self.Paragraph_Mean_UD_Highest_Score_For_Any_Topic_Histogram=BA.UD_Highest_Score_For_Any_Topic_Histogram
                self.Paragraph_Mean_NMF_Filterd_Score_Histogram=BA.NMF_Filterd_Score_Histogram
                self.Paragraph_Mean_UD_Filtered_Score_Histogram=BA.UD_Filtered_Score_Histogram
                self.NMF_Topic_Paragraph_Mean_Distribution=BA.NMF_Topic_Distribution
                self.UD_Topic_Paragraph_Mean_Distribution=BA.UD_Topic_Distribution
                self.NMF_Highest_Score_Topic_Paragraph_Mean_Distribution=BA.NMF_Highest_Score_Topic_Distribution
                self.UD_Highest_Score_Topic_Paragraph_Mean_Distribution=BA.UD_Highest_Score_Topic_Distribution
            return BA
    def get_Similarity_Analysis_DataFrame(self,level='Sentence',Similarity_Method='Weighted_Jaccard',set_values=True):
        if level is 'Sentence':
            if Similarity_Method is 'Weighted_Jaccard':
                SAD=Similarity_Analysis_DataFrame([self.get_Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame(),self.get_Sentence_UD_Weighted_Jaccard_Similarity_DataFrame(),self.get_Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame()])
                if set_values:
                    self.Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame=SAD.NMF_Similarity_DataFrame
                    self.Sentence_UD_Weighted_Jaccard_Similarity_DataFrame=SAD.UD_Similarity_DataFrame
                    self.Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame=SAD.NMF_UD_Similarity_DataFrame
                    return SAD
            if Similarity_Method is'Set_Jaccard':
                SAD=Similarity_Analysis_DataFrame([self.get_Sentence_NMF_Set_Jaccard_Similarity_DataFrame(),self.get_Sentence_UD_Set_Jaccard_Similarity_DataFrame(),self.get_Sentence_NMF_UD_Set_Jaccard_Similarity_DataFrame()])
                if set_values:
                    self.Sentence_NMF_Set_Jaccard_Similarity_DataFrame=SAD.NMF_Similarity_DataFrame
                    self.Sentence_UD_Set_Jaccard_Similarity_DataFrame=SAD.UD_Similarity_DataFrame
                    self.Sentence_NMF_UD_Set_Jaccard_Similarity_DataFrame=SAD.NMF_UD_Similarity_DataFrame
                    return SAD
        if level is 'Paragraph_Max':
            if Similarity_Method is 'Weighted_Jaccard':
                SAD=Similarity_Analysis_DataFrame([self.get_Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame(),self.get_Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame(),self.get_Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_DataFrame()])
                if set_values:
                    self.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame=SAD.NMF_Similarity_DataFrame
                    self.Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame=SAD.UD_Similarity_DataFrame
                    self.Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_DataFrame=SAD.NMF_UD_Similarity_DataFrame
                    return SAD
            if Similarity_Method is'Set_Jaccard':
                SAD=Similarity_Analysis_DataFrame([self.get_Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame(),self.get_Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame(),self.get_Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_DataFrame()])
                if set_values:
                    self.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame=SAD.NMF_Similarity_DataFrame
                    self.Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame=SAD.UD_Similarity_DataFrame
                    self.Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_DataFrame=SAD.NMF_UD_Similarity_DataFrame
                    return SAD
        if level is 'Paragraph_Mean':
            if Similarity_Method is 'Weighted_Jaccard':
                SAD=Similarity_Analysis_DataFrame([self.get_Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame(),self.get_Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame(),self.get_Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame()])
                if set_values:
                    self.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame=SAD.NMF_Similarity_DataFrame
                    self.Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame=SAD.UD_Similarity_DataFrame
                    self.Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame=SAD.NMF_UD_Similarity_DataFrame
                    return SAD
            if Similarity_Method is'Set_Jaccard':
                SAD=Similarity_Analysis_DataFrame([self.get_Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame(),self.get_Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame(),self.get_Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_DataFrame()])
                if set_values:
                    self.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame=SAD.NMF_Similarity_DataFrame
                    self.Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame=SAD.UD_Similarity_DataFrame
                    self.Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_DataFrame=SAD.NMF_UD_Similarity_DataFrame
                    return SAD
    def get_Similarity_Analysis_HeatMap(self,level='Sentence',Similarity_Method='Weighted_Jaccard',set_values=True):
        if level is 'Sentence':
            if Similarity_Method is 'Weighted_Jaccard':
                SAH=Similarity_Analysis_HeatMap([self.get_Sentence_NMF_Weighted_Jaccard_Similarity_HeatMap(),self.get_Sentence_UD_Weighted_Jaccard_Similarity_HeatMap(),self.get_Sentence_NMF_UD_Weighted_Jaccard_Similarity_HeatMap()])
                if set_values:
                    self.Sentence_NMF_Weighted_Jaccard_Similarity_HeatMap=SAH.NMF_Similarity_HeatMap
                    self.Sentence_UD_Weighted_Jaccard_Similarity_HeatMap=SAH.UD_Similarity_HeatMap
                    self.Sentence_NMF_UD_Weighted_Jaccard_Similarity_HeatMap=SAH.NMF_UD_Similarity_HeatMap
                    return SAH
            if Similarity_Method is'Set_Jaccard':
                SAH=Similarity_Analysis_HeatMap([self.get_Sentence_NMF_Set_Jaccard_Similarity_HeatMap(),self.get_Sentence_UD_Set_Jaccard_Similarity_HeatMap(),self.get_Sentence_NMF_UD_Set_Jaccard_Similarity_HeatMap()])
                if set_values:
                    self.Sentence_NMF_Set_Jaccard_Similarity_HeatMap=SAH.NMF_Similarity_HeatMap
                    self.Sentence_UD_Set_Jaccard_Similarity_HeatMap=SAH.UD_Similarity_HeatMap
                    self.Sentence_NMF_UD_Set_Jaccard_Similarity_HeatMap=SAH.NMF_UD_Similarity_HeatMap
                    return SAH
        if level is 'Paragraph_Max':
            if Similarity_Method is 'Weighted_Jaccard':
                SAH=Similarity_Analysis_HeatMap([self.get_Paragraph_Max_NMF_Weighted_Jaccard_Similarity_HeatMap(),self.get_Paragraph_Max_UD_Weighted_Jaccard_Similarity_HeatMap(),self.get_Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_HeatMap(),])
                if set_values:
                    self.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_HeatMap=SAH.NMF_Similarity_HeatMap
                    self.Paragraph_Max_UD_Weighted_Jaccard_Similarity_HeatMap=SAH.UD_Similarity_HeatMap
                    self.Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_HeatMap=SAH.NMF_UD_Similarity_HeatMap
                    return SAH
            if Similarity_Method is'Set_Jaccard':
                SAH=Similarity_Analysis_HeatMap([self.get_Paragraph_Max_NMF_Set_Jaccard_Similarity_HeatMap(),self.get_Paragraph_Max_UD_Set_Jaccard_Similarity_HeatMap(),self.get_Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_HeatMap()])
                if set_values:
                    self.Paragraph_Max_NMF_Set_Jaccard_Similarity_HeatMap=SAH.NMF_Similarity_HeatMap
                    self.Paragraph_Max_UD_Set_Jaccard_Similarity_HeatMap=SAH.UD_Similarity_HeatMap
                    self.Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_HeatMap=SAH.NMF_UD_Similarity_HeatMap
                    return SAH
        if level is 'Paragraph_Mean':
            if Similarity_Method is 'Weighted_Jaccard':
                SAH=Similarity_Analysis_HeatMap([self.get_Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_HeatMap(),self.get_Paragraph_Mean_UD_Weighted_Jaccard_Similarity_HeatMap(),self.get_Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_HeatMap()])
                if set_values:
                    self.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_HeatMap=SAH.NMF_Similarity_HeatMap
                    self.Paragraph_Mean_UD_Weighted_Jaccard_Similarity_HeatMap=SAH.UD_Similarity_HeatMap
                    self.Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_HeatMap=SAH.NMF_UD_Similarity_HeatMap
                    return SAH
            if Similarity_Method is'Set_Jaccard':
                SAH=Similarity_Analysis_HeatMap([self.get_Paragraph_Mean_NMF_Set_Jaccard_Similarity_HeatMap(),self.get_Paragraph_Mean_UD_Set_Jaccard_Similarity_HeatMap(),self.get_Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_HeatMap()])
                if set_values:
                    self.Paragraph_Mean_NMF_Set_Jaccard_Similarity_HeatMap=SAH.NMF_Similarity_HeatMap
                    self.Paragraph_Mean_UD_Set_Jaccard_Similarity_HeatMap=SAH.UD_Similarity_HeatMap
                    self.Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_HeatMap=SAH.NMF_UD_Similarity_HeatMap
                    return SAH
    def get_Embeddings_Similarity_Analysis_DataFrame(self,level='Sentence',Similarity_Method='Weighted_Embeddings',set_values=True):
        if level is 'Sentence':
            if Similarity_Method is 'Weighted_Embeddings':
                SAD=Embeddings_Similarity_Analysis_DataFrame([self.get_Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame()[1],self.get_Sentence_UD_Weighted_Embeddings_Similarity_DataFrame()[1],self.get_Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame()[1]])
                print
                if set_values:
                    self.Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame=SAD.NMF_Embeddings_Similarity_DataFrame
                    self.Sentence_UD_Weighted_Embeddings_Similarity_DataFrame=SAD.UD_Embeddings_Similarity_DataFrame
                    self.Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame=SAD.NMF_UD_Embeddings_Similarity_DataFrame
                    return SAD
            if Similarity_Method is'Set_Embeddings':
                SAD=Embeddings_Similarity_Analysis_DataFrame([self.get_Sentence_NMF_Set_Embeddings_Similarity_DataFrame()[1],self.get_Sentence_UD_Set_Embeddings_Similarity_DataFrame()[1],self.get_Sentence_NMF_UD_Set_Embeddings_Similarity_DataFrame()[1]])
                if set_values:
                    self.Sentence_NMF_Set_Embeddings_Similarity_DataFrame=SAD.NMF_Embeddings_Similarity_DataFrame
                    self.Sentence_UD_Set_Embeddings_Similarity_DataFrame=SAD.UD_Embeddings_Similarity_DataFrame
                    self.Sentence_NMF_UD_Set_Embeddings_Similarity_DataFrame=SAD.NMF_UD_Embeddings_Similarity_DataFrame
                    return SAD
        if level is 'Paragraph_Max':
            if Similarity_Method is 'Weighted_Embeddings':
                SAD=Embeddings_Similarity_Analysis_DataFrame([self.get_Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame()[1],self.get_Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame()[1],self.get_Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_DataFrame()[1]])
                if set_values:
                    self.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame=SAD.NMF_Embeddings_Similarity_DataFrame
                    self.Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame=SAD.UD_Embeddings_Similarity_DataFrame
                    self.Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_DataFrame=SAD.NMF_UD_Embeddings_Similarity_DataFrame
                    return SAD
            if Similarity_Method is'Set_Embeddings':
                SAD=Embeddings_Similarity_Analysis_DataFrame([self.get_Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame()[1],self.get_Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame()[1],self.get_Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_DataFrame()[1]])
                if set_values:
                    self.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame=SAD.NMF_Embeddings_Similarity_DataFrame
                    self.Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame=SAD.UD_Embeddings_Similarity_DataFrame
                    self.Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_DataFrame=SAD.NMF_UD_Embeddings_Similarity_DataFrame
                    return SAD
        if level is 'Paragraph_Mean':
            if Similarity_Method is 'Weighted_Embeddings':
                SAD=Embeddings_Similarity_Analysis_DataFrame([self.get_Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame()[1],self.get_Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame()[1],self.get_Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame()[1]])
                if set_values:
                    self.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame=SAD.NMF_Embeddings_Similarity_DataFrame
                    self.Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame=SAD.UD_Embeddings_Similarity_DataFrame
                    self.Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame=SAD.NMF_UD_Embeddings_Similarity_DataFrame
                    return SAD
            if Similarity_Method is'Set_Embeddings':
                SAD=Embeddings_Similarity_Analysis_DataFrame([self.get_Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame()[1],self.get_Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame()[1],self.get_Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_DataFrame()[1]])
                if set_values:
                    self.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame=SAD.NMF_Embeddings_Similarity_DataFrame
                    self.Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame=SAD.UD_Embeddings_Similarity_DataFrame
                    self.Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_DataFrame=SAD.NMF_UD_Embeddings_Similarity_DataFrame
                    return SAD
                
    def get_Embeddings_Similarity_Analysis_HeatMap(self,level='Sentence',Similarity_Method='Weighted_Embeddings',set_values=True):
        if level is 'Sentence':
            if Similarity_Method is 'Weighted_Embeddings':
                SAH=Embeddings_Similarity_Analysis_HeatMap([self.get_Sentence_NMF_Weighted_Embeddings_Similarity_HeatMap(),self.get_Sentence_UD_Weighted_Embeddings_Similarity_HeatMap(),self.get_Sentence_NMF_UD_Weighted_Embeddings_Similarity_HeatMap()])
                if set_values:
                    self.Sentence_NMF_Weighted_Embeddings_Similarity_HeatMap=SAH.NMF_Embeddings_Similarity_HeatMap
                    self.Sentence_UD_Weighted_Embeddings_Similarity_HeatMap=SAH.UD_Embeddings_Similarity_HeatMap
                    self.Sentence_NMF_UD_Weighted_Embeddings_Similarity_HeatMap=SAH.NMF_UD_Embeddings_Similarity_HeatMap
                    return SAH
            if Similarity_Method is'Set_Embeddings':
                SAH=Embeddings_Similarity_Analysis_HeatMap([self.get_Sentence_NMF_Set_Embeddings_Similarity_HeatMap(),self.get_Sentence_UD_Set_Embeddings_Similarity_HeatMap(),self.get_Sentence_NMF_UD_Set_Embeddings_Similarity_HeatMap()])
                if set_values:
                    self.Sentence_NMF_Set_Embeddings_Similarity_HeatMap=SAH.NMF_Embeddings_Similarity_HeatMap
                    self.Sentence_UD_Set_Embeddings_Similarity_HeatMap=SAH.UD_Embeddings_Similarity_HeatMap
                    self.Sentence_NMF_UD_Set_Embeddings_Similarity_HeatMap=SAH.NMF_UD_Embeddings_Similarity_HeatMap
                    return SAH
        if level is 'Paragraph_Max':
            if Similarity_Method is 'Weighted_Embeddings':
                SAH=Embeddings_Similarity_Analysis_HeatMap([self.get_Paragraph_Max_NMF_Weighted_Embeddings_Similarity_HeatMap(),self.get_Paragraph_Max_UD_Weighted_Embeddings_Similarity_HeatMap(),self.get_Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_HeatMap(),])
                if set_values:
                    self.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_HeatMap=SAH.NMF_Embeddings_Similarity_HeatMap
                    self.Paragraph_Max_UD_Weighted_Embeddings_Similarity_HeatMap=SAH.UD_Embeddings_Similarity_HeatMap
                    self.Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_HeatMap=SAH.NMF_UD_Embeddings_Similarity_HeatMap
                    return SAH
            if Similarity_Method is'Set_Embeddings':
                SAH=Embeddings_Similarity_Analysis_HeatMap([self.get_Paragraph_Max_NMF_Set_Embeddings_Similarity_HeatMap(),self.get_Paragraph_Max_UD_Set_Embeddings_Similarity_HeatMap(),self.get_Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_HeatMap()])
                if set_values:
                    self.Paragraph_Max_NMF_Set_Embeddings_Similarity_HeatMap=SAH.NMF_Embeddings_Similarity_HeatMap
                    self.Paragraph_Max_UD_Set_Embeddings_Similarity_HeatMap=SAH.UD_Embeddings_Similarity_HeatMap
                    self.Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_HeatMap=SAH.NMF_UD_Embeddings_Similarity_HeatMap
                    return SAH
        if level is 'Paragraph_Mean':
            if Similarity_Method is 'Weighted_Embeddings':
                SAH=Embeddings_Similarity_Analysis_HeatMap([self.get_Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_HeatMap(),self.get_Paragraph_Mean_UD_Weighted_Embeddings_Similarity_HeatMap(),self.get_Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_HeatMap()])
                if set_values:
                    self.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_HeatMap=SAH.NMF_Embeddings_Similarity_HeatMap
                    self.Paragraph_Mean_UD_Weighted_Embeddings_Similarity_HeatMap=SAH.UD_Embeddings_Similarity_HeatMap
                    self.Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_HeatMap=SAH.NMF_UD_Embeddings_Similarity_HeatMap
                    return SAH
            if Similarity_Method is'Set_Embeddings':
                SAH=Embeddings_Similarity_Analysis_HeatMap([self.get_Paragraph_Mean_NMF_Set_Embeddings_Similarity_HeatMap(),self.get_Paragraph_Mean_UD_Set_Embeddings_Similarity_HeatMap(),self.get_Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_HeatMap()])
                if set_values:
                    self.Paragraph_Mean_NMF_Set_Embeddings_Similarity_HeatMap=SAH.NMF_Embeddings_Similarity_HeatMap
                    self.Paragraph_Mean_UD_Set_Embeddings_Similarity_HeatMap=SAH.UD_Embeddings_Similarity_HeatMap
                    self.Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_HeatMap=SAH.NMF_UD_Embeddings_Similarity_HeatMap
                    return SAH
def main():
    config = ConfigParser()
    config.read('/Users/arun/Documents/GitHub/Topic_Analysis/Integrated_TM_config.ini')
    print(config.sections())
    dataset=config['Mandatory']['dataset']
    topic_top_words=config['Mandatory']['topic_top_words']
    NMF_Topics_Start=config['Mandatory'].getint('nmf_topics_start')
    NMF_Topics=config['Mandatory'].getint('nmf_topics')
    UD_Topics_Start=config['Mandatory'].getint('ud_topics_start')
    UD_Topics=config['Mandatory'].getint('ud_topics')
    NMF_Percentile_Threshold=config['Mandatory'].getint('nmf_percentile_threshold')
    UD_Threshold=config['Mandatory'].getfloat('ud_threshold')
    Results_Folder=config['Mandatory']['results_folder']
    if not os.path.exists(Results_Folder):
        os.makedirs(Results_Folder)
    Folder_To_Store=Results_Folder+'/'+config['Mandatory']['folder_to_store']
    if not os.path.exists(Folder_To_Store):
        os.makedirs(Folder_To_Store)
    TA=Topic_Analysis(dataset, topic_top_words, NMF_Topics_Start=NMF_Topics_Start,NMF_Topics=NMF_Topics,UD_Topics_Start=UD_Topics_Start, UD_Topics=UD_Topics,NMF_Percentile_Threshold=NMF_Percentile_Threshold, UD_Threshold=UD_Threshold)
    with open(Folder_To_Store+'/TA_State.txt', 'w') as f:
        f.write(TA.to_String())
    f.close()
    if config['Company_Analysis'].getboolean('ca'):
        set_values=config['Company_Analysis'].getboolean('set_values')
        TACWA=TA.get_Company_Wise_Analysis(set_values=set_values)
        CA_To_Store_Folder = Results_Folder+'/'+config['Company_Analysis']['ca_to_store_folder']
        if not os.path.exists(CA_To_Store_Folder):
            os.makedirs(CA_To_Store_Folder)
        TACWA.Company_Wise_Sentence_to_NMF_Topic_Distribution.to_csv(CA_To_Store_Folder+'/Company_Wise_Sentence_to_NMF_Topic_Distribution.csv')
        TACWA.Company_Wise_Sentence_to_UD_Topic_Distribution.to_csv(CA_To_Store_Folder+'/Company_Wise_Sentence_to_UD_Topic_Distribution.csv')
        TACWA.Company_Sentence_to_NMF_Topic_Distribution_Heatmap.figure.savefig(CA_To_Store_Folder+'/Company_Sentence_to_NMF_Topic_Distribution_Heatmap')
        TACWA.Company_Sentence_to_UD_Topic_Distribution_Heatmap.figure.savefig(CA_To_Store_Folder+'/Company_Sentence_to_UD_Topic_Distribution_Heatmap')
        TACWA.Company_Wise_Paragraph_Max_to_NMF_Topic_Distribution.to_csv(CA_To_Store_Folder+'/Company_Wise_Paragraph_Max_to_NMF_Topic_Distribution.csv')
        TACWA.Company_Wise_Paragraph_Max_to_UD_Topic_Distribution.to_csv(CA_To_Store_Folder+'/Company_Wise_Paragraph_Max_to_UD_Topic_Distribution.csv')
        TACWA.Company_Paragraph_Max_to_NMF_Topic_Distribution_Heatmap.figure.savefig(CA_To_Store_Folder+'/Company_Paragraph_Max_to_NMF_Topic_Distribution_Heatmap')
        TACWA.Company_Paragraph_Max_to_UD_Topic_Distribution_Heatmap.figure.savefig(CA_To_Store_Folder+'/Company_Paragraph_Max_to_UD_Topic_Distribution_Heatmap')
        TACWA.Company_Wise_Paragraph_Mean_to_NMF_Topic_Distribution.to_csv(CA_To_Store_Folder+'/Company_Wise_Paragraph_Mean_to_NMF_Topic_Distribution.csv')
        TACWA.Company_Wise_Paragraph_Mean_to_UD_Topic_Distribution.to_csv(CA_To_Store_Folder+'/Company_Wise_Paragraph_Mean_to_UD_Topic_Distribution.csv')
        TACWA.Company_Paragraph_Mean_to_NMF_Topic_Distribution_Heatmap.figure.savefig(CA_To_Store_Folder+'/Company_Paragraph_Mean_to_NMF_Topic_Distribution_Heatmap')
        TACWA.Company_Paragraph_Mean_to_UD_Topic_Distribution_Heatmap.figure.savefig(CA_To_Store_Folder+'/Company_Paragraph_Mean_to_UD_Topic_Distribution_Heatmap')
        TACWA.Company_Topic_Score_By_Sentence_Mean_Topic_Score.to_csv(CA_To_Store_Folder+'/Company_Topic_Score_By_Sentence_Mean_Topic_Score.csv')
    if config['Basic_Analysis'].getboolean('ba'):
        Basic_analysis_Folder=Results_Folder+'/'+config['Basic_Analysis']['basic_analysis_folder']
        if not os.path.exists(Basic_analysis_Folder):
                os.makedirs(Basic_analysis_Folder)
        if config['Basic_Analysis'].getboolean('sentence'):
            set_values=config['Basic_Analysis'].getboolean('sentence_set_values')
            TABA_Sentence=TA.get_Basic_Analysis(level='Sentence',set_values=set_values)
            Sentence_Folder_To_Store=Basic_analysis_Folder+'/'+config['Basic_Analysis']['sentence_folder_to_store']
            if not os.path.exists(Sentence_Folder_To_Store):
                os.makedirs(Sentence_Folder_To_Store)
            TABA_Sentence.filtered_data.to_csv(Sentence_Folder_To_Store+'/Sentence_filtered_data.csv')
            TABA_Sentence.NMF_Score_Histogram[0].to_csv(Sentence_Folder_To_Store+'/Sentence_NMF_Score_Histogram.csv')
            TABA_Sentence.NMF_Score_Histogram[1].figure.savefig(Sentence_Folder_To_Store+'/Sentence_NMF_Score_Histogram')
            TABA_Sentence.UD_Score_Histogram[0].to_csv(Sentence_Folder_To_Store+'/Sentence_UD_Score_Histogram.csv')
            TABA_Sentence.UD_Score_Histogram[1].figure.savefig(Sentence_Folder_To_Store+'/Sentence_UD_Score_Histogram')
            TABA_Sentence.NMF_Highest_Score_For_Any_Topic_Histogram[0].to_csv(Sentence_Folder_To_Store+'/Sentence_NMF_Highest_Score_For_Any_Topic_Histogram.csv')
            TABA_Sentence.NMF_Highest_Score_For_Any_Topic_Histogram[1].figure.savefig(Sentence_Folder_To_Store+'/Sentence_NMF_Highest_Score_For_Any_Topic_Histogram')
            TABA_Sentence.UD_Highest_Score_For_Any_Topic_Histogram[0].to_csv(Sentence_Folder_To_Store+'/Sentence_UD_Highest_Score_For_Any_Topic_Histogram.csv')
            TABA_Sentence.UD_Highest_Score_For_Any_Topic_Histogram[1].figure.savefig(Sentence_Folder_To_Store+'/Sentence_UD_Highest_Score_For_Any_Topic_Histogram')
            TABA_Sentence.NMF_Filterd_Score_Histogram[0].to_csv(Sentence_Folder_To_Store+'/Sentence_NMF_Filterd_Score_Histogram.csv')
            TABA_Sentence.NMF_Filterd_Score_Histogram[1].figure.savefig(Sentence_Folder_To_Store+'/Sentence_NMF_Filterd_Score_Histogram')
            TABA_Sentence.UD_Filtered_Score_Histogram[0].to_csv(Sentence_Folder_To_Store+'/Sentence_UD_Filtered_Score_Histogram.csv')
            TABA_Sentence.UD_Filtered_Score_Histogram[1].figure.savefig(Sentence_Folder_To_Store+'/Sentence_UD_Filtered_Score_Histogram')
            TABA_Sentence.NMF_Topic_Distribution[0].to_csv(Sentence_Folder_To_Store+'/NMF_Topic_Sentence_Distribution.csv')
            TABA_Sentence.NMF_Topic_Distribution[1].figure.savefig(Sentence_Folder_To_Store+'/NMF_Topic_Sentence_Distribution')
            TABA_Sentence.UD_Topic_Distribution[0].to_csv(Sentence_Folder_To_Store+'/UD_Topic_Sentence_Distribution.csv')
            TABA_Sentence.UD_Topic_Distribution[1].figure.savefig(Sentence_Folder_To_Store+'/UD_Topic_Sentence_Distribution')
            TABA_Sentence.NMF_Highest_Score_Topic_Distribution[0].to_csv(Sentence_Folder_To_Store+'/NMF_Highest_Score_Topic_Sentence_Distribution.csv')
            TABA_Sentence.NMF_Highest_Score_Topic_Distribution[1].figure.savefig(Sentence_Folder_To_Store+'/NMF_Highest_Score_Topic_Sentence_Distribution')
            TABA_Sentence.UD_Highest_Score_Topic_Distribution[0].to_csv(Sentence_Folder_To_Store+'/UD_Highest_Score_Topic_Sentence_Distribution.csv')
            TABA_Sentence.UD_Highest_Score_Topic_Distribution[1].figure.savefig(Sentence_Folder_To_Store+'/UD_Highest_Score_Topic_Sentence_Distribution')

        if config['Basic_Analysis'].getboolean('paragraph_max'):
            set_values=config['Basic_Analysis'].getboolean('paragraph_max_set_values')
            TABA_Paragraph_Max=TA.get_Basic_Analysis(level='Paragraph_Max',set_values=set_values)
            Paragraph_Max_Folder_To_Store=Basic_analysis_Folder+'/'+config['Basic_Analysis']['paragraph_max_folder_to_store']
            if not os.path.exists(Paragraph_Max_Folder_To_Store):
                os.makedirs(Paragraph_Max_Folder_To_Store)
            TABA_Paragraph_Max.filtered_data.to_csv(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_filtered_data.csv')
            TABA_Paragraph_Max.NMF_Score_Histogram[0].to_csv(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_NMF_Score_Histogram.csv')
            TABA_Paragraph_Max.NMF_Score_Histogram[1].figure.savefig(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_NMF_Score_Histogram')
            TABA_Paragraph_Max.UD_Score_Histogram[0].to_csv(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_UD_Score_Histogram.csv')
            TABA_Paragraph_Max.UD_Score_Histogram[1].figure.savefig(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_UD_Score_Histogram')
            TABA_Paragraph_Max.NMF_Highest_Score_For_Any_Topic_Histogram[0].to_csv(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_NMF_Highest_Score_For_Any_Topic_Histogram.csv')
            TABA_Paragraph_Max.NMF_Highest_Score_For_Any_Topic_Histogram[1].figure.savefig(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_NMF_Highest_Score_For_Any_Topic_Histogram')
            TABA_Paragraph_Max.UD_Highest_Score_For_Any_Topic_Histogram[0].to_csv(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_UD_Highest_Score_For_Any_Topic_Histogram.csv')
            TABA_Paragraph_Max.UD_Highest_Score_For_Any_Topic_Histogram[1].figure.savefig(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_UD_Highest_Score_For_Any_Topic_Histogram')
            TABA_Paragraph_Max.NMF_Filterd_Score_Histogram[0].to_csv(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_NMF_Filterd_Score_Histogram.csv')
            TABA_Paragraph_Max.NMF_Filterd_Score_Histogram[1].figure.savefig(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_NMF_Filterd_Score_Histogram')
            TABA_Paragraph_Max.UD_Filtered_Score_Histogram[0].to_csv(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_UD_Filtered_Score_Histogram.csv')
            TABA_Paragraph_Max.UD_Filtered_Score_Histogram[1].figure.savefig(Paragraph_Max_Folder_To_Store+'/Paragraph_Max_UD_Filtered_Score_Histogram')
            TABA_Paragraph_Max.NMF_Topic_Distribution[0].to_csv(Paragraph_Max_Folder_To_Store+'/NMF_Topic_Paragraph_Max_Distribution.csv')
            TABA_Paragraph_Max.NMF_Topic_Distribution[1].figure.savefig(Paragraph_Max_Folder_To_Store+'/NMF_Topic_Paragraph_Max_Distribution')
            TABA_Paragraph_Max.UD_Topic_Distribution[0].to_csv(Paragraph_Max_Folder_To_Store+'/UD_Topic_Paragraph_Max_Distribution.csv')
            TABA_Paragraph_Max.UD_Topic_Distribution[1].figure.savefig(Paragraph_Max_Folder_To_Store+'/UD_Topic_Paragraph_Max_Distribution')
            TABA_Paragraph_Max.NMF_Highest_Score_Topic_Distribution[0].to_csv(Paragraph_Max_Folder_To_Store+'/NMF_Highest_Score_Topic_Paragraph_Max_Distribution.csv')
            TABA_Paragraph_Max.NMF_Highest_Score_Topic_Distribution[1].figure.savefig(Paragraph_Max_Folder_To_Store+'/NMF_Highest_Score_Topic_Paragraph_Max_Distribution')
            TABA_Paragraph_Max.UD_Highest_Score_Topic_Distribution[0].to_csv(Paragraph_Max_Folder_To_Store+'/UD_Highest_Score_Topic_Paragraph_Max_Distribution.csv')
            TABA_Paragraph_Max.UD_Highest_Score_Topic_Distribution[1].figure.savefig(Paragraph_Max_Folder_To_Store+'/UD_Highest_Score_Topic_Paragraph_Max_Distribution')
            
        if config['Basic_Analysis'].getboolean('paragraph_mean'):
            set_values=config['Basic_Analysis'].getboolean('paragraph_mean_set_values')
            TABA_Paragraph_Mean=TA.get_Basic_Analysis(level='Paragraph_Mean',set_values=set_values)
            Paragraph_Mean_Folder_To_Store=Basic_analysis_Folder+'/'+config['Basic_Analysis']['paragraph_mean_folder_to_store']
            if not os.path.exists(Paragraph_Mean_Folder_To_Store):
                os.makedirs(Paragraph_Mean_Folder_To_Store)
            TABA_Paragraph_Mean.filtered_data.to_csv(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_filtered_data.csv')
            TABA_Paragraph_Mean.NMF_Score_Histogram[0].to_csv(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_NMF_Score_Histogram.csv')
            TABA_Paragraph_Mean.NMF_Score_Histogram[1].figure.savefig(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_NMF_Score_Histogram')
            TABA_Paragraph_Mean.UD_Score_Histogram[0].to_csv(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_UD_Score_Histogram.csv')
            TABA_Paragraph_Mean.UD_Score_Histogram[1].figure.savefig(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_UD_Score_Histogram')
            TABA_Paragraph_Mean.NMF_Highest_Score_For_Any_Topic_Histogram[0].to_csv(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_NMF_Highest_Score_For_Any_Topic_Histogram.csv')
            TABA_Paragraph_Mean.NMF_Highest_Score_For_Any_Topic_Histogram[1].figure.savefig(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_NMF_Highest_Score_For_Any_Topic_Histogram')
            TABA_Paragraph_Mean.UD_Highest_Score_For_Any_Topic_Histogram[0].to_csv(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_UD_Highest_Score_For_Any_Topic_Histogram.csv')
            TABA_Paragraph_Mean.UD_Highest_Score_For_Any_Topic_Histogram[1].figure.savefig(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_UD_Highest_Score_For_Any_Topic_Histogram')
            TABA_Paragraph_Mean.NMF_Filterd_Score_Histogram[0].to_csv(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_NMF_Filterd_Score_Histogram.csv')
            TABA_Paragraph_Mean.NMF_Filterd_Score_Histogram[1].figure.savefig(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_NMF_Filterd_Score_Histogram')
            TABA_Paragraph_Mean.UD_Filtered_Score_Histogram[0].to_csv(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_UD_Filtered_Score_Histogram.csv')
            TABA_Paragraph_Mean.UD_Filtered_Score_Histogram[1].figure.savefig(Paragraph_Mean_Folder_To_Store+'/Paragraph_Mean_UD_Filtered_Score_Histogram')
            TABA_Paragraph_Mean.NMF_Topic_Distribution[0].to_csv(Paragraph_Mean_Folder_To_Store+'/NMF_Topic_Paragraph_Mean_Distribution.csv')
            TABA_Paragraph_Mean.NMF_Topic_Distribution[1].figure.savefig(Paragraph_Mean_Folder_To_Store+'/NMF_Topic_Paragraph_Mean_Distribution')
            TABA_Paragraph_Mean.UD_Topic_Distribution[0].to_csv(Paragraph_Mean_Folder_To_Store+'/UD_Topic_Paragraph_Mean_Distribution.csv')
            TABA_Paragraph_Mean.UD_Topic_Distribution[1].figure.savefig(Paragraph_Mean_Folder_To_Store+'/UD_Topic_Paragraph_Mean_Distribution')
            TABA_Paragraph_Mean.NMF_Highest_Score_Topic_Distribution[0].to_csv(Paragraph_Mean_Folder_To_Store+'/NMF_Highest_Score_Topic_Paragraph_Mean_Distribution.csv')
            TABA_Paragraph_Mean.NMF_Highest_Score_Topic_Distribution[1].figure.savefig(Paragraph_Mean_Folder_To_Store+'/NMF_Highest_Score_Topic_Paragraph_Mean_Distribution')
            TABA_Paragraph_Mean.UD_Highest_Score_Topic_Distribution[0].to_csv(Paragraph_Mean_Folder_To_Store+'/UD_Highest_Score_Topic_Paragraph_Mean_Distribution.csv')
            TABA_Paragraph_Mean.UD_Highest_Score_Topic_Distribution[1].figure.savefig(Paragraph_Mean_Folder_To_Store+'/UD_Highest_Score_Topic_Paragraph_Mean_Distribution')
    
    if config['Similarity_Analysis_DataFrame'].getboolean('sad'):
        Similarity_Analysis_Dataframe_Folder=Results_Folder+'/'+config['Similarity_Analysis_DataFrame']['similarity_analysis_dataframe_folder']
        if not os.path.exists(Similarity_Analysis_Dataframe_Folder):
                os.makedirs(Similarity_Analysis_Dataframe_Folder)
        if config['Similarity_Analysis_DataFrame'].getboolean('sentence_wj'):
            set_values=config['Similarity_Analysis_DataFrame'].getboolean('sentence_wj_set_values')
            Sentence_wj_Folder_To_Store=Similarity_Analysis_Dataframe_Folder+'/'+config['Similarity_Analysis_DataFrame']['sentence_wj_folder_to_store']
            if not os.path.exists(Sentence_wj_Folder_To_Store):
                os.makedirs(Sentence_wj_Folder_To_Store)
            TASAWJ_Sentence=TA.get_Similarity_Analysis_DataFrame(level='Sentence',Similarity_Method='Weighted_Jaccard', set_values=set_values)
            TASAWJ_Sentence.NMF_Similarity_DataFrame.to_csv(Sentence_wj_Folder_To_Store+'/Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame.csv')
            TASAWJ_Sentence.UD_Similarity_DataFrame.to_csv(Sentence_wj_Folder_To_Store+'/Sentence_UD_Weighted_Jaccard_Similarity_DataFrame.csv')
            TASAWJ_Sentence.NMF_UD_Similarity_DataFrame.to_csv(Sentence_wj_Folder_To_Store+'/Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame.csv')
        if config['Similarity_Analysis_DataFrame'].getboolean('sentence_sj'):
            set_values=config['Similarity_Analysis_DataFrame'].getboolean('sentence_sj_set_values')
            Sentence_sj_Folder_To_Store=Similarity_Analysis_Dataframe_Folder+'/'+config['Similarity_Analysis_DataFrame']['sentence_sj_folder_to_store']
            if not os.path.exists(Sentence_sj_Folder_To_Store):
                os.makedirs(Sentence_sj_Folder_To_Store)
            TASASJ_Sentence=TA.get_Similarity_Analysis_DataFrame(level='Sentence',Similarity_Method='Set_Jaccard', set_values=set_values)
            TASASJ_Sentence.NMF_Similarity_DataFrame.to_csv(Sentence_sj_Folder_To_Store+'/Sentence_NMF_Set_Jaccard_Similarity_DataFrame.csv')
            TASASJ_Sentence.UD_Similarity_DataFrame.to_csv(Sentence_sj_Folder_To_Store+'/Sentence_UD_Set_Jaccard_Similarity_DataFrame.csv')
            TASASJ_Sentence.NMF_UD_Similarity_DataFrame.to_csv(Sentence_sj_Folder_To_Store+'/Sentence_NMF_UD_Set_Jaccard_Similarity_DataFrame.csv')
        if config['Similarity_Analysis_DataFrame'].getboolean('paragraph_max_wj'):
            set_values=config['Similarity_Analysis_DataFrame'].getboolean('paragraph_max_wj_set_values')
            Paragraph_Max_wj_Folder_To_Store=Similarity_Analysis_Dataframe_Folder+'/'+config['Similarity_Analysis_DataFrame']['paragraph_max_wj_folder_to_store']
            if not os.path.exists(Paragraph_Max_wj_Folder_To_Store):
                os.makedirs(Paragraph_Max_wj_Folder_To_Store)
            TASAWJ_Paragraph_Max=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Max',Similarity_Method='Weighted_Jaccard', set_values=set_values)
            TASAWJ_Paragraph_Max.NMF_Similarity_DataFrame.to_csv(Paragraph_Max_wj_Folder_To_Store+'/Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame.csv')
            TASAWJ_Paragraph_Max.UD_Similarity_DataFrame.to_csv(Paragraph_Max_wj_Folder_To_Store+'/Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame.csv')
            TASAWJ_Paragraph_Max.NMF_UD_Similarity_DataFrame.to_csv(Paragraph_Max_wj_Folder_To_Store+'/Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_DataFrame.csv')
        if config['Similarity_Analysis_DataFrame'].getboolean('paragraph_max_sj'):
            set_values=config['Similarity_Analysis_DataFrame'].getboolean('paragraph_max_sj_set_values')
            Paragraph_Max_sj_Folder_To_Store=Similarity_Analysis_Dataframe_Folder+'/'+config['Similarity_Analysis_DataFrame']['paragraph_max_sj_folder_to_store']
            if not os.path.exists(Paragraph_Max_sj_Folder_To_Store):
                os.makedirs(Paragraph_Max_sj_Folder_To_Store)
            TASASJ_Paragraph_Max=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Max',Similarity_Method='Set_Jaccard', set_values=set_values)
            TASASJ_Paragraph_Max.NMF_Similarity_DataFrame.to_csv(Paragraph_Max_sj_Folder_To_Store+'/Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame.csv')
            TASASJ_Paragraph_Max.UD_Similarity_DataFrame.to_csv(Paragraph_Max_sj_Folder_To_Store+'/Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame.csv')
            TASASJ_Paragraph_Max.NMF_UD_Similarity_DataFrame.to_csv(Paragraph_Max_sj_Folder_To_Store+'/Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_DataFrame.csv')
        if config['Similarity_Analysis_DataFrame'].getboolean('paragraph_mean_wj'):
            set_values=config['Similarity_Analysis_DataFrame'].getboolean('paragraph_mean_wj_set_values')
            Paragraph_Mean_wj_Folder_To_Store=Similarity_Analysis_Dataframe_Folder+'/'+config['Similarity_Analysis_DataFrame']['paragraph_mean_wj_folder_to_store']
            if not os.path.exists(Paragraph_Mean_wj_Folder_To_Store):
                os.makedirs(Paragraph_Mean_wj_Folder_To_Store)
            TASAWJ_Paragraph_Mean=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Mean',Similarity_Method='Weighted_Jaccard', set_values=set_values)
            TASAWJ_Paragraph_Mean.NMF_Similarity_DataFrame.to_csv(Paragraph_Mean_wj_Folder_To_Store+'/Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame.csv')
            TASAWJ_Paragraph_Mean.UD_Similarity_DataFrame.to_csv(Paragraph_Mean_wj_Folder_To_Store+'/Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame.csv')
            TASAWJ_Paragraph_Mean.NMF_UD_Similarity_DataFrame.to_csv(Paragraph_Mean_wj_Folder_To_Store+'/Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame.csv')
        if config['Similarity_Analysis_DataFrame'].getboolean('paragraph_mean_sj'):
            set_values=config['Similarity_Analysis_DataFrame'].getboolean('paragraph_mean_sj_set_values')
            Paragraph_Mean_sj_Folder_To_Store=Similarity_Analysis_Dataframe_Folder+'/'+config['Similarity_Analysis_DataFrame']['paragraph_mean_sj_folder_to_store']
            if not os.path.exists(Paragraph_Mean_sj_Folder_To_Store):
                os.makedirs(Paragraph_Mean_sj_Folder_To_Store)
            TASASJ_Paragraph_Mean=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Mean',Similarity_Method='Set_Jaccard', set_values=set_values)
            TASASJ_Paragraph_Mean.NMF_Similarity_DataFrame.to_csv(Paragraph_Mean_sj_Folder_To_Store+'/Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame.csv')
            TASASJ_Paragraph_Mean.UD_Similarity_DataFrame.to_csv(Paragraph_Mean_sj_Folder_To_Store+'/Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame.csv')
            TASASJ_Paragraph_Mean.NMF_UD_Similarity_DataFrame.to_csv(Paragraph_Mean_sj_Folder_To_Store+'/Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_DataFrame.csv')
    if config['Similarity_Analysis_HeatMap'].getboolean('sah'):
        Similarity_Analysis_HeatMap_Folder=Results_Folder+'/'+config['Similarity_Analysis_HeatMap']['similarity_analysis_heatmap_folder']
        if not os.path.exists(Similarity_Analysis_HeatMap_Folder):
                os.makedirs(Similarity_Analysis_HeatMap_Folder)
        if config['Similarity_Analysis_HeatMap'].getboolean('sentence_wj'):
            if  TA.Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame is None:
                TA.Sentence_NMF_Weighted_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Weighted_Jaccard', set_values=True)
            if TA.Sentence_UD_Weighted_Jaccard_Similarity_DataFrame is None:
                TA.Sentence_UD_Weighted_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Weighted_Jaccard', set_values=True)
            if TA.Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame is None:
                TA.Sentence_NMF_UD_Weighted_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Weighted_Jaccard', set_values=True)
            set_values=config['Similarity_Analysis_HeatMap'].getboolean('sentence_wj_set_values')
            Sentence_wj_Folder_To_Store=Similarity_Analysis_HeatMap_Folder+'/'+config['Similarity_Analysis_HeatMap']['sentence_wj_folder_to_store']
            if not os.path.exists(Sentence_wj_Folder_To_Store):
                os.makedirs(Sentence_wj_Folder_To_Store)
            TASAWJ_Sentence=TA.get_Similarity_Analysis_HeatMap(level='Sentence',Similarity_Method='Weighted_Jaccard', set_values=set_values)
            TASAWJ_Sentence.NMF_Similarity_HeatMap.figure.savefig(Sentence_wj_Folder_To_Store+'/Sentence_NMF_Weighted_Jaccard_Similarity_HeatMap')
            TASAWJ_Sentence.UD_Similarity_HeatMap.figure.savefig(Sentence_wj_Folder_To_Store+'/Sentence_UD_Weighted_Jaccard_Similarity_HeatMap')
            TASAWJ_Sentence.NMF_UD_Similarity_HeatMap.figure.savefig(Sentence_wj_Folder_To_Store+'/Sentence_NMF_UD_Weighted_Jaccard_Similarity_HeatMap')
        if config['Similarity_Analysis_HeatMap'].getboolean('sentence_sj'):
            set_values=config['Similarity_Analysis_HeatMap'].getboolean('sentence_sj_set_values')
            Sentence_sj_Folder_To_Store=Similarity_Analysis_HeatMap_Folder+'/'+config['Similarity_Analysis_HeatMap']['sentence_sj_folder_to_store']
            if not os.path.exists(Sentence_sj_Folder_To_Store):
                os.makedirs(Sentence_sj_Folder_To_Store)
            if TA.Sentence_NMF_Set_Jaccard_Similarity_DataFrame is None:
                TA.Sentence_NMF_Set_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Set_Jaccard', set_values=True)
            if TA.Sentence_UD_Set_Jaccard_Similarity_DataFrame is None:
                TA.Sentence_UD_Set_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Set_Jaccard', set_values=True)
            if TA.Sentence_NMF_UD_Set_Jaccard_Similarity_DataFrame is None:
                TA.Sentence_NMF_UD_Set_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Set_Jaccard', set_values=True)
            TASASJ_Sentence=TA.get_Similarity_Analysis_HeatMap(level='Sentence',Similarity_Method='Set_Jaccard', set_values=set_values)
            TASASJ_Sentence.NMF_Similarity_HeatMap.figure.savefig(Sentence_sj_Folder_To_Store+'/Sentence_NMF_Set_Jaccard_Similarity_HeatMap')
            TASASJ_Sentence.UD_Similarity_HeatMap.figure.savefig(Sentence_sj_Folder_To_Store+'/Sentence_UD_Set_Jaccard_Similarity_HeatMap')
            TASASJ_Sentence.NMF_UD_Similarity_HeatMap.figure.savefig(Sentence_sj_Folder_To_Store+'/Sentence_NMF_UD_Set_Jaccard_Similarity_HeatMap')
        if config['Similarity_Analysis_HeatMap'].getboolean('paragraph_max_wj'):
            set_values=config['Similarity_Analysis_HeatMap'].getboolean('paragraph_max_wj_set_values')
            Paragraph_Max_wj_Folder_To_Store=Similarity_Analysis_HeatMap_Folder+'/'+config['Similarity_Analysis_HeatMap']['paragraph_max_wj_folder_to_store']
            if not os.path.exists(Paragraph_Max_wj_Folder_To_Store):
                os.makedirs(Paragraph_Max_wj_Folder_To_Store)
            if TA.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame is None:
                TA.Paragraph_Max_NMF_Weighted_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Weighted_Jaccard', set_values=True)
            if TA.Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame is None:
                TA.Paragraph_Max_UD_Weighted_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Weighted_Jaccard', set_values=True)
            if TA.Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_DataFrame is None:
                TA.Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Weighted_Jaccard', set_values=True)
            TASAWJ_Paragraph_Max=TA.get_Similarity_Analysis_HeatMap(level='Paragraph_Max',Similarity_Method='Weighted_Jaccard', set_values=set_values)
            TASAWJ_Paragraph_Max.NMF_Similarity_HeatMap.figure.savefig(Paragraph_Max_wj_Folder_To_Store+'/Paragraph_Max_NMF_Weighted_Jaccard_Similarity_HeatMap')
            TASAWJ_Paragraph_Max.UD_Similarity_HeatMap.figure.savefig(Paragraph_Max_wj_Folder_To_Store+'/Paragraph_Max_UD_Weighted_Jaccard_Similarity_HeatMap')
            TASAWJ_Paragraph_Max.NMF_UD_Similarity_HeatMap.figure.savefig(Paragraph_Max_wj_Folder_To_Store+'/Paragraph_Max_NMF_UD_Weighted_Jaccard_Similarity_HeatMap')
        if config['Similarity_Analysis_HeatMap'].getboolean('paragraph_max_sj'):
            set_values=config['Similarity_Analysis_HeatMap'].getboolean('paragraph_max_sj_set_values')
            Paragraph_Max_sj_Folder_To_Store=Similarity_Analysis_HeatMap_Folder+'/'+config['Similarity_Analysis_HeatMap']['paragraph_max_sj_folder_to_store']
            if not os.path.exists(Paragraph_Max_sj_Folder_To_Store):
                os.makedirs(Paragraph_Max_sj_Folder_To_Store)
            if TA.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame is None:
                TA.Paragraph_Max_NMF_Set_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Set_Jaccard', set_values=True)
            if TA.Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame is None:
                TA.Paragraph_Max_UD_Set_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Set_Jaccard', set_values=True)
            if TA.Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_DataFrame is None:
                TA.Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Set_Jaccard', set_values=True)
            TASASJ_Paragraph_Max=TA.get_Similarity_Analysis_HeatMap(level='Paragraph_Max',Similarity_Method='Set_Jaccard', set_values=set_values)
            TASASJ_Paragraph_Max.NMF_Similarity_HeatMap.figure.savefig(Paragraph_Max_sj_Folder_To_Store+'/Paragraph_Max_NMF_Set_Jaccard_Similarity_HeatMap')
            TASASJ_Paragraph_Max.UD_Similarity_HeatMap.figure.savefig(Paragraph_Max_sj_Folder_To_Store+'/Paragraph_Max_UD_Set_Jaccard_Similarity_HeatMap')
            TASASJ_Paragraph_Max.NMF_UD_Similarity_HeatMap.figure.savefig(Paragraph_Max_sj_Folder_To_Store+'/Paragraph_Max_NMF_UD_Set_Jaccard_Similarity_HeatMap')
        if config['Similarity_Analysis_HeatMap'].getboolean('paragraph_mean_wj'):
            set_values=config['Similarity_Analysis_HeatMap'].getboolean('paragraph_mean_wj_set_values')
            Paragraph_Mean_wj_Folder_To_Store=Similarity_Analysis_HeatMap_Folder+'/'+config['Similarity_Analysis_HeatMap']['paragraph_mean_wj_folder_to_store']
            if not os.path.exists(Paragraph_Mean_wj_Folder_To_Store):
                os.makedirs(Paragraph_Mean_wj_Folder_To_Store)
            if TA.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame is None:
                TA.Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Weighted_Jaccard', set_values=True)
            if TA.Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame is None:
                TA.Paragraph_Mean_UD_Weighted_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Weighted_Jaccard', set_values=True)
            if TA.Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame is None:
                TA.Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Weighted_Jaccard', set_values=True)
            TASAWJ_Paragraph_Mean=TA.get_Similarity_Analysis_HeatMap(level='Paragraph_Mean',Similarity_Method='Weighted_Jaccard', set_values=set_values)
            TASAWJ_Paragraph_Mean.NMF_Similarity_HeatMap.figure.savefig(Paragraph_Mean_wj_Folder_To_Store+'/Paragraph_Mean_NMF_Weighted_Jaccard_Similarity_HeatMap')
            TASAWJ_Paragraph_Mean.UD_Similarity_HeatMap.figure.savefig(Paragraph_Mean_wj_Folder_To_Store+'/Paragraph_Mean_UD_Weighted_Jaccard_Similarity_HeatMap')
            TASAWJ_Paragraph_Mean.NMF_UD_Similarity_HeatMap.figure.savefig(Paragraph_Mean_wj_Folder_To_Store+'/Paragraph_Mean_NMF_UD_Weighted_Jaccard_Similarity_HeatMap')
        if config['Similarity_Analysis_HeatMap'].getboolean('paragraph_mean_sj'):
            set_values=config['Similarity_Analysis_HeatMap'].getboolean('paragraph_mean_sj_set_values')
            Paragraph_Mean_sj_Folder_To_Store=Similarity_Analysis_HeatMap_Folder+'/'+config['Similarity_Analysis_HeatMap']['paragraph_mean_sj_folder_to_store']
            if not os.path.exists(Paragraph_Mean_sj_Folder_To_Store):
                os.makedirs(Paragraph_Mean_sj_Folder_To_Store)
            if TA.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame is None:
                TA.Paragraph_Mean_NMF_Set_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Set_Jaccard', set_values=True)
            if TA.Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame is None:
                TA.Paragraph_Mean_UD_Set_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Set_Jaccard', set_values=True)
            if TA.Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_DataFrame is NoSectionError:
                TA.Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_DataFrame=TA.get_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Set_Jaccard', set_values=True)
            TASASJ_Paragraph_Mean=TA.get_Similarity_Analysis_HeatMap(level='Paragraph_Mean',Similarity_Method='Set_Jaccard', set_values=set_values)
            TASASJ_Paragraph_Mean.NMF_Similarity_HeatMap.figure.savefig(Paragraph_Mean_sj_Folder_To_Store+'/Paragraph_Mean_NMF_Set_Jaccard_Similarity_HeatMap')
            TASASJ_Paragraph_Mean.UD_Similarity_HeatMap.figure.savefig(Paragraph_Mean_sj_Folder_To_Store+'/Paragraph_Mean_UD_Set_Jaccard_Similarity_HeatMap')
            TASASJ_Paragraph_Mean.NMF_UD_Similarity_HeatMap.figure.savefig(Paragraph_Mean_sj_Folder_To_Store+'/Paragraph_Mean_NMF_UD_Set_Jaccard_Similarity_HeatMap')
          
    if config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('sad'):
        Embeddings_Similarity_Analysis_Dataframe_Folder=Results_Folder+'/'+config['Embeddings_Similarity_Analysis_DataFrame']['Embeddings_Similarity_Analysis_DataFrame_folder']
        if not os.path.exists(Embeddings_Similarity_Analysis_Dataframe_Folder):
                print("creating folder")
                os.makedirs(Embeddings_Similarity_Analysis_Dataframe_Folder)
        if config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('sentence_we'):
            set_values=config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('sentence_we_set_values')
            Sentence_we_Folder_To_Store=Embeddings_Similarity_Analysis_Dataframe_Folder+'/'+config['Embeddings_Similarity_Analysis_DataFrame']['sentence_we_folder_to_store']
            if not os.path.exists(Sentence_we_Folder_To_Store):
                os.makedirs(Sentence_we_Folder_To_Store)
            TASAWE_Sentence=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Sentence',Similarity_Method='Weighted_Embeddings', set_values=set_values)
            TASAWE_Sentence.NMF_Embeddings_Similarity_DataFrame.to_csv(Sentence_we_Folder_To_Store+'/Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame.csv')
            TASAWE_Sentence.UD_Embeddings_Similarity_DataFrame.to_csv(Sentence_we_Folder_To_Store+'/Sentence_UD_Weighted_Embeddings_Similarity_DataFrame.csv')
            TASAWE_Sentence.NMF_UD_Embeddings_Similarity_DataFrame.to_csv(Sentence_we_Folder_To_Store+'/Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame.csv')
        if config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('sentence_se'):
            set_values=config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('sentence_se_set_values')
            Sentence_se_Folder_To_Store=Embeddings_Similarity_Analysis_Dataframe_Folder+'/'+config['Embeddings_Similarity_Analysis_DataFrame']['sentence_se_folder_to_store']
            if not os.path.exists(Sentence_se_Folder_To_Store):
                os.makedirs(Sentence_se_Folder_To_Store)
            TASASE_Sentence=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Sentence',Similarity_Method='Set_Embeddings', set_values=set_values)
            TASASE_Sentence.NMF_Embeddings_Similarity_DataFrame.to_csv(Sentence_se_Folder_To_Store+'/Sentence_NMF_Set_Embeddings_Similarity_DataFrame.csv')
            TASASE_Sentence.UD_Embeddings_Similarity_DataFrame.to_csv(Sentence_se_Folder_To_Store+'/Sentence_UD_Set_Embeddings_Similarity_DataFrame.csv')
            TASASE_Sentence.NMF_UD_Embeddings_Similarity_DataFrame.to_csv(Sentence_se_Folder_To_Store+'/Sentence_NMF_UD_Set_Embeddings_Similarity_DataFrame.csv')
        if config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('paragraph_max_we'):
            set_values=config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('paragraph_max_we_set_values')
            Paragraph_Max_we_Folder_To_Store=Embeddings_Similarity_Analysis_Dataframe_Folder+'/'+config['Embeddings_Similarity_Analysis_DataFrame']['paragraph_max_we_folder_to_store']
            if not os.path.exists(Paragraph_Max_we_Folder_To_Store):
                os.makedirs(Paragraph_Max_we_Folder_To_Store)
            TASAWE_Paragraph_Max=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Max',Similarity_Method='Weighted_Embeddings', set_values=set_values)
            TASAWE_Paragraph_Max.NMF_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Max_we_Folder_To_Store+'/Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame.csv')
            TASAWE_Paragraph_Max.UD_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Max_we_Folder_To_Store+'/Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame.csv')
            TASAWE_Paragraph_Max.NMF_UD_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Max_we_Folder_To_Store+'/Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_DataFrame.csv')
        if config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('paragraph_max_se'):
            set_values=config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('paragraph_max_se_set_values')
            Paragraph_Max_se_Folder_To_Store=Embeddings_Similarity_Analysis_Dataframe_Folder+'/'+config['Embeddings_Similarity_Analysis_DataFrame']['paragraph_max_se_folder_to_store']
            if not os.path.exists(Paragraph_Max_se_Folder_To_Store):
                os.makedirs(Paragraph_Max_se_Folder_To_Store)
            TASASE_Paragraph_Max=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Max',Similarity_Method='Set_Embeddings', set_values=set_values)
            TASASE_Paragraph_Max.NMF_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Max_se_Folder_To_Store+'/Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame.csv')
            TASASE_Paragraph_Max.UD_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Max_se_Folder_To_Store+'/Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame.csv')
            TASASE_Paragraph_Max.NMF_UD_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Max_se_Folder_To_Store+'/Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_DataFrame.csv')
        if config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('paragraph_mean_we'):
            set_values=config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('paragraph_mean_we_set_values')
            Paragraph_Mean_we_Folder_To_Store=Embeddings_Similarity_Analysis_Dataframe_Folder+'/'+config['Embeddings_Similarity_Analysis_DataFrame']['paragraph_mean_we_folder_to_store']
            if not os.path.exists(Paragraph_Mean_we_Folder_To_Store):
                os.makedirs(Paragraph_Mean_we_Folder_To_Store)
            TASAWE_Paragraph_Mean=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Mean',Similarity_Method='Weighted_Embeddings', set_values=set_values)
            TASAWE_Paragraph_Mean.NMF_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Mean_we_Folder_To_Store+'/Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame.csv')
            TASAWE_Paragraph_Mean.UD_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Mean_we_Folder_To_Store+'/Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame.csv')
            TASAWE_Paragraph_Mean.NMF_UD_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Mean_we_Folder_To_Store+'/Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame.csv')
        if config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('paragraph_mean_se'):
            set_values=config['Embeddings_Similarity_Analysis_DataFrame'].getboolean('paragraph_mean_se_set_values')
            Paragraph_Mean_se_Folder_To_Store=Embeddings_Similarity_Analysis_Dataframe_Folder+'/'+config['Embeddings_Similarity_Analysis_DataFrame']['paragraph_mean_se_folder_to_store']
            if not os.path.exists(Paragraph_Mean_se_Folder_To_Store):
                os.makedirs(Paragraph_Mean_se_Folder_To_Store)
            TASASE_Paragraph_Mean=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Mean',Similarity_Method='Set_Embeddings', set_values=set_values)
            TASASE_Paragraph_Mean.NMF_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Mean_se_Folder_To_Store+'/Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame.csv')
            TASASE_Paragraph_Mean.UD_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Mean_se_Folder_To_Store+'/Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame.csv')
            TASASE_Paragraph_Mean.NMF_UD_Embeddings_Similarity_DataFrame.to_csv(Paragraph_Mean_se_Folder_To_Store+'/Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_DataFrame.csv') 
    
    if config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('sah'):
        Embeddings_Similarity_Analysis_HeatMap_Folder=Results_Folder+'/'+config['Embeddings_Similarity_Analysis_HeatMap']['Embeddings_Similarity_Analysis_HeatMap_folder']
        if not os.path.exists(Embeddings_Similarity_Analysis_HeatMap_Folder):
                os.makedirs(Embeddings_Similarity_Analysis_HeatMap_Folder)
        if config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('sentence_we'):
            if  TA.Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame is None:
                TA.Sentence_NMF_Weighted_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Weighted_Embeddings', set_values=True)
            if TA.Sentence_UD_Weighted_Embeddings_Similarity_DataFrame is None:
                TA.Sentence_UD_Weighted_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Weighted_Embeddings', set_values=True)
            if TA.Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame is None:
                TA.Sentence_NMF_UD_Weighted_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Weighted_Embeddings', set_values=True)
            set_values=config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('sentence_we_set_values')
            Sentence_we_Folder_To_Store=Embeddings_Similarity_Analysis_HeatMap_Folder+'/'+config['Embeddings_Similarity_Analysis_HeatMap']['sentence_we_folder_to_store']
            if not os.path.exists(Sentence_we_Folder_To_Store):
                os.makedirs(Sentence_we_Folder_To_Store)
            TASAWE_Sentence=TA.get_Embeddings_Similarity_Analysis_HeatMap(level='Sentence',Similarity_Method='Weighted_Embeddings', set_values=set_values)
            TASAWE_Sentence.NMF_Embeddings_Similarity_HeatMap.figure.savefig(Sentence_we_Folder_To_Store+'/Sentence_NMF_Weighted_Embeddings_Similarity_HeatMap')
            TASAWE_Sentence.UD_Embeddings_Similarity_HeatMap.figure.savefig(Sentence_we_Folder_To_Store+'/Sentence_UD_Weighted_Embeddings_Similarity_HeatMap')
            TASAWE_Sentence.NMF_UD_Embeddings_Similarity_HeatMap.figure.savefig(Sentence_we_Folder_To_Store+'/Sentence_NMF_UD_Weighted_Embeddings_Similarity_HeatMap')
        if config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('sentence_se'):
            set_values=config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('sentence_se_set_values')
            Sentence_se_Folder_To_Store=Embeddings_Similarity_Analysis_HeatMap_Folder+'/'+config['Embeddings_Similarity_Analysis_HeatMap']['sentence_se_folder_to_store']
            if not os.path.exists(Sentence_se_Folder_To_Store):
                os.makedirs(Sentence_se_Folder_To_Store)
            if TA.Sentence_NMF_Set_Embeddings_Similarity_DataFrame is None:
                TA.Sentence_NMF_Set_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Set_Embeddings', set_values=True)
            if TA.Sentence_UD_Set_Embeddings_Similarity_DataFrame is None:
                TA.Sentence_UD_Set_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Set_Embeddings', set_values=True)
            if TA.Sentence_NMF_UD_Set_Embeddings_Similarity_DataFrame is None:
                TA.Sentence_NMF_UD_Set_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Sentence', Similarity_Method='Set_Embeddings', set_values=True)
            TASASE_Sentence=TA.get_Embeddings_Similarity_Analysis_HeatMap(level='Sentence',Similarity_Method='Set_Embeddings', set_values=set_values)
            TASASE_Sentence.NMF_Embeddings_Similarity_HeatMap.figure.savefig(Sentence_se_Folder_To_Store+'/Sentence_NMF_Set_Embeddings_Similarity_HeatMap')
            TASASE_Sentence.UD_Embeddings_Similarity_HeatMap.figure.savefig(Sentence_se_Folder_To_Store+'/Sentence_UD_Set_Embeddings_Similarity_HeatMap')
            TASASE_Sentence.NMF_UD_Embeddings_Similarity_HeatMap.figure.savefig(Sentence_se_Folder_To_Store+'/Sentence_NMF_UD_Set_Embeddings_Similarity_HeatMap')
        if config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('paragraph_max_we'):
            set_values=config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('paragraph_max_we_set_values')
            Paragraph_Max_we_Folder_To_Store=Embeddings_Similarity_Analysis_HeatMap_Folder+'/'+config['Embeddings_Similarity_Analysis_HeatMap']['paragraph_max_we_folder_to_store']
            if not os.path.exists(Paragraph_Max_we_Folder_To_Store):
                os.makedirs(Paragraph_Max_we_Folder_To_Store)
            if TA.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame is None:
                TA.Paragraph_Max_NMF_Weighted_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Weighted_Embeddings', set_values=True)
            if TA.Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame is None:
                TA.Paragraph_Max_UD_Weighted_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Weighted_Embeddings', set_values=True)
            if TA.Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_DataFrame is None:
                TA.Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Weighted_Embeddings', set_values=True)
            TASAWE_Paragraph_Max=TA.get_Embeddings_Similarity_Analysis_HeatMap(level='Paragraph_Max',Similarity_Method='Weighted_Embeddings', set_values=set_values)
            TASAWE_Paragraph_Max.NMF_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Max_we_Folder_To_Store+'/Paragraph_Max_NMF_Weighted_Embeddings_Similarity_HeatMap')
            TASAWE_Paragraph_Max.UD_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Max_we_Folder_To_Store+'/Paragraph_Max_UD_Weighted_Embeddings_Similarity_HeatMap')
            TASAWE_Paragraph_Max.NMF_UD_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Max_we_Folder_To_Store+'/Paragraph_Max_NMF_UD_Weighted_Embeddings_Similarity_HeatMap')
        if config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('paragraph_max_se'):
            set_values=config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('paragraph_max_se_set_values')
            Paragraph_Max_se_Folder_To_Store=Embeddings_Similarity_Analysis_HeatMap_Folder+'/'+config['Embeddings_Similarity_Analysis_HeatMap']['paragraph_max_se_folder_to_store']
            if not os.path.exists(Paragraph_Max_se_Folder_To_Store):
                os.makedirs(Paragraph_Max_se_Folder_To_Store)
            if TA.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame is None:
                TA.Paragraph_Max_NMF_Set_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Set_Embeddings', set_values=True)
            if TA.Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame is None:
                TA.Paragraph_Max_UD_Set_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Set_Embeddings', set_values=True)
            if TA.Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_DataFrame is None:
                TA.Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Max', Similarity_Method='Set_Embeddings', set_values=True)
            TASASE_Paragraph_Max=TA.get_Embeddings_Similarity_Analysis_HeatMap(level='Paragraph_Max',Similarity_Method='Set_Embeddings', set_values=set_values)
            TASASE_Paragraph_Max.NMF_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Max_se_Folder_To_Store+'/Paragraph_Max_NMF_Set_Embeddings_Similarity_HeatMap')
            TASASE_Paragraph_Max.UD_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Max_se_Folder_To_Store+'/Paragraph_Max_UD_Set_Embeddings_Similarity_HeatMap')
            TASASE_Paragraph_Max.NMF_UD_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Max_se_Folder_To_Store+'/Paragraph_Max_NMF_UD_Set_Embeddings_Similarity_HeatMap')
        if config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('paragraph_mean_we'):
            set_values=config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('paragraph_mean_we_set_values')
            Paragraph_Mean_we_Folder_To_Store=Embeddings_Similarity_Analysis_HeatMap_Folder+'/'+config['Embeddings_Similarity_Analysis_HeatMap']['paragraph_mean_we_folder_to_store']
            if not os.path.exists(Paragraph_Mean_we_Folder_To_Store):
                os.makedirs(Paragraph_Mean_we_Folder_To_Store)
            if TA.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame is None:
                TA.Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Weighted_Embeddings', set_values=True)
            if TA.Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame is None:
                TA.Paragraph_Mean_UD_Weighted_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Weighted_Embeddings', set_values=True)
            if TA.Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame is None:
                TA.Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Weighted_Embeddings', set_values=True)
            TASAWE_Paragraph_Mean=TA.get_Embeddings_Similarity_Analysis_HeatMap(level='Paragraph_Mean',Similarity_Method='Weighted_Embeddings', set_values=set_values)
            TASAWE_Paragraph_Mean.NMF_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Mean_we_Folder_To_Store+'/Paragraph_Mean_NMF_Weighted_Embeddings_Similarity_HeatMap')
            TASAWE_Paragraph_Mean.UD_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Mean_we_Folder_To_Store+'/Paragraph_Mean_UD_Weighted_Embeddings_Similarity_HeatMap')
            TASAWE_Paragraph_Mean.NMF_UD_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Mean_we_Folder_To_Store+'/Paragraph_Mean_NMF_UD_Weighted_Embeddings_Similarity_HeatMap')
        if config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('paragraph_mean_se'):
            set_values=config['Embeddings_Similarity_Analysis_HeatMap'].getboolean('paragraph_mean_se_set_values')
            Paragraph_Mean_se_Folder_To_Store=Embeddings_Similarity_Analysis_HeatMap_Folder+'/'+config['Embeddings_Similarity_Analysis_HeatMap']['paragraph_mean_se_folder_to_store']
            if not os.path.exists(Paragraph_Mean_se_Folder_To_Store):
                os.makedirs(Paragraph_Mean_se_Folder_To_Store)
            if TA.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame is None:
                TA.Paragraph_Mean_NMF_Set_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Set_Embeddings', set_values=True)
            if TA.Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame is None:
                TA.Paragraph_Mean_UD_Set_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Set_Embeddings', set_values=True)
            if TA.Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_DataFrame is NoSectionError:
                TA.Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_DataFrame=TA.get_Embeddings_Similarity_Analysis_DataFrame(level='Paragraph_Mean', Similarity_Method='Set_Embeddings', set_values=True)
            TASASE_Paragraph_Mean=TA.get_Embeddings_Similarity_Analysis_HeatMap(level='Paragraph_Mean',Similarity_Method='Set_Embeddings', set_values=set_values)
            TASASE_Paragraph_Mean.NMF_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Mean_se_Folder_To_Store+'/Paragraph_Mean_NMF_Set_Embeddings_Similarity_HeatMap')
            TASASE_Paragraph_Mean.UD_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Mean_se_Folder_To_Store+'/Paragraph_Mean_UD_Set_Embeddings_Similarity_HeatMap')
            TASASE_Paragraph_Mean.NMF_UD_Embeddings_Similarity_HeatMap.figure.savefig(Paragraph_Mean_se_Folder_To_Store+'/Paragraph_Mean_NMF_UD_Set_Embeddings_Similarity_HeatMap')
             

    
start_time = time.time()
if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))

