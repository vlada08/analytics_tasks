
# coding: utf-8

# # Analysis of the Offshore leaks
# 
# ## 1. Data structure 

# Four DB were downloaded from the source. Now I will check data structure and make a brief overview about all datasets. THis is necessary to understand how this data can be used and what kind of information we can get from these datasets.
# 
# Downloaded databases:
# <ol>
#   <li>Bahama leaks.</li>
#   <li>Offshore leaks.</li>
#   <li>Panama papers.</li>
#   <li>Paradise Papers.</li>
# </ol>
# Each of them contains 5 csv files with information about connection between entities and other information about nodes which could can be helpful to find patterns in datasets. As I wasn't worked previously with such kind data materials I want to look deeper into the one of them to undesrand data structure better.
# 
# My plan for this analysis:
# <ol>
#   <li>Investigate Bahama leaks data. To understand which kind of data performed, missing values and how all files are connected.</li>
#   <li>Compare all 4 data leaks to analyze differences and similarities.</li>
#   <li>Next I would like to merge with Automated pipeline as it would good to find automated solution how we represent and describe each dataset. These reports generation also allows us to compare datasets in optimal way.
# </ol>
# 
# Let see data.
# 

# In[605]:


import os
os.chdir(os.path.expanduser("~/Desktop/offshore leaks/csv_bahamas_leaks"))
#os.getcwd()
#Load Bahama leaks. Firstly imort pandas package for downloading analyzing datasets

import pandas as pd
import numpy as np

db1_ent = pd.read_csv("bahamas_leaks.nodes.entity.csv",low_memory=False) 
db1_ent.head()


# In[606]:


db1_ent.count() #we see total rows number is 175 888. Countries information is missing. Company status and and type 
                #is missing too. Let check unique nodes.


# In[607]:


bah_nodes = np.unique(db1_ent['node_id']) #remove duplicates in array
#type(bah_nodes)
len(bah_nodes) == len(db1_ent['node_id']) #Nodes are unique


# In[608]:


bah_names = np.unique(db1_ent['name'])
len(bah_names) #Comp.names not unique for all nodes. Let find repeating names


# In[609]:


#db1_ent['ibcRUC'].value_counts()
db1_ent['jurisdiction'].value_counts() ##sourceID and jurisdiction is the same for all companies and nodes
t=db1_ent['name'].value_counts()
len(t[t[:]>2]) #Companies with more than one node
t[t[:]>2]


# Currently we have identified that there are 175 888 nodes and 75 514 companies. For company KM BM-C-SEVEN LTD. there is the biggest nodes amount, i.e. 12. For 346 companies there are more than one node and for 10 (TOP 10) companies more than 2 nodes. It would be also good to analyze companies names to see if there similar names or in some cases couls be the same company but differs some symbol for example for one case is LTD and for other - LTD. So the first step could be exclude all punctuation from the names and extra spaces.
# 
# We also see one note in dataset.

# In[610]:


print(db1_ent['note'].value_counts())
note_row = pd.notnull(db1_ent['note'])
db1_ent[note_row] #row with non empty note


# Let load others datasets Bahamas leaks dataset and see how we can merge and use them for further analysis.

# In[611]:


db1_adr = pd.read_csv("bahamas_leaks.nodes.address.csv",low_memory=False) 
print(db1_adr.head())
#We see addresses for some nodes. Let merge 2 datasets on node_id. Before we merge them it would be good to check for 
#double nodes to avoid duplication
d1 = db1_ent['node_id'].value_counts()
print(len(d1[d1[:]>1])) #no duplicates

bah_dataset1 = pd.merge(db1_ent, db1_adr, on='node_id')
bah_dataset1.head() #there is no common nodes for both datasets
#nodes = db1_ent['node_id']
#nodes[nodes==20010494]


# In[612]:


#Download intermediary dataset
db1_interm = pd.read_csv("bahamas_leaks.nodes.intermediary.csv",low_memory=False) 
db1_interm.head()


# In[613]:


#Download officer dataset
db1_offic = pd.read_csv("bahamas_leaks.nodes.officer.csv",low_memory=False) 
db1_offic.head()


# In[614]:


#merge datasets to see how they are connected
bah_dataset2 = pd.merge(db1_ent, db1_offic, on='node_id')
bah_dataset2.head() #not connected by node id

#merge datasets to see how they are connected
bah_dataset3 = pd.merge(db1_ent, db1_interm, on='node_id')
bah_dataset3.head()  #not connected

#merge datasets to see how they are connected
bah_dataset4 = pd.merge(db1_offic, db1_interm, on='node_id')
bah_dataset4.head()  #not connected

#All 3 datasets have different nodes


# In[615]:


#Load edges datasets
db1_edges = pd.read_csv("bahamas_leaks.edges.csv",low_memory=False) 
db1_edges.head()


# In[616]:


print(db1_edges['rel_type'].value_counts())
db1_edges['valid_until'].value_counts() ##sourceID and valid_until is the same for all rows


# Now we see that in the edge file we see connected nodes and attributes for each of node. We can see also relations type. In total there are 8 relations type. Now it would be good to create final dataset with connected nodes and all attributes for them. Firstly I'll exclude empty columns (where all values are NaN) and look through the other values in each dataset. For example if variable "valid_until" is the same for all rows I'll take it out. Then merge with node_1 and node_2.

# In[617]:


#Entities dataset
db1_ent2 =  db1_ent[['node_id','name','incorporation_date','ibcRUC']]
db1_ent2.count() 
#db1_adr['jurisdiction'].value_counts()
#db1_ent2['jurisdiction_description'].value_counts()
#db1_ent[['country_codes','address']]


# In[618]:


#Address datasets
db1_adr2 =  db1_adr[['country_codes','countries','node_id','address']]
print(db1_adr2.count())
#db1_adr2['note'].value_counts()


# In[619]:


#Officers dataset
db1_offic2 = db1_offic[['country_codes','countries','node_id','name']]
#Add dummy column to recognize as Officer
db1_offic2 = db1_offic2.assign(Officer='1')
#print(db1_offic.count())
#db1_offic['name'].value_counts()
db1_offic2.count()


# In[620]:


#Intermediary dataset
db1_interm2 = db1_interm[['country_codes','node_id','name']]
#Add new column to identify intermediary
db1_interm2 = db1_interm2.assign(Intermed='1')
#print(db1_interm.count())
#db1_interm['name'].value_counts()
db1_interm2.count()


# Now we can merge datasets. Firstly get addresses for all connected nodes.

# In[621]:


#Need to remain fields to understand that info is about first node (from address DB)
db1_adr2_1 = db1_adr2.add_suffix('_addr_node1')
db1_ent2_1 = db1_ent2.add_suffix('_ent_node1')
db1_offic2_1 = db1_offic2.add_suffix('_offic_node1')
db1_interm2_1 = db1_interm2.add_suffix('_interm_node1')
#At this step we can think about optimize this process. For example we can create a funtion which renames columns
#for data frames.

#Create function 

def merge_datasets_left(data1,data2,l,r,drop):
    dataset = data1.merge(data2, left_on=[l], right_on=[r], how='left')
    dataset = dataset.drop([drop],1)
    return dataset


#db1_adr2_1.head()
#Left join would be used as we are not sure that for each node exist address information

bah_dataset_merge = merge_datasets_left(db1_edges,db1_adr2_1,'node_1','node_id_addr_node1','node_id_addr_node1')
bah_dataset_merge = merge_datasets_left(bah_dataset_merge,db1_ent2_1,'node_1','node_id_ent_node1','node_id_ent_node1')
bah_dataset_merge = merge_datasets_left(bah_dataset_merge,db1_offic2_1,'node_1','node_id_offic_node1','node_id_offic_node1')
bah_dataset_merge = merge_datasets_left(bah_dataset_merge,db1_interm2_1,'node_1','node_id_interm_node1','node_id_interm_node1')
bah_dataset_merge.head()
#To exclude big dataframe, check values 
bah_dataset_merge.count()


#The same will be done for second node
db1_adr2_2 = db1_adr2.add_suffix('_addr_node2')
db1_ent2_2 = db1_ent2.add_suffix('_ent_node2')
db1_offic2_2 = db1_offic2.add_suffix('_offic_node2')
db1_interm2_2 = db1_interm2.add_suffix('_interm_node2')

bah_dataset_merge = merge_datasets_left(bah_dataset_merge,db1_adr2_2,'node_2','node_id_addr_node2','node_id_addr_node2')
bah_dataset_merge = merge_datasets_left(bah_dataset_merge,db1_ent2_2,'node_2','node_id_ent_node2','node_id_ent_node2')
bah_dataset_merge = merge_datasets_left(bah_dataset_merge,db1_offic2_2,'node_2','node_id_offic_node2','node_id_offic_node2')
bah_dataset_merge = merge_datasets_left(bah_dataset_merge,db1_interm2_2,'node_2','node_id_interm_node2','node_id_interm_node2')

print(bah_dataset_merge.shape) #datasets size


# After merging all information it would be good to investigate the whole picture using netwotk ananlysis. For example now we have information about relationships type and about nodes. We also identified officers and intermediaries. It would be interesting to see differences by relationships groups. I am not going to do it on this step, I'll go further and analyze other datasets.

# Now we will load remaining datasets.

# In[622]:


#Offshore leaks
os.chdir(os.path.expanduser("~/Desktop/offshore leaks/csv_offshore_leaks"))
db2_ent = pd.read_csv("offshore_leaks.nodes.entity.csv",low_memory=False)
db2_adr = pd.read_csv("offshore_leaks.nodes.address.csv",low_memory=False)
db2_offic = pd.read_csv("offshore_leaks.nodes.officer.csv",low_memory=False)
db2_interm = pd.read_csv("offshore_leaks.nodes.intermediary.csv",low_memory=False)
db2_edges = pd.read_csv("offshore_leaks.edges.csv",low_memory=False)

#Panama papers
os.chdir(os.path.expanduser("~/Desktop/offshore leaks/csv_panama_papers"))
db3_ent = pd.read_csv("panama_papers.nodes.entity.csv",low_memory=False)
db3_adr = pd.read_csv("panama_papers.nodes.address.csv",low_memory=False)
db3_offic = pd.read_csv("panama_papers.nodes.officer.csv",low_memory=False)
db3_interm = pd.read_csv("panama_papers.nodes.intermediary.csv",low_memory=False)
db3_edges = pd.read_csv("panama_papers.edges.csv",low_memory=False)

#Paradise papers
os.chdir(os.path.expanduser("~/Desktop/offshore leaks/csv_paradise_papers"))
db4_ent = pd.read_csv("paradise_papers.nodes.entity.csv",low_memory=False)
db4_adr = pd.read_csv("paradise_papers.nodes.address.csv",low_memory=False)
db4_offic = pd.read_csv("paradise_papers.nodes.officer.csv",low_memory=False)
db4_interm = pd.read_csv("paradise_papers.nodes.intermediary.csv",low_memory=False)
db4_edges = pd.read_csv("paradise_papers.edges.csv",low_memory=False)
#one additional DB
db4_other =  pd.read_csv("paradise_papers.nodes.other.csv",low_memory=False)


# Analyzing first dataset we got that there is file with connected nodes and then we can find additional information about included nodes, e.g. Country, Name, incorporation date etc.. Firstly I'd like to compare edges files to see how nodes ties are shown in each source. After I'll compare attributes information.
# 
# Comparing edges data it would be important compare relations type, how many nodes "are in the graph", share of officers and intermediaries. The biggest degree of the nodes, centrality measures etc.

# In[623]:



print('Bahamas:' , db1_edges.head())
print('Offshore leaks:' , db2_edges.head())
print('Panama papers:' , db3_edges.head())
print('Paradise papers:' , db4_edges.head())


# Firstly, need to mark that Bahamas data leaks are represented by undirected graph. While other 3 leaks are directed graphs. It is quite important difference. Then we can compare relations type.

# In[624]:


db1_edges['rel_type'].value_counts()
print(db1_edges.groupby(['rel_type']).count())
rel_type1 = pd.DataFrame(db1_edges.groupby(['rel_type'])['rel_type'].count()/db1_edges['rel_type'].count())
rel_type2 = pd.DataFrame(db2_edges.groupby(['TYPE'])['TYPE'].count()/db2_edges['TYPE'].count())
rel_type3 = pd.DataFrame(db3_edges.groupby(['TYPE'])['TYPE'].count()/db3_edges['TYPE'].count())
rel_type4 = pd.DataFrame(db4_edges.groupby(['TYPE'])['TYPE'].count()/db4_edges['TYPE'].count())

print(rel_type1)
print(rel_type2)
print(rel_type3)
print(rel_type4)
import matplotlib.pyplot as plt

#Now I would like to join all relatioships types into one graph to compare percentage ratio. It would be also good 
#to compare absolute values in the same time.


# In[625]:



rel_type1 = rel_type1.rename(columns={"rel_type":"Bahamas"})
rel_type2 = rel_type2.rename(columns={"TYPE":"Offshore leaks"})
rel_type3 = rel_type3.rename(columns={"TYPE":"Panama papers"})
rel_type4 = rel_type4.rename(columns={"TYPE":"Paradise papers"})
#print(rel_type1)
rel_outer = ((rel_type1.join(rel_type2)).join(rel_type3)).join(rel_type4)
print(rel_outer)
rel_outer.plot(y=["Bahamas", "Offshore leaks", "Panama papers","Paradise papers"], kind="bar")


# I'll not pay much attention to the graph design on this step as time is limited. But it would be also good to add 
# values change size. I would also change order of the relationships type to show 3 "the most popular" types.
# 
# From the graph we can see that the most popular relationships types are intermediary of, officer of and registered address. For Bahamas the biggest part of ties are based on intermediary relationship, for Offshore lakes, Panamas papers, Paradise paper - officer of.
# 
# Then we can look at entities datasets.
# 

# In[626]:


print('Bahamas:' , db1_ent.head())
print('Offshore leaks:' , db2_ent.head())
print('Panama papers:' , db3_ent.head())
print('Paradise papers:' , db4_ent.head())


# In[627]:


bah_ent = pd.DataFrame(db1_ent.count()).rename(columns={0:"Bahamas"})
offsh_ent = pd.DataFrame(db2_ent.count()).rename(columns={0:"Offshore_Lakes"})
panam_ent = pd.DataFrame(db3_ent.count()).rename(columns={0:"Panama papers"})
parad_ent = pd.DataFrame(db4_ent.count()).rename(columns={0:"Paradise papers"})
ent_outer = ((bah_ent.join(offsh_ent)).join(panam_ent)).join(parad_ent)
ent_outer


# In the table above we can see differences between 4 datasets for entities. For example there is no country information in Bahamas but we can see it for others datasets. Closed date is only in Paradise papers for some entities. For some entities in Panama and Paradise papers are missing names. In Panama papers and Bahamas no information about companies type.
# 
# We can get the similar picture for others dataset, e.g. addresses, officers and intermediaries. I'll do it for addresses dataset as is may be important for the next task.

# In[628]:


bah_adr = pd.DataFrame(db1_adr.count()).rename(columns={0:"Bahamas"})
offsh_adr = pd.DataFrame(db2_adr.count()).rename(columns={0:"Offshore_Lakes"})
panam_adr = pd.DataFrame(db3_adr.count()).rename(columns={0:"Panama papers"})
parad_adr = pd.DataFrame(db4_adr.count()).rename(columns={0:"Paradise papers"})
adr_outer = ((bah_adr.join(offsh_adr)).join(panam_adr)).join(parad_adr)
adr_outer


# As we can see in all datasets is full information about countries. Next step could be aggregate countries which are involved in all datasets.

# In[629]:


bah_adr = pd.DataFrame(db1_adr.groupby(['country_codes'])['country_codes'].count()).rename(columns={"country_codes":"Bahamas"})
bah_adr = bah_adr.sort_values("Bahamas",ascending = False)

off_adr = pd.DataFrame(db2_adr.groupby(['country_codes'])['country_codes'].count()).rename(columns={"country_codes":"Offshore_Lakes"})
off_adr = off_adr.sort_values("Offshore_Lakes",ascending = False)

pan_adr = pd.DataFrame(db3_adr.groupby(['country_codes'])['country_codes'].count()).rename(columns={"country_codes":"Panama papers"})
pan_adr = pan_adr.sort_values("Panama papers",ascending = False)

par_adr = pd.DataFrame(db4_adr.groupby(['country_codes'])['country_codes'].count()).rename(columns={"country_codes":"Paradise papers"})
par_adr = par_adr.sort_values("Paradise papers",ascending = False)

# We can that many countries are involved, so it would be good to find most most popular countries for each group
#print(bah_adr.head(10))
print(off_adr.head(10))
print(pan_adr.head(10))
print(par_adr.head(10))


# In[630]:


#off_adr.head(10).plot(kind='bar')
bah_adr.head(10).plot(kind='bar')
off_adr.head(10).plot(kind='bar')
pan_adr.head(10).plot(kind='bar')
par_adr.head(10).plot(kind='bar')
#plt.tight_layout()
plt.show()
#plt.subplot(2, 2, 3)
#plt.subplot(2, 2, 4)


# We can see that different countries are involved in each dataset. Only for Bahamas we can highlight the one Bahamas country.

# ## Country coverage and connections.
# In this part I'll analyse countries coverage and country-wise interconnections. I'll take Panama papers data set to make this part simplier. Fistly we need to merge dataset with the "edges" file

# In[631]:



#At the first step I'll take as example one node and look informnation about it in other datasets.
print('From entity dataset: \n', db3_ent[db3_ent['node_id'] == 10000005])
print('From address dataset: \n',db3_adr[db3_adr['node_id'] == 10000005])
print('From officers dataset: \n',db3_offic[db3_offic['node_id'] == 10000005])
print('From intermediary dataset: \n',db3_interm[db3_interm['node_id'] == 10000005])
#db3_ent.head()

#As Graph is quite big I'll analyze connections by relations types.

Pan_rel_offic = db3_edges[db3_edges['TYPE'] == 'officer_of']
Pan_rel_interm = db3_edges[db3_edges['TYPE'] == 'intermediary_of']
Pan_rel_addres = db3_edges[db3_edges['TYPE'] == 'registered_address']

#Find node which include in Officer dataset
Pan_rel_offic.head()
print(Pan_rel_addres.head())
#12000003 to 10024966 
print('From officers dataset: \n',db3_offic[db3_offic['node_id'] == 12000003])
print('From entity dataset: \n', db3_ent[db3_ent['node_id'] == 12000003])
print('From address dataset: \n',db3_adr[db3_adr['node_id'] == 12000003])

#In this step we can conclude that for the relationships officers_of we'll officers database
#relation based on address
#10000035
#print('From officers dataset: \n',db3_offic[db3_offic['node_id'] == 10000035])
print('From entity dataset: \n', db3_ent[db3_ent['node_id'] == 10000035])
#print('From address dataset: \n',db3_adr[db3_adr['node_id'] == 10000035])

#And for relarionships type - registrated address we'll use entities DB to get info about organzation


# To summarize previous results, can conclude that to simplify the process I will separate edges dataset by relationships types (3 types). Then I'll merge nodes with attributes and in the result I am planning to get 1 undirected graph (based on the address relationship type) and 2 directed based on rel.types officers and intermediaries. As our aim is to undesrtand countries connection Graphs will be weighted and as weights would be used number of connections between nodes. 

# In[632]:


Pan_rel_addres.head(10)

#Firstly analyze relations based address. Merge with entity dataset.

#Need to remain fields to understand that info is about first node (from address DB)
#db3_adr2_1 = db3_adr.add_suffix('_addr_node1')
db3_ent2_1 = db3_ent.add_suffix('_ent_node1')
db3_adr2_2 = db3_adr.add_suffix('_adr_node2')

pan_dataset_merge_ent = pd.DataFrame(merge_datasets_left(Pan_rel_addres,db3_ent2_1,'START_ID',
                                                         'node_id_ent_node1','node_id_ent_node1'))

#Reduce dimension to avoid too big dataset. It always worth to pay attention to dates which allows indicate
#entity's activity, how long it appeared and in which period. I'll not go deeper in this step.
pan_dataset_merge_ent_red = pan_dataset_merge_ent[['START_ID','END_ID','country_codes_ent_node1',
                                                   'countries_ent_node1']]

pan_dataset_merge_ent_red = pd.DataFrame(merge_datasets_left(pan_dataset_merge_ent_red,db3_adr2_2,'END_ID',
                                                         'node_id_adr_node2','node_id_adr_node2'))

pan_dataset_merge_ent_red = pan_dataset_merge_ent_red[['START_ID','END_ID','country_codes_ent_node1',
                                                   'countries_ent_node1','name_adr_node2',
                                                   'country_codes_adr_node2','countries_adr_node2']]
pan_dataset_merge_ent_red.head(50)
d = pan_dataset_merge_ent_red
d[(d['country_codes_ent_node1']!=d['country_codes_adr_node2'])].count()
#Countries name always the same or NULL
#pd.crosstab(pan_dataset_merge_ent_red.country_codes_ent_node1,
 #           pan_dataset_merge_ent_red.country_codes_adr_node2, margins=True)
#pan_dataset_merge_ent.count()

###Let analyze officers of and intermediaries of datasets. Firstly join edges set 
#(with relations type - officers of) and officers set

db3_offic2_1 = db3_offic.add_suffix('_off_node1')
db3_ent2_2 = db3_ent.add_suffix('_ent_node2')

pan_dataset_merge_offic = pd.DataFrame(merge_datasets_left(Pan_rel_offic,db3_offic2_1,'START_ID',
                                                         'node_id_off_node1','node_id_off_node1'))

pan_dataset_merge_offic_red = pan_dataset_merge_offic[['START_ID','END_ID',
                                                   'name_off_node1','country_codes_off_node1',
                                                   'countries_off_node1']]
#pan_dataset_merge_offic_red.head()

#add Info about second node (from entities DB)
pan_dataset_merge_offic_red = pd.DataFrame(merge_datasets_left(pan_dataset_merge_offic_red,db3_ent2_2,
                                                              'END_ID','node_id_ent_node2','node_id_ent_node2'))


pan_dataset_merge_offic_red.count()
pan_dataset_merge_offic_red = pan_dataset_merge_offic_red[['START_ID','END_ID',
                                                   'name_off_node1','country_codes_off_node1',
                                                  'countries_off_node1','name_ent_node2','country_codes_ent_node2'
,                                                          'countries_ent_node2']]

#pan_dataset_merge_offic_red.head()

#db4_other.head()


# On this step we have a graph with officers and related company information. Officer - individual and he is connected to entity and as we can see countries can differ. That's why we will make directed weighted graph and try to vizualize how it looks like.

# In[633]:



#Pckages are installed using pip install in command line
import networkx as nx

PanGraph1 = nx.DiGraph()

#Reduce and calculate weights for the graph. Firstly take absolute numbers
 #df.groupby(['Team', 'Pos']).agg({'Age': ['mean', 'min', 'max']})
pan_aggr_countries = pan_dataset_merge_offic_red.groupby([
    'country_codes_off_node1','country_codes_ent_node2']).agg({'START_ID':['count']})
pan_aggr_countries  = pan_aggr_countries.reset_index()

pan_aggr_countries.columns = ['country_off', 'country_ent', 'weight']
#pan_aggr_countries = sorted(pan_aggr_countries)
print(pan_aggr_countries.head())

PanGraph1 = nx.from_pandas_edgelist(pan_aggr_countries, 'country_off', 'country_ent',edge_attr='weight',
                                   create_using=nx.DiGraph())


len(PanGraph1.nodes()) #200 countries involves in the network
print(nx.info(PanGraph1))

#Calculate indegree and outdegree for each node
out_d = PanGraph1.out_degree(weight=True)
in_d = PanGraph1.in_degree(weight=True)

#To see indegrees and outdegrees distribution
in_degrees = [PanGraph1.in_degree(n,weight=True) for n in PanGraph1.nodes()]
out_degrees = [PanGraph1.out_degree(n,weight=True) for n in PanGraph1.nodes()]
bins = np.linspace(0, 80,50)
plt.hist(in_degrees, bins, alpha=0.5, label='in_degree')
plt.hist(out_degrees, bins, alpha=0.5, label='out_degree')
plt.legend(loc='upper right')
plt.show()

#y = np.arange(len(PanGraph1.nodes()))
#plt.bar(y,in_degrees)
#plt.show()

#print(sorted(out_d,key=lambda x: x[1],reverse=True))
print(sorted(out_d,key=lambda x: x[1],reverse=True))
#there we can see 


# Here we can see differences between indegree and outdegree distribution. Indegree is more concentrated to the 
# smaller values. That mean that smaller number of entities with big degree and we can choose more popular of them.  The biggest in-degree is for HKG, CHE, GBR, ARE, JEY.
# 
# The biggest out-degree is for VGB, GBR, PAN, USA, JEY. Out-degree values are more evenly distributed. 
# So the biggest officers amount come from mentioned countries but how they are connected with other countries. We can choose subgraph with these nodes to see how they are conencted with others countries. 
# 
# It would be good to normalize weights for each officer's country. For example, summarize total outdegree for officers's country and then divide weights with other countries. It will help us to find most common ties for each officers'country. We can also calculate for each officer's country outdegree with his home country and foreign to see if it mostly connected with home entities or not. I'll show some examples how it could be done.

# In[634]:



#pan_dataset_merge_ent_red.head(50)
#add_additional column to the main dataset
main = pan_aggr_countries

main = main.assign(home =  (main['country_off'] == main['country_ent']))
#main.head()
main_transf = pd.crosstab(main.country_off, main.home, values=main.weight, aggfunc='sum')
#main_transf/main_transf.sum(axis = 1, skipna = True) 
perc_table = main_transf.apply(lambda x: x / main_transf.sum(axis = 1, skipna = True) )
perc_table.count()
#To be able compare countries we can define function which show countries connections home/foreign together
#with absolute outdegree value which is important to make statistically significant conclusions.

def country_con(code):
    set1 = main_transf[main_transf.index == code]
    set2 = perc_table[perc_table.index == code]
    Total =  set1.sum(axis = 1, skipna = True)

    false= set2[0].values[0]
    true = set2[1].values[0]
    t = Total[0]
    #to make statistically significant conclusions let minimal total amount 30 
    if Total[0] > 30:
        if false > true:
            text = print(code,': ','Connections with foreign countries is stronger than with home in proportion:',false,':',true)
        elif false<true:
            text = print(code,': ','Connections with foreign countries is weaker than with home in proportion:',false,':',true)
        else:
            text = print(code,': ','Connections are equal')
            
    
    else:
        text = print(code,': ','Total number is too small')
        
    return  text, t ,true
        
        
    
#print(perc_table[perc_table.index == 'SWE']) 
#print(main_transf[main_transf.index=='SWE'])
print(country_con('SWE'))
#Let check VGB, GBR, PAN, USA, JEY. 
country_con('VGB')
country_con('GBR')
country_con('PAN')

#PanGraph1.nodes()
#we can also find countries which have stronger ties with home entities rather than foreign
for nodes in PanGraph1.nodes():
    set1 = main_transf[main_transf.index == nodes]
    set2 = perc_table[perc_table.index == nodes]
    Total =  set1.sum(axis = 1, skipna = True)

    false= set2[0].values[0]
    true = set2[1].values[0]
    t = Total[0]
    if (t>30) & (true>0.5):
        print(nodes,'; ',round(true,2))
        
#We can see that the strongest connection with home is HKG,URY, ECU


# Something similar could be done with intermediary dataset. After analysis for each relationship's type done we can compare differencies between types. For example which countries have the biggest degrees depending on relationships type. Then we can also analyze connection home/foreign. For example, for one relationship could be a tendency to have more stronger connection with home. When we do such kind analysis we can also calculate mean or medians and test hypotheses for statistically significant differences between the groups. Unfortunately I don't have a time to realize it now. All these analysis can help to understand "behaviors" between countries and find areas which could be investigated deeper. 

# ## Pipeline with primary analysis for the data leak.
# File name will be primary.py. This script will allow get short summary about input data, like numbers nodes, variables, relationships type etc. One of the best way to save results are excel and of course would be good to create dashboard with graphs and tables. 
# 

# ## Modelling different node types
# In this step graph would be created based on nodes. Nodes could be entities, officers, intermediaries. To build classification model we need identify features which could be used to classify new object. 
# 
# I'll give a brief overview how I would started to build it. As objects are part of the graph, the first step could be to calculate measures like degree(in/out could be useful in this case), centrality, clustering coefficient, betweenness etc.
# 

# In[635]:


from array import array
from scipy.stats import mannwhitneyu
db3_off2_1 = db3_offic.add_suffix('_off_node1')
db3_ent2_2 = db3_adr.add_suffix('_ent_node2')
db3_interm2_1 = db3_interm.add_suffix('_interm_node1')

pan_dataset_merge_ent = pd.DataFrame(merge_datasets_left(db3_edges,db3_off2_1,'START_ID',
                                                         'node_id_off_node1','node_id_off_node1'))

pan_dataset_merge_ent_red = pan_dataset_merge_ent[['START_ID','END_ID','TYPE','country_codes_off_node1',
                                                   'countries_off_node1']]

pan_dataset_merge_ent_red = pd.DataFrame(merge_datasets_left(pan_dataset_merge_ent_red,db3_ent2_2,'END_ID',
                                                         'node_id_ent_node2','node_id_ent_node2'))

pan_dataset_merge_ent_red = pan_dataset_merge_ent_red[['START_ID','END_ID','TYPE','country_codes_off_node1',
                                                   'countries_off_node1','name_ent_node2',
                                                   'country_codes_ent_node2','countries_ent_node2']]

pan_dataset_merge_ent_red = pd.DataFrame(merge_datasets_left(pan_dataset_merge_ent_red,db3_interm2_1,'START_ID',
                                                         'node_id_interm_node1','node_id_interm_node1'))

pan_dataset_merge_ent_red = pan_dataset_merge_ent_red[['START_ID','END_ID','TYPE','country_codes_off_node1',
                                                   'countries_off_node1',
                                                   'country_codes_ent_node2','countries_ent_node2',
                                                      'name_interm_node1','country_codes_interm_node1',
                                                      'countries_interm_node1']]

main_dataset = pan_dataset_merge_ent_red
#in this step assume that if officer address is not null - node is Officer
main_dataset = main_dataset.assign(Off = main_dataset['countries_off_node1'].notnull() )
main_dataset = main_dataset.assign(Ent = main_dataset['countries_ent_node2'].notnull() )
main_dataset = main_dataset.assign(Intermed = main_dataset['countries_interm_node1'].notnull() )

offic_set = main_dataset[main_dataset['Off']== True]
entit_set = main_dataset[main_dataset['Ent']== True]

G_off = nx.from_pandas_edgelist(offic_set, 'START_ID', 'END_ID',create_using=nx.DiGraph()) #source, check outdegree
G_ent = nx.from_pandas_edgelist(entit_set, 'START_ID', 'END_ID',create_using=nx.DiGraph()) #check indegree                                  
len(G_off.nodes()) #200 countries involves in the network
print(nx.info(G_off))
print(nx.info(G_ent))

#From in-degrees and out-degrees we can see that they differs for 2 groups
#Average out degree:   1.0432(officer) and Average in degree:   0.6183(entity) 
#We can check differences usinf hypotheses testing (choosing nodes with not null outdegree)
officers = offic_set['START_ID']
entities = entit_set['END_ID']
#len(officers)

off_degre =  G_off.out_degree(nodes for nodes in officers)
ent_degre =  G_ent.out_degree(nodes for nodes in entities)
from numpy import mean
from numpy import std
from scipy.stats import mannwhitneyu
#print('Officiers: mean=%.3f stdv=%.3f' % (mean(off_degre), std(off_degre)))
#print('Entities: mean=%.3f stdv=%.3f' % (mean(ent_degre), std(ent_degre)))

#can use built tests to check median differencex - do they differ
#=[ off_degre [nodes] for nodes in officers] 

#bins = np.linspace(0, 80,50)
#plt.hist(off_degre, bins, alpha=0.5, label='Off_degree')
##plt.legend(loc='upper right')
#plt.show()


# In[ ]:





# In[636]:




