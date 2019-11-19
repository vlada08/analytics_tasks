
# coding: utf-8

# In[44]:


#Panama papers
import os
import pandas as pd
import xlsxwriter
os.chdir(os.path.expanduser("~/Desktop/offshore leaks/csv_panama_papers"))
db3_ent = pd.read_csv("panama_papers.nodes.entity.csv",low_memory=False)
db3_adr = pd.read_csv("panama_papers.nodes.address.csv",low_memory=False)
db3_offic = pd.read_csv("panama_papers.nodes.officer.csv",low_memory=False)
db3_interm = pd.read_csv("panama_papers.nodes.intermediary.csv",low_memory=False)
db3_edges = pd.read_csv("panama_papers.edges.csv",low_memory=False)

Pan_rel_offic = db3_edges[db3_edges['TYPE'] == 'officer_of']
Pan_rel_interm = db3_edges[db3_edges['TYPE'] == 'intermediary_of']
Pan_rel_addres = db3_edges[db3_edges['TYPE'] == 'registered_address']

db3_off2_1 = db3_offic.add_suffix('_off_node1')
db3_ent2_2 = db3_adr.add_suffix('_ent_node2')
db3_interm2_1 = db3_interm.add_suffix('_interm_node1')

def merge_datasets_left(data1,data2,l,r,drop):
    dataset = data1.merge(data2, left_on=[l], right_on=[r], how='left')
    dataset = dataset.drop([drop],1)
    return dataset

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

pan_dataset_merge_ent_red.head(50)
d = pan_dataset_merge_ent_red
s1 = d.count()
s1.rename(columns = {0:'Count'})
s1


# In[43]:


# Create an new Excel file and add a worksheet.
writer = pd.ExcelWriter('data_leak.xlsx', engine='xlsxwriter')
workbook = xlsxwriter.Workbook()


s1.to_excel(writer, sheet_name='Sheet1',startcol=2,header = True)



writer.save()
# Get the xlsxwriter objects from the dataframe writer object.
#workbook  = writer.book
#worksheet = writer.sheets['Sheet1']
#bold = workbook.add_format({'bold': 1})

#workbook.write('A1', 'Panama papers data leak: Summary',bold)
#workbook.write('A2', 'Variables', bold)
#workbook.write('B2', d.count() , bold)

 
#df1.to_excel(writer, sheet_name='Sheet1')  # Default position, cell A1.
#df2.to_excel(writer, sheet_name='Sheet1', startcol=3)
#df3.to_excel(writer, sheet_name='Sheet1', startrow=6)

# It is also possible to write the dataframe without the header and index.
#df4.to_excel(writer, sheet_name='Sheet1',
#             startrow=7, startcol=4, header=False, index=False)


# Create a Pandas dataframe from the data.
#df = pd.DataFrame([10, 20, 30, 20, 15, 30, 45])

# Create a Pandas Excel writer using XlsxWriter as the engine.
#writer = pd.ExcelWriter('simple.xlsx', engine='xlsxwriter')
#df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.

# Write some numbers, with row/column notation.
#worksheet.write(2, 0, 123)
#worksheet.write(3, 0, 123.456)

#workbook.close()

