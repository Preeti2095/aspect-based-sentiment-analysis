import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
root = ET.parse('text_data/Laptop_Train_v2.xml').getroot()
#
doc_dict=[]
for i,value in enumerate(root.findall('sentence')):
    review = value.find('text').text
    doc_dict.append(review)

doc_df = pd.DataFrame(doc_dict,columns=["reviews"])


all_terms=[]
all_polarity=[]
for i in range(len(root)):
    term=[]
    polarity=[]

    if root[i].find('aspectTerms/')!=None:

        for aspect in root[i].findall('aspectTerms/'):
            # print(aspect.attrib['term'])
            # print("aspect ",aspect.attrib['term'])
            term.append(aspect.attrib['term'])
            polarity.append(aspect.attrib['polarity'])
        # if term != None:
        #     print(term)
        all_terms.append(term)
        all_polarity.append(polarity)
    else:
        all_terms.append(np.NAN)
        all_polarity.append(np.NAN)

# print(len(all_terms))
doc_df['Aspect']=all_terms
doc_df['polarity']=all_polarity

# print(doc_df['Aspect'])
# print(len(all_terms))
# print(len(all_polarity))
print(doc_df)
print(doc_df['polarity'].value_counts())
print(doc_df['Aspect'].isnull().sum()/3045)

