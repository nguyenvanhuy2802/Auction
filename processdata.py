import pandas as pd 
import numpy as np 
from numpy import isnan
import matplotlib as plt

# from model import SVMModel,DecisionTreeModel,RandomModel,LinearModel,Model

# df= pd.read_csv('dmc2006_train.csv')
# #7998 rows and 26 columns
# # khi in ra ta có thể thấy đc trong 2 cột là listing_subtitle và gms có chứa những giá trị mising
# # trong đó listing_subtitle : 6948  ~ chiếm 86,86% dữ liệu  của listing_subtitle
# #           gms             : 118 ~ chiếm 1.4 % dữ liệu cửa gms



# label = df.iloc[:,-1] # all class label
# data = df.iloc[:,:-1] # all data 
# # get all columns has NaN value 
# df_nan = df.columns[df.isna().any()]
# #get all columns hasn't NaN value 
# # df_dropnan = df.dropna(axis=1)

# #get columns 
# categorical_cols = data.select_dtypes(exclude=np.number) # => all the columns need to process


# #=> trong tập dữ liệu có 2 trường là văn bản là item_leaf_category_name ,listing_title,listing_subtitle 
# #=> ta sẽ endcode 2 trường này sữ dụng kĩ thuật vector văn bản 
# doc_dtypes_col = categorical_cols[['item_leaf_category_name','listing_title','listing_subtitle']]

# #=> có 2 trường là dạng date 
# # 6/24/05
# # 8/13/05
# # 9/8/05
# # 10/14/05
# df['listing_start_date'] = pd.to_datetime(df['listing_start_date'], dayfirst=True)
# df['listing_end_date'] = pd.to_datetime(df['listing_end_date'], dayfirst=True)
# #=> chuyển các cột dữ liệu sang datetime


# #=> các trường còn lai là Y , N 
# # Lọc ra những cột chỉ chứa dữ liệu 'Y' và 'N'
# filtered_columns = categorical_cols.columns[categorical_cols.apply(lambda x: x.isin(['Y', 'N'])).all()]
# for col in filtered_columns:
#   df[col] = df[col].astype('category')
# #=> chuyển sang category 

# lưu df đã cấu hình dtypes sang pickle để đọc nhan  hơn
# df.to_pickle("dmc2006_train_pickle.pickle")

# # # đọc lên từ pickle file
# df_pickle= pd.read_pickle('dmc2006_train_pickle.pickle')
# # print(df_pickle.dtypes)

# # => xử lí giá trị missing trước 
# #xử lí các dữ liệu nan thành các số  0  rồi sau đó thay thế bằng các model trong gms
# df_pickle['gms'] = df_pickle['gms'].fillna(0.0)
# #xử lí các dữ liệu nan thành các chuối rỗng rồi sau đó thay thế bằng các model sau khi chuyển hóa vector trong listing_subtitle
# df_pickle['listing_subtitle']=df_pickle['listing_subtitle'].fillna(' ')

# #=> save to a pickle file is not have mising value 
# df_pickle.to_pickle("dmc2006_train_pickle_not_missing.pickle")



#=======================================================================

# # đọc lên từ pickle file
# df_pickle_not_missing= pd.read_pickle('dmc2006_train_pickle_not_missing.pickle')
# # print(df_pickle.dtypes)


# # chuyển hóa các object thành vector suwe dụng CountVectorizer của sklearn.feature_extraction.text
# # lấy ra các object 
# object_dtypes_col=df_pickle_not_missing.select_dtypes(include=['object']).columns

# # print(object_dtypes_col.iloc[:,:1])
# from sklearn.feature_extraction.text import TfidfVectorizer

# # initialize the TfidfVectorizer object
# tfidf_vectorizer = TfidfVectorizer()

# #Vector all object 
# #initialize a emrty dataframe to hold all vector

# for col in object_dtypes_col:
# # # use TfidfVectorizer
#   X_tfidf = tfidf_vectorizer.fit_transform(df_pickle_not_missing[col])
# #   # df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
#   vector_list=X_tfidf.toarray().tolist()
#   series = pd.Series(vector_list)
#   df_pickle_not_missing[col] = series


# # encode value Y N to 1 0 
# # get all columns has categorical type
# categorical_type_col = df_pickle_not_missing.select_dtypes(include=['category']).columns

# for col in categorical_type_col:
#   df_pickle_not_missing[col] = df_pickle_not_missing[col].map({'Y': 1, 'N': 0})
#   df_pickle_not_missing[col] = df_pickle_not_missing[col].astype('int64')

# # # save to a pickle file is not have mising value and encode value to numeric
# df_pickle_not_missing.to_pickle("pickle_encode.pickle")

#==========================================
# data_train=pd.read_pickle('./processDocument/pickle_encode.pickle') # => đây là dữ liệu đã xử lí giwof xử dụng để train model

# #choose feature s


# # print(data_train.applymap(lambda x : isinstance(x,(list,dict,set,tuple))).any())

# # process data vector to average of  sum
# invalid_columns = data_train.columns[data_train.applymap(lambda x: isinstance(x, (list, dict, set, tuple))).any()]
# for col in invalid_columns:
#   data_train[col] = data_train[col].apply(lambda x: np.mean(x))

# data_train['days_from_reference']  =  (data_train['listing_end_date'] - data_train['listing_start_date'] ) .dt.days
# data_train = data_train.drop(columns=['listing_end_date','listing_start_date'])
# data_train.insert(0, 'days_from_reference', data_train.pop('days_from_reference'))
# data_train = data_train.apply(lambda x : x.astype(float) if x.dtype == 'int64' else x )
# feature=data_train.iloc[:,:-1]
# label = data_train.iloc[:,-1].astype(int) 
# data_train["gms_greater_avg"] = label
# data_train.to_pickle('train_data_processed.pickle')

# ====================================================================================================== hoàn tất xử lí dữ liệu 
