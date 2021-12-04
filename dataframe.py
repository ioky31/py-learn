%%%%%%%%%%%%%%%%%%%%%%% 保存为excel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from pandas.core.frame import DataFrame
df = DataFrame(ps)  # 将字典转换成为数据框
writer = pd.ExcelWriter('p_value.xlsx')
df.to_excel(writer, index=False, encoding='utf-8', sheet_name='Sheet1')
# df_test.to_excel(writer, index=False, header=X_test.columns.values.tolist(), encoding='utf-8', sheet_name='Sheet1')

writer.save()

%%%%%%%%%%%%%%%%%%%%%%% dataframe拼接 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)  # 将df2数据与df1合并


%%%%%%%%%%%%%%%%%%%%%%% dataframe索引 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 下标索引
.iloc()
