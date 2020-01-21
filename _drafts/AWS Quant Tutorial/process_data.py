import 
for i in range(15):
        new_stock_df = stock_df.copy()
        stock_df = pd.concat([stock_df, new_stock_df])
    
    print("*****************************************",stock_df.shape)