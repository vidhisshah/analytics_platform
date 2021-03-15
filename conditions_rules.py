def add_col_vals(df, new_col_name, cols_to_add):
    #Don't let add null value
    df[new_col_name] = sum(df[i] for i in cols_to_add)

def sub_col_vals(df, new_col_name, cols_to_subtr):
    df[new_col_name] = df[cols_to_subtr[0]]- sum(df[i] for i in cols_to_subtr[1:])

def mul_col_vals(df, new_col_name, cols_to_mul):
	return "WIP"

def div_col_vals(df, new_col_name, cols_to_div):        
	return "WIP"