# %%
import pandas as pd
import duckdb
import os

# %%
df = pd.read_csv("/datasets/raw_data/walmart_cleaned.csv")
# drop column Unnamed: 0
df = df.drop(columns=["Unnamed: 0"])
# %%
# query df using duckdb to get the top departments
# Initialize a DuckDB connection
con = duckdb.connect()

# Register the DataFrame as a DuckDB table
con.register("base_table", df)


# %%
con.execute(
    """select distinct dept, sum(Weekly_Sales) as total_sales 
            from base_table
            group by dept
            order by total_sales desc
            limit 10"""
).df()
# %%
top_depts_df = con.execute(
    """
with top_dept as (
    select distinct dept, 
    sum(Weekly_Sales) as total_sales
    from base_table
    group by dept
    order by total_sales desc
    limit 10)
select distinct b.dept, 
a.date as wk_date, 
sum(Weekly_Sales) as total_sales,
avg(Temperature) as avg_temp
from base_table a join top_dept b
on a.dept = b.dept
group by b.dept, wk_date"""
).df()
# %%
top_depts_df.columns = ["dept", "wk_date", "total_sales", "avg_temp"]
# %%
#convert week date to datetime
top_depts_df["wk_date"] = pd.to_datetime(top_depts_df["wk_date"])
# %%
top_depts_df['day_of_week'] = top_depts_df['wk_date'].dt.day_name()
# %%
top_depts_df.to_csv('forecasting_dept_data.csv', index=False)
# %%
