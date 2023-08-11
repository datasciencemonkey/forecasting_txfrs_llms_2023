import gradio as gr
from datetime import date, datetime
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
from babel.numbers import format_currency
from rich import print
import re
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate
import plotly.express as px
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
from datetime import date
import duckdb

df = pd.read_csv("data/actuals_and_forecasts.csv")
df["sales"] = df["sales"].astype(float).round(2)


class Tags(BaseModel):
    intent: str = Field(
        ...,
        enum=["plot", "get_data", "summary/aggregate", "other"],
        description="describes how intent of the utterance. User intent can be to either get a plot or get data. other is for statements that do not fall into any of the two categories",
    )
    ctype: str = Field(
        ...,
        description="describes if the user is interested in actuals or forecasts",
        enum=["actuals", "forecasts", "not_mentioned"],
    )
    dept_name: str = Field(
        ...,
        enum=[
            "Cereal",
            "Accessories",
            "Dinner Foods",
            "Bedding",
            "Appliances",
            "Canned Foods",
            "Dairy",
            "Beverages",
            "Cleaning Supplies",
            "Beauty",
            "OTHER",
        ],
        description="describes which department the utterance is about. If not in list, it is OTHER",
    )
    date_type: Optional[str] = Field(
        None,
        description="date of the statement",
        enum=["week", "month", "year"],
    )
    date_entity: Optional[str] = Field(
        None,
        description="The date entity in the utterance in the format MM/DD/YYYY if date_type is week, MM/YYYY if date_type is month and YYYY if date_type is year",
    )


parser_llm = ChatOpenAI(temperature=0, model="gpt-4")
chain = create_tagging_chain_pydantic(Tags, parser_llm)


def _generate_where_clause(res: Tags):
    clause = ""
    if res.ctype == "actuals":
        clause = "where ctype = 'actuals'"
    elif res.ctype == "forecasts":
        clause = "where ctype = 'forecasts'"
    elif res.ctype == "not_mentioned":
        clause = "where ctype in ('actuals','forecasts')"
    if res.dept_name != "OTHER":
        clause += f" and dept_name = '{res.dept_name}'"
    if res.date_type is not None:
        if res.date_type == "week":
            clause += f" and datepart('week', wk_date::DATE) = {res.date_entity}"
        elif res.date_type == "month":
            clause += f" and date_trunc('month', wk_date::DATE) = '{datetime.strptime(res.date_entity, '%m/%Y').strftime('%Y-%m-%d')}'"
        elif res.date_type == "year":
            clause += f" and datepart('year', wk_date::DATE) = {res.date_entity}"
    return clause


def _get_data(clause: str):
    query = f"""
    select * from df {clause}
    """
    print(query)
    res_df = duckdb.query(query).df()
    print(res_df)
    return res_df


def perform_action(res: Tags, message=""):
    if res.dept_name == "OTHER" and res.ctype == "not_mentioned":
        return "Sorry, I don't understand that. Please try again.",None, None
    else:
        clause = _generate_where_clause(res)
        summary_df = _get_data(clause)
        summary_df = summary_df.sort_values(by="wk_date")
        result = f"""Total {summary_df.ctype.unique()[0] if summary_df.ctype.nunique() ==1 else "actuals & forecasts" } for the period between 
            {summary_df.wk_date.min()} and {summary_df.wk_date.max()} 
            for {summary_df.dept_name.unique()[0] if summary_df.dept_name.nunique()==1 else "categories"} is {format_currency(summary_df.sales.sum().round(2),'USD', locale='en_US')} 
            with an average of {format_currency(summary_df.sales.mean().round(2),'USD', locale='en_US')}. 
            The highest {summary_df.ctype.unique()[0]} 
            is {format_currency(summary_df.sales.max(),'USD', locale='en_US')} for week {summary_df.wk_date[summary_df.sales.idxmax()]} 
            and the lowest is {format_currency(summary_df.sales.min(),'USD', locale='en_US')} for week {summary_df.wk_date[summary_df.sales.idxmin()]}"""
        data_plot = px.line(
            summary_df, x="wk_date", y="sales", color="dept_name", title=message
        )
        return re.sub(r"\s+", " ", result).strip(),data_plot,summary_df


def chatter(message: str):
    try:
        print(message)
        res = chain.run(message)
        print(res.json(indent=2))
        return perform_action(res=res, message=message)
    except Exception as e:
        print(e)
        return "Sorry, I don't understand that. Please try again."


with gr.Blocks(theme="soft") as demo:
    gr.Markdown("""#### LUI for Forecasting Demo
                This demo shows how to use LUI for Forecasting to ask questions about the data.
                The data is from 2012 for a ficiticious retail store.""")
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Ask your question here!")
            submit_button = gr.Button(value="Ask!", label="Submit")
            examples = [
                "What are the actuals for Beverages in Jan 2012?",
                "What are the forecasts for Dairy in 2012?",
                "Summarize all the data for Cereal in 2012",
            ]
            gr.Examples(examples=examples, inputs=[input_text], label="Examples")
        with gr.Column():
            outputs = [
                gr.Markdown(label="Summary"),
                gr.Plot(label="Chart"),
                gr.Dataframe(label="Result Set"),
            ]
    submit_button.click(chatter, inputs=[input_text], outputs=outputs)

if __name__ == "__main__":
    demo.launch()
