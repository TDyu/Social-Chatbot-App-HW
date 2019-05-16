#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""Chatbot for job.
"""
import re
from datetime import datetime
from dateutil import parser
import json

from flask import Flask
from flask import request
from flask import make_response
app = Flask(__name__)
from bs4 import BeautifulSoup as bs
import requests
import pandas as pd


def crawler_job():
    """Crawling jobs data.

    Returns:
      A pandas.DataFrame representation of the sorted crawling result of jobs by start date.
    """
    fields = []
    details = {}

    # Get the response of target web.
    response = requests.get(
        "http://sdsweb.oit.fcu.edu.tw/FCU_Employment/jobQueryAction.do")

    # Parse the response text.
    soup = bs(response.text, 'lxml')
    # soup = bs(response.text, 'html.parser')

    # Get the job tables.
    job_tables = soup.select('#AutoNumber8')

    # Deal with field titles. In the table id=AutoNumber7 and AutoNumber8. Need to filter out the duplicate.
    # id is AutoNumber7
    for field in soup.select('#AutoNumber7 td[style="BORDER-TOP-WIDTH: 1px; BORDER-LEFT-WIDTH: 1px; BORDER-BOTTOM: 1px solid; BORDER-RIGHT-WIDTH: 1px"]'):
        field = field.text.strip()
        if field != '' and field not in fields:
            fields.append(field)
            details[field] = []
    # id is AutoNumber8
    # Because some fields are not in all job tables, we need to read all tables and check the field title.
    for job_table in job_tables:
        for field in job_table.select('p[style="LINE-HEIGHT: 150%"]'):
            field = field.text.strip()
            if field != '' and field not in fields:
                fields.append(field)
                details[field] = []
    fields.append('發布連結')
    details['發布連結'] = []

    # Deal with the information of job.
    job_count = 0
    for job_table in job_tables:
        # Deal with one job table.
        # Corresponding field title.
        field = ''
        for table_data in job_table.select('td'):
            # Recruitment department and its link.
            if 'bgcolor="#e4e4e4" colspan="2" height="20" width="577"' in str(table_data):
                department = table_data.find('a')
                # [0-9]+: number
                link_num = int(re.findall(r'[0-9]+', str(department))[0])
                link = 'http://sdsweb.oit.fcu.edu.tw/FCU_Employment/jobQueryAction.do#' + str(link_num)
                details['發布連結'].append(link)
                details['徵聘單位'].append(department.find('b').text.strip())

            # Deal with other field information.
            if len(table_data.select('p[style="LINE-HEIGHT: 150%"]')) == 1:
                field = table_data.select(
                    'p[style="LINE-HEIGHT: 150%"]')[0].text.strip()
            if len(table_data.select('p[style="WORD-SPACING: 1px; LINE-HEIGHT: 150%"]')) == 1:
                information = table_data.select(
                    'p[style="WORD-SPACING: 1px; LINE-HEIGHT: 150%"]')[0].text.strip()
                # Deal with listing by blank characters.
                blank_characters = ['\r', '\n', '\t', ' ']
                has_blank = False
                for element in blank_characters:
                    if element in information:
                        has_blank = True
                        break
                if has_blank:
                    information = re.sub(r'(\r|\n|\t)+', '\n', information)
                    information = re.sub(r' {2}', '', information)
                if information.strip() == '':
                    details[field].append('無')
                else:
                    details[field].append(information)
        # If there are not some fields in the job table, need to vacancy.
        job_count += 1
        for key, value in details.items():
            if len(value) < job_count:
                details[key].append('無')

    # Store to DataFrame.
    job_df = pd.DataFrame({
        fields[0]: details[fields[0]],  # 徵聘單位
        fields[1]: details[fields[1]],  # 擬聘職務
        fields[2]: details[fields[2]],  # 預計起聘日期
        fields[3]: details[fields[3]],  # 應徵條件
        fields[4]: details[fields[4]],  # 檢附資料
        fields[5]: details[fields[5]],  # 截止日期
        fields[6]: details[fields[6]],  # 聯絡方式
        fields[7]: details[fields[7]],  # 備註事項
        fields[8]: details[fields[8]],  # 發布連結
    })

    # Format the date.
    date_format = '%Y-%m-%d'
    job_df[fields[2]] = pd.to_datetime(job_df[fields[2]], format=date_format)
    job_df[fields[5]] = pd.to_datetime(job_df[fields[5]], format=date_format)

    # Sort the jobs by the starting date.
    job_df = job_df.sort_values(by=fields[2])

    return job_df


def get_head_job(job_df):
    """Get the head five jobs.

    Args:
        job_df: pandas.DataFrame, The crawling result of job sorted by start date.

    Returns:
        A str representation of result message.
    """
    result = '預計起聘日期最早的前五個工作機會: \n\n'
    result = _organize_result_string(job_df.head())

    return result


def get_filtered_job(job_df, department=None, job_title=None, start=None, end=None, keyword=None, has_keyword=None):
    """Get the jobs according to the date or keyword or department or job_title.

    Args:
        job_df: pandas.DataFrame, The crawling result of job sorted by start date.
        department: str, Requirement department.
        job_title: str, Requirement job title.
        start: datetime.date, Day period starting.
        end: datetime.date, Day period ending.
        keyword: str, Query keyword.

    Returns:
        A str representation of result message.
    """
    # Filter out the date time if it needed.
    if end != None:
        if keyword == '起聘':
            job_df = job_df[(job_df['預計起聘日期'] >= start)
                            & (job_df['預計起聘日期'] <= end)]
        elif keyword == '截止':
            job_df = job_df[(job_df['截止日期'] >= start)
                            & (job_df['截止日期'] <= end)]
    elif start != None:
        if keyword == '起聘':
            job_df = job_df[job_df['預計起聘日期'] == start]
        elif keyword == '截止':
            job_df = job_df[job_df['截止日期'] == start]
    else:
        today = datetime.today().date()
        if keyword == '起聘':
            job_df = job_df[job_df['預計起聘日期'] >= today]
        elif keyword == '截止':
            job_df = job_df[job_df['截止日期'] >= today]

    # Filter out department.
    if department != None:
        job_df = job_df[job_df['徵聘單位'] == department]

    # Filter out job title.
    if job_title != None:
        job_df = job_df[job_df['擬聘職務'].str.contains(job_title)]
        if job_title == '助理':
            job_df = job_df[~job_df['擬聘職務'].str.contains('助理教授')]

    # Filter out keyword without time limiting.
    if keyword != None and has_keyword != None and keyword in job_df:
        if has_keyword:
            job_df = job_df[job_df[keyword] != '無']
        else:
            job_df = job_df[job_df[keyword] == '無']

    result = '根據條件的工作機會: \n'
    result += _organize_result_string(job_df)

    return result


def _organize_result_string(job_df):
    result = ''
    index = 0
    for job in job_df.iterrows():
        # for index, job in job_df.iterrows():
        job = job[1]
        
        result += str(index + 1) + ': 【' + job['徵聘單位'] + '-' + job['擬聘職務'] + '】\n'
        result += ' - 預計起聘日期: ' + job['預計起聘日期'].strftime('%Y-%m-%d') + '\n'
        result += ' - 截止日期: ' + job['截止日期'].strftime('%Y-%m-%d') + '\n'
        if job['應徵條件'] != '無':
            conditions = job['應徵條件'].split('\n')
            result += ' - 應徵條件:\n'
            for condition in conditions:
                if condition != '':
                    result += '\t' + condition + '\n'
        else:
            result += ' - 應徵條件: 無\n'
        if job['檢附資料'] != '無':
            conditions = job['檢附資料'].split('\n')
            result += ' - 檢附資料:\n'
            for condition in conditions:
                if condition != '':
                    result += '\t' + condition + '\n'
        else:
            result += ' - 檢附資料: 無\n'
        if job['聯絡方式'] != '無':
            conditions = job['聯絡方式'].split('\n')
            result += ' - 聯絡方式:\n'
            for condition in conditions:
                if condition != '':
                    result += '\t' + condition + '\n'
        else:
            result += ' - 備註事項: 無\n'
        if job['備註事項'] != '無':
            conditions = job['備註事項'].split('\n')
            result += ' - 備註事項:\n'
            for condition in conditions:
                if condition != '':
                    result += '\t' + condition + '\n'
        else:
            result += ' - 備註事項: 無\n'
        result += ' - 發布連結: ' + job['發布連結'] + ')'

        if index != len(job_df) - 1:
            result += '\n'
        index += 1
    
    result = result.strip()
    if result != '':
        return result
    else:
        return '無'


@app.route('/')
def verify():
    return 'Hello world', 200


@app.route('/webhook', methods=['POST'])
def webhook():
    """Deal with webhook of dialogflow.
    """
    # Get the request from dialogflow.
    webhook_request = request.get_json(silent=True, force=True)

    # Take some items from request data.
    # query_action = webhook_request.get('queryResult').get('action')
    query_action = webhook_request['queryResult']['action']
    query_text = webhook_request['queryResult']['queryText']
    parameter_any = webhook_request['queryResult']['parameters']['any']
    parameter_date = webhook_request['queryResult']['parameters']['date']
    parameter_period = webhook_request['queryResult']['parameters']['date-period']
    print(webhook_request)

    start = None
    end = None
    if parameter_date != '':
        start = parser.parse(parameter_date).date()
    elif parameter_period != '' and parameter_period != []:
        start = parser.parse(parameter_period['startDate']).date()
        end = parser.parse(parameter_period['endDate']).date()

    # Organize response.
    response = {}
    result = ''
    if query_action == 'ask_job':
        parameter_job_title = webhook_request['queryResult']['parameters']['job_title']
        parameter_department = webhook_request['queryResult']['parameters']['department']
        # Crawling job.
        job_df = crawler_job()
        # Query job with department.
        if parameter_department != '' and parameter_job_title == '' and start == None and end == None and parameter_any == '':
            print('Query job with department.')
            result = get_filtered_job(job_df, department=parameter_department)
        # Query job with department and time and keyword.
        elif parameter_department != '' and parameter_job_title == '' and (start != None or end != None) and parameter_any != '':
            print('Query job with department and time and keyword.')
            result = get_filtered_job(
                job_df, department=parameter_department, start=start, end=end, keyword=parameter_any)
        # Query job with job title.
        elif parameter_department == '' and parameter_job_title != '' and start == None and end == None and parameter_any == '':
            print('Query job with job title.')
            result = get_filtered_job(job_df, job_title=parameter_job_title)
        # Query job with job title and time and keyword.
        elif parameter_department == '' and parameter_job_title != '' and (start != None or end != None) and parameter_any != '':
            print('Query job with job title and time and keyword.')
            result = get_filtered_job(
                job_df, job_title=parameter_job_title, start=start, end=end, keyword=parameter_any)
        # Query job with time and keyword.
        elif parameter_department == '' and parameter_job_title == '' and (start != None or end != None) and parameter_any != '':
            print('Query job with time and keyword.')
            result = get_filtered_job(
                job_df, start=start, end=end, keyword=parameter_any)
        # Query job with keyword.
        elif parameter_department == '' and parameter_job_title == '' and start == None and end == None and parameter_any != '':
            print('Query job with keyword.')
            if '不' in query_text or '沒' in query_text:
                result = get_filtered_job(
                    job_df, keyword=parameter_any, has_keyword=False)
            else:
                result = get_filtered_job(
                    job_df, keyword=parameter_any, has_keyword=True)
        # Get five head jobs.
        else:
            result = get_head_job(job_df)
    response = {
        'fulfillmentText': result,
        'source': 'agent'
    }
    response = json.dumps(response, indent=4)
    webhook_response = make_response(response)
    webhook_response.headers['Content-Type'] = 'application/json'

    return webhook_response


if __name__ == '__main__':
    app.run(port=5000)
