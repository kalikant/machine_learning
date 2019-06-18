from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import text
from sqlalchemy import func
from sqlalchemy import create_engine, MetaData
import random
from flask import json
from flask import request
import pandas as pd
from pandas.io.json import json_normalize
from yahoofinancials import YahooFinancials
import json
import csv
import pickle


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///TradeIdea.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
engine = create_engine('sqlite:///TradeIdea.sqlite3')

class StockBasicInfo(db.Model):
    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    ticker = db.Column(db.String)
    incepetion_date = db.Column(db.String)

class StockPrices(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ticker = db.Column(db.String)
    formatted_date = db.Column(db.String)
    high = db.Column(db.String)
    low = db.Column(db.String)
    open = db.Column(db.String)
    close = db.Column(db.String)
    adjclose = db.Column(db.String)
    volume = db.Column(db.String)


# This decorator takes the class/namedtuple to convert any JSON data in incoming request to.
def map_json_to_model(class_):
    def wrap(f):
        def decorator(*args):
            obj = class_(**request.get_json())
            return f(obj)
        return decorator
    return wrap


@app.route("/", methods=['GET'])
def home():
    return "Hello Python!"


@app.route('/saveStockBasicInfo', methods=['POST'])
@map_json_to_model(StockBasicInfo)
def save_stock_info(stockBasicInfo):

    db.session.add(stockBasicInfo)
    db.session.commit()
    db.session.flush()

    return 'saveStockBasicInfo saved successfully'


@app.route('/getInceptionDate', methods=['GET'])
def get_inception_date():
    ticker = request.args['ticker']
    ticker = "'" + ticker + "'"
    sql = "SELECT * FROM stock_basic_info WHERE ticker =  {}".format(ticker)
    with engine.connect() as con:
        output =con.execute(sql)

    return output


@app.route('/pullStockPrices', methods=['GET'])
def pull_historical_stocks_prices():
    tickers = ['AAPL','MSFT','GOOGL','IBM','INTC','FB','ORCL']
    from_dates = ['1980-12-12','1986-03-13','2004-08-19','1962-01-02','1980-03-17','2012-05-18','1986-03-12']
    to_date = '2019-06-17'
    for i in range(0,len(tickers)):
        ticker = tickers[i]
        from_date = from_dates[i]
        yf = YahooFinancials(ticker)
        historical_stock_prices = yf.get_historical_price_data(from_date, to_date, 'daily')
        file_name = ticker+'_'+ from_date + '_to_' + to_date + '.json'
        with open(file_name,'w') as f:
            json.dump(historical_stock_prices,f)
            print(file_name + ' has been written successfully')

    return 'success'

@map_json_to_model(StockPrices)
def save_stock_info(stockPrices):

    db.session.add(stockPrices)
    db.session.commit()
    db.session.flush()

    return 'saveStockBasicInfo saved successfully'


@app.route("/initialize", methods=['GET'])
def initialize():
    #metadata = MetaData()
    #engine = create_engine('sqlite:///TradeIdea.sqlite3')
    #metadata.create_all(engine)
    return "TradeIdea DB created successfully"

if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=8080)