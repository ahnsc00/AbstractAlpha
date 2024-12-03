from gym_trading_env.downloader import download
import os
import datetime

def create_coin_data():
    api_key = os.getenv('')
    api_secret = os.getenv('7ZUZVCMVXI3PIOS5')

    client = Client(api_key, api_secret)
    print('-- Account accessed --')

    # get latest price from Binance API
    btc_price = client.get_symbol_ticker(symbol="BTCUSDT")

    api_key = os.getenv('')
    api_secret = os.getenv('')
    client = Client(api_key, api_secret)

    btc_price = client.get_symbol_ticker(symbol="BTCUSDT")

    # 코인 최초 거래일, 밀리초(millisecond)를 초(second)로 변환
    timestamp_initial = client._get_earliest_valid_timestamp('BTCUSDT', '1d')
    timestamp_initial = timestamp_initial / 1000.
    timestamp = client._get_earliest_valid_timestamp('BTCUSDT', '1d')

    print('\n', ' 이 코인의 바이낸스 최초 거래일')
    print('timestamp_initial :', timestamp_initial)

    # 시간 변환 : timestamp --> UTC --> KST
    utc_binan_BTC_init = datetime.utcfromtimestamp(timestamp_initial)
    kst_binan_BTC_init = utc_binan_BTC_init + timedelta(hours=9)

    # Binance API로 불러오는 시계열 데이터 : ohlcv 형태의 캔들 데이터
    now = datetime.now()
    bars = client.get_historical_klines('BTCUSDT', '1h', '2017-01-10 00:00:00', now.strftime('%Y-%m-%d %H:%M:%S'),
                                        limit=1000)

    # Pandas DataFrame으로 저장
    btc_df = pd.DataFrame(bars)
    btc_df = pd.DataFrame(bars, columns=['time_open', 'open', 'high', 'low', 'close', 'volume', 'time_close', 'vol_qa',
                                         'n_trades', 'coin_qa', 'USDT_qa', 'ign'])

    btc_df['time_open'] = pd.to_datetime(btc_df['time_open'], unit='ms')
    btc_df['time_close'] = pd.to_datetime(btc_df['time_close'], unit='ms')

    columns_to_convert = ['open', 'close', 'high', 'low', 'volume']
    for column in columns_to_convert:
        btc_df[column] = btc_df[column].astype(float)

def create_coin_data():
    download(exchange_names=["binance"],
                 symbols=["BTC/USDT"],
             timeframe="15m",
             dir="../data",
             since=datetime.datetime(year=2018, month=1, day=1))
create_coin_data()
