import webbrowser

try:
    import ccxt
except ImportError:
    ccxt = None


def abrir_navegador(url="https://pancakeswap.finance/prediction?token=BNB"):
    """
    Abre o navegador padrão com a URL especificada.
    
    Args:
        url (str): A URL a ser aberta no navegador.
    """
    # Altere o caminho para o executável do seu navegador se necessário
    # chrome_path = 'C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe %s'

    webbrowser.open(url)
    return


def get_bnb_usd_price():
    if ccxt is None:
        return None
    try:
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker("BNB/USDT")
        return ticker['last']  # último preço negociado
    except Exception as e:
        print("Erro ao buscar preço:", e)
        return None
