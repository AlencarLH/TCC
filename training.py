
import pandas as pd
from base import to_csv
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pypfopt import risk_models
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

#especificando os parametros utilizados para a chamada do arquivo de pre-processamento
def load_data():

    years = ['2018'] # apenas incluir os outros anos na lista
    path = '/root/Documents/'
    type_file = 'TXT'
    final_file = 'dados.csv'
  
    dfs = []
    for year in years:
        name_file = year
        df = to_csv(path, name_file, [year], type_file, final_file)
        dfs.append(df)

    merged_df = pd.concat(dfs)
    return merged_df


def calculate_returns(df_asset):
    df_asset['valor_fechamento_anterior'] = df_asset['valor_fechamento'].shift(1)
    df_asset['daily_returns'] = (df_asset['valor_fechamento'] - df_asset['valor_fechamento_anterior']) / df_asset['valor_fechamento_anterior']
    # df_asset.dropna(inplace=True)
    df_asset['daily_returns'].fillna(method='ffill', inplace=True)  # preenchendo valores nulos pelo valor mais recente
    
    return df_asset


def filtering_assets(df, sigla):
    df_asset = df[df['sigla'] == sigla].copy()

    df_asset['pregao'] = pd.to_datetime(df_asset['pregao'], format='%Y-%m-%d')
    df_asset['mm5d'] = df_asset.groupby('sigla')['valor_fechamento'].rolling(5).mean().reset_index(0, drop=True)
    df_asset['mm21d'] = df_asset.groupby('sigla')['valor_fechamento'].rolling(21).mean().reset_index(0, drop=True)
    df_asset['valor_fechamento'] = df_asset.groupby('sigla')['valor_fechamento'].shift(-1)
    df_asset.dropna(inplace=True)
    df_asset = df_asset.reset_index(drop=True)

    if df_asset.empty:
        print(f"No data available for asset {sigla}. Skipping feature selection.")
        return df_asset, None

    df_asset = calculate_returns(df_asset)

    return df_asset


def select_best_features(df_asset):
    features = df_asset.loc[:, ['qtd_negocios', 'valor_max', 'valor_min', 'mm21d']]
    labels = df_asset['valor_fechamento']
    features_list = ('valor_abertura', 'valor_max', 'valor_min', 'qtd_negocios', 'volume', 'mm5d', 'mm21d')
    k_best_features = SelectKBest(k='all')
    k_best_features.fit_transform(features, labels)
    k_best_features_scores = k_best_features.scores_
    raw_pairs = zip(features_list[1:], k_best_features_scores)
    ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))
    k_best_features_final = dict(ordered_pairs[:15])
    best_features = k_best_features_final.keys()
    print("----------------------------------------------------------------------------")
    print('')
    # print("Melhores features:")
    # print(k_best_features_final)
    return features, labels


def train_neural_network(features, labels):
    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(features)
    rn = MLPRegressor(max_iter=5000)
    rn.fit(X_train_scale, labels)
    valor_novo = features[-50:]
    previsao = scaler.transform(valor_novo)
    pred = rn.predict(previsao)
    
    return pred


def train_random_forest(features, labels):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(features)
    
    rf = RandomForestRegressor()
    rf.fit(X_train_scaled, labels)

    valor_novo = features[-50:]
    previsao = scaler.transform(valor_novo)
    pred = rf.predict(previsao)

    return pred

def train_svm(features, labels):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(features)

    svm = SVR()
    svm.fit(X_train_scaled, labels)

    valor_novo = features[-50:]
    previsao = scaler.transform(valor_novo)
    pred = svm.predict(previsao)

    return pred


def get_optimal_weights(cov_matrix, expected_returns_rf, expected_returns_nn, expected_returns_svm):
    ef_rf = EfficientFrontier(expected_returns_rf, cov_matrix)
    ef_nn = EfficientFrontier(expected_returns_nn, cov_matrix)
    ef_svm = EfficientFrontier(expected_returns_svm, cov_matrix)

    risk_free_rate = 0.03
    #target_return = 0.05

    weights_rf = ef_rf.max_sharpe(risk_free_rate)
    weights_nn = ef_nn.max_sharpe(risk_free_rate)
    weights_svm = ef_svm.max_sharpe(risk_free_rate)

    print()
    print("RANDOM FOREST: Pesos calculados para cada ação:.")
    print()
    print(weights_rf)

    print()
    print("RANDOM FOREST: Pesos calculados para cada ação:.")
    print()
    print(weights_nn)

    print()
    print("SVM: Pesos calculados para cada ação:.")
    print()
    print(weights_svm)
    


def printing_predictions(asset, features, labels, rf_pred, nn_pred, svm_pred):
    df = load_data()
    df_asset = df[df['sigla'] == asset]
    pregao_full = df_asset['pregao']
    pregao = pregao_full[-50:]
    res_full = df_asset['valor_fechamento']
    res = res_full[-50:]
    
    df = pd.DataFrame({'pregao': pregao, 'valor_real': res, 'previsao_rf': rf_pred, 'previsao_nn': nn_pred, 'previsao_svr': svm_pred})
    df.set_index('pregao', inplace=True)
    
    mse_rf = mean_squared_error(df['valor_real'], df['previsao_rf'])
    mse_nn = mean_squared_error(df['valor_real'], df['previsao_nn'])
    mse_svm = mean_squared_error(df['valor_real'], df['previsao_svr'])

    
    print('')
    print(asset)
    print(df)
    print()
    print("Random Forest MSE: ", mse_rf)
    print("Neural Network MSE: ", mse_nn)
    print("SVR MSE: ", mse_svm)

   


def main():
    df = load_data()

    assets = ['BBDC4', 'BBAS3', 'BPAC11', 'BRSR6', 'SANB11', 'BIDI4', 'ITUB4']
    
    # Empresas separadas por setor
    #assets = [
    # Energético
    #'PETR4', 'EQTL3', 'ELET6', 'ENGI11', 'BRDT3',
    # Materiais
    #'VALE3', 'GGBR4', 'BRKM5',
    # Industrial
    #'EMBR3', 'AZUL4', 'GOLL4',
    # Utilidade Pública
    #'CCRO3', 'CPFE3', 'EGIE3', 'TIET11', 'CMIG4', 'CSMG3',
    # Saúde
    #'HAPV3', 'SULA11', 'FLRY3', 'GNDI3', 'RADL3',
    # Financeiro
    #'ITUB4', 'BBDC4', 'B3SA3', 'BBAS3', 'SANB11',
    # Consumo Cíclico
    #'MGLU3', 'VVAR3', 'LAME4', 'CVCB3', 'BTOW3',
    # Consumo Não-cíclico
    #'ABEV3', 'BRFS3',
    # Tecnologia da Informação
    #'HYPE3', 'LINX3',
    # Serviços de Comunicação
    #'JBSS3', 'TOTS3', 'VIVT4', 'TIMP3',
    # Imobiliário
    #'MULT3', 'BRML3', 'CYRE3'
    #]

    # Lista para armazenar resultados de todas as ações
    results = pd.DataFrame(columns=['Asset', 'Features', 'Prediction'])
    # Dicionario usado para armazenar os daily returns
    returns_dict = {}
    returns_df = pd.DataFrame()
    expected_returns_list_rf = []
    expected_returns_list_nn = []
    expected_returns_list_svm = []
    bbdc4_plot = []

    for asset in assets:
        
        df_asset = filtering_assets(df, asset)

        features, labels = select_best_features(df_asset)

        # atribuindo os daily returns da ação current
        df_asset = calculate_returns(df_asset)
        returns_dict[asset] = df_asset['daily_returns'].values

        # treinando a rede com as features e labels encontradas
        rf_pred = train_random_forest(features, labels)
        nn_pred = train_neural_network(features, labels)
        svm_pred = train_svm(features, labels)


        rounded_returns_rf = [round(val, 6) for val in rf_pred.tolist()]
        rounded_returns_nn = [round(val, 6) for val in nn_pred.tolist()]
        rounded_returns_svm = [round(val, 6) for val in svm_pred.tolist()]
        

        expected_returns_rf = np.array(rounded_returns_rf)
        expected_returns_nn = np.array(rounded_returns_nn)
        expected_returns_svm = np.array(rounded_returns_svm)


        expected_returns_list_rf.append(expected_returns_rf)
        expected_returns_list_nn.append(expected_returns_nn)
        expected_returns_list_svm.append(expected_returns_svm)


        # mostrando as previsões
        printing_predictions(asset, features, labels, rf_pred, nn_pred, svm_pred)

        # Criando uma nova linha contendo as ações, features e previsões para cada uma
        row = {'Asset': asset, 'Features': features.columns.tolist(), 'Random Forest Prediction': rf_pred.tolist(), 'Neural Netwok MLP Regressor Prediction': nn_pred.tolist(), 'SVM': svm_pred.tolist()}
        results = results.append(row, ignore_index=True)
 
        returns = df_asset['daily_returns'].tolist()
        returns_df[asset] = df_asset['daily_returns']
        
        # print(f"Daily Returns de {asset}:")
        # print(returns)
        print()

        
    cov_matrix = risk_models.sample_cov(returns_df)
    
    print("Covariance Matrix:")
    print(cov_matrix)

    expected_returns_rf = pd.Series([np.mean(asset_returns) for asset_returns in expected_returns_list_rf], index=assets)
    expected_returns_nn = pd.Series([np.mean(asset_returns) for asset_returns in expected_returns_list_nn], index=assets)
    expected_returns_svm = pd.Series([np.mean(asset_returns) for asset_returns in expected_returns_list_svm], index=assets)


    #print()
    #print("Expected returns - media para a Random Forest: ")
    #print(expected_returns_rf)
    #print("Expected returns - media para a Rede Neural: ")
    #print(expected_returns_nn)
    #print("Expected returns - media para a Rede Neural: ")
    #print(expected_returns_svm)

    get_optimal_weights(cov_matrix, expected_returns_rf, expected_returns_nn, expected_returns_svm)

    # covariance matrix
    #fig, ax = plt.subplots(figsize=(16,14))
    #cax = ax.matshow(cov_matrix, cmap='coolwarm')
    #fig.colorbar(cax)
#
    #
    #ax.set_xticklabels([''] + list(range(len(cov_matrix))))
    #ax.set_yticklabels([''] + list(range(len(cov_matrix))))
#
    #plt.show()
    

if __name__ == '__main__':
    main()
