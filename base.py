


import pandas as pd

def read_files(file_path, file_name, years, file_type):
    full_path = f"{file_path}{file_name}.{file_type}"
    
    column_specs = [
        (2, 10),
        (10, 12),
        (12, 24),
        (27, 39),
        (56, 69),
        (69, 82),
        (82, 95),
        (108, 121),
        (152, 170),
        (170, 188)
    ]
 
    column_names = [
        'date',
        'codbdi',
        'symbol',
        'company_name',
        'opening_price',
        'highest_price',
        'lowest_price',
        'closing_price',
        'trade_quantity',
        'trade_volume'
    ]

    data_frame = pd.read_fwf(full_path, colspecs=column_specs, names=column_names, skiprows=1)
    
    return data_frame


def filter_stocks(data_frame):
    filtered_df = data_frame[data_frame['codbdi'] == 2]  # Filter by codbdi = 2 (standard lot code)
    filtered_df = filtered_df.drop(['codbdi'], axis=1)  # Drop the codbdi column once we've selected the standard lot stocks
    
    return filtered_df


def parse_date(data_frame):
    data_frame['date'] = pd.to_datetime(data_frame['date'], format='%Y%m%d')
    
    return data_frame


def parse_values(data_frame):
    data_frame['opening_price'] /= 100
    data_frame['highest_price'] /= 100
    data_frame['lowest_price'] /= 100
    data_frame['closing_price'] /= 100
    
    return data_frame


def concat_files(file_path, file_name, years, file_type, final_file):
    data_frame_final = None
    
    for i, year in enumerate(years):
        data_frame = read_files(file_path, file_name, year, file_type)
        data_frame = filter_stocks(data_frame)
        data_frame = parse_date(data_frame)
        data_frame = parse_values(data_frame)
        
        if i == 0:
            data_frame_final = data_frame
        else:
            data_frame_final = pd.concat([data_frame_final, data_frame])
    
    data_frame_final.to_csv(f'{file_path}/{final_file}', index=False)
    
    return data_frame_final

