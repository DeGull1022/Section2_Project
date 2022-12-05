import pandas as pd
import numpy as np

df = pd.read_csv('players_20.csv')

df1 = df.drop(columns = ['sofifa_id', 'player_url', 'short_name', 'long_name', 'dob', 'nationality', 'club', 'overall', 'value_eur', 'wage_eur', 'preferred_foot', 'international_reputation', 'weak_foot', 'skill_moves', 
                         'real_face', 'release_clause_eur', 'player_tags', 'team_position', 'team_jersey_number', 'loaned_from', 'joined', 'contract_valid_until', 'nation_position', 'nation_jersey_number', 'player_traits',
                         'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb'])

df1['gk_val'] = df1['gk_diving'] + df1['gk_handling'] + df1['gk_kicking']+ df1['gk_reflexes']+ df1['gk_speed']+ df1['gk_positioning']
df1['paces'] = df1['pace'] + df1['shooting'] + df1['passing']+ df1['dribbling']+ df1['defending']+ df1['physic']
df1['attacking_score'] = df1['attacking_crossing'] + df1['attacking_finishing'] + df1['attacking_heading_accuracy'] + df1['attacking_short_passing'] + df1['attacking_volleys']
df1['skill_score'] = df1['skill_dribbling'] + df1['skill_curve'] + df1['skill_fk_accuracy'] + df1['skill_long_passing'] + df1['skill_ball_control']
df1['movement_score'] = df1['movement_acceleration'] + df1['movement_sprint_speed'] + df1['movement_agility'] + df1['movement_reactions'] + df1['movement_balance']
df1['power_score'] = df1['power_shot_power'] + df1['power_jumping'] + df1['power_stamina'] + df1['power_strength'] + df1['power_long_shots']
df1['mentality_score'] = df1['mentality_aggression'] + df1['mentality_interceptions'] + df1['mentality_positioning'] + df1['mentality_vision'] + df1['mentality_penalties'] + df1['mentality_composure']
df1['defending_score'] = df1['defending_marking'] + df1['defending_standing_tackle'] + df1['defending_sliding_tackle']
df1['goalkeeping_score'] = df1['goalkeeping_diving'] + df1['goalkeeping_handling'] + df1['goalkeeping_kicking'] + df1['goalkeeping_positioning'] + df1['goalkeeping_reflexes']

df2 = df1.drop(columns = ['gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning', 
                          'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 
                          'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 
                          'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 
                          'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 
                          'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 
                          'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure', 
                          'defending_marking', 'defending_standing_tackle', 'defending_sliding_tackle', 
                          'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']).fillna(0)

df2['physical_score'] = df2['gk_val'] + df2['paces'] 
df3 = df2.drop(columns = ['gk_val', 'paces'])

df3.work_rate = df3.work_rate.replace({'High/High':6, 'High/Medium':5, 'High/Low':4, 'Medium/High':5, 'Medium/Medium':4, 'Medium/Low':3, 'Low/High':4, 'Low/Medium':3, 'Low/Low':2}).astype('int')

df4 = df3.drop(columns = ['player_positions', 'body_type'])

data = df4


from sklearn.model_selection import train_test_split

target = 'potential'
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns = target), data[target], train_size=0.80, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.80, test_size=0.20, random_state=42)

from category_encoders import OrdinalEncoder
from xgboost import XGBRegressor

encoder = OrdinalEncoder()
X_train_encoded = encoder.fit_transform(X_train) # 학습데이터
X_val_encoded = encoder.transform(X_val) # 검증데이터

boosting = XGBRegressor(
    max_depth = 3,
    mid_child_weight = 10,
    n_estimators= 500,
    objective='reg:squarederror', # default
    learning_rate=0.2,
    n_jobs=-1
)

eval_set = [(X_train_encoded, y_train), 
            (X_val_encoded, y_val)]

boosting.fit(X_train_encoded, y_train, 
          eval_set=eval_set,
          early_stopping_rounds=50
         )

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
y_pred = boosting.predict(X_val_encoded)

print('R^2', r2_score(y_val, y_pred))
MSE = mean_squared_error(y_val, y_pred)
print('RMSE', np.sqrt(MSE))

### 하이퍼 파라미터 검증 start ###
from sklearn.model_selection import GridSearchCV

cv_params = {
            # 'max_depth':np.arange(5, 10, 1),
            'max_depth' : [1,3,5],
            'min_child_weight' : [1,5,10],
            'n_estimators': [100,200,500,1000]
            }

fix_params = {
            'booster' : 'gbtree'
            # 'objective' : 'binary:logistic'
            }

csv = GridSearchCV(XGBRegressor(**fix_params),
                  cv_params, 
                  scoring = 'r2',
                  cv = 5,
                  n_jobs= -1)
csv.fit(X_train, y_train)

df_csv = pd.DataFrame(csv.cv_results_)
print(df_csv)
print(csv.best_score_)
print(csv.best_params_)


df_csv.to_csv('C:\\Users\\blck1\\Desktop\\CodeStates\\Project\\Section2\\df.csv', sep=',', na_rep='NaN') 

y_pred_best = csv.best_estimator_.predict(X_val)
print('R^2_best', r2_score(y_val, y_pred_best))


# 검증 데이터 예측하기
y_pred = csv.predict(X_val)

# 정확도 평가
print('R^2', r2_score(y_val, y_pred))
MSE = mean_squared_error(y_val, y_pred)
print('RMSE', np.sqrt(MSE))
### 하이퍼 파라미터 검증 end ###


### 테스트데이터 검증 start ###
X_test_encoded = encoder.transform(X_test) 
y_pred_test = csv.predict(X_test_encoded)

print('R^2_test', r2_score(y_test, y_pred_test))
MSE = mean_squared_error(y_test, y_pred_test)
print('RMSE_test', np.sqrt(MSE))
### 테스트데이터 검증 end ###