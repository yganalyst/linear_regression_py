import pandas as pd
import numpy as np
import statsmodels.api as sm

def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 0.05, 
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step(전진선택법)
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        # 1개 속성씩 넣어서(1차 회귀) P-value 계산
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min() # 최소 P-value 추출
        if best_pval < threshold_in: # 임계점 보다 작으면,
            best_feature = new_pval.idxmin() # 최소 P-value를 갖는 속성을 선택
            included.append(best_feature) # 리스트에 추가
            changed=True
            if verbose:
                print('Add  %s with p-value %s' % (best_feature, round(best_pval,3)))
                
        # backward step(후진제거법)
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:] # 절편을 제외한 모든 P-value 추출
        worst_pval = pvalues.max() # 최대 P-value 추출
        if worst_pval > threshold_out:# 임계점 보다 크면(threshold_out)
            changed=True
            worst_feature = pvalues.idxmax() # 가장 p-value가 큰 속성 추출
            included.remove(worst_feature) # 리스트에서 제외
            if verbose:
                print('Drop %s with p-value %s' % (worst_feature, round(worst_pval,3)))
        if not changed: # 변화가 없으면 중단
            break
    print("Final Attribute :", included)
    model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
    print(model.summary())
    return included, model
    
def backward_selection(X, y, 
                       threshold_out = 0.05, 
                       verbose=True):
    included = list(X.columns)
    while True:
        changed=False
        # backward step(후진제거법)
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:] # 절편을 제외한 모든 P-value 추출
        worst_pval = pvalues.max() # 최대 P-value 추출
        if worst_pval > threshold_out: # 임계점 보다 크면(threshold_out),
            changed=True
            worst_feature = pvalues.idxmax() # 가장 p-value가 큰 속성 추출
            included.remove(worst_feature) # 리스트에서 제외
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed: # 변화가 없으면 중단
            break
    print("Final Attribute :", included)
    model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
    print(model.summary())    
    return included, model

def forward_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.05, 
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step(전진선택법)
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        # 1개 속성씩 넣어서(1차 회귀) P-value 계산
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min() # 최소 P-value 추출
        if best_pval < threshold_in: # 임계점 보다 작으면,
            best_feature = new_pval.idxmin() # 최소 P-value를 갖는 속성을 선택
            included.append(best_feature) # 리스트에 추가
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval)) # 추가된 속성과 P-value 정보 print
        if not changed: # 변화가 없으면 중단
            break
    print("Final Attribute :", included)    
    model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
    print(model.summary())
    return included, model
